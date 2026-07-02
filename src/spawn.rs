//! Delegation: hand a self-contained subtask to an isolated sub-agent that runs
//! its own loop and reports back a structured result. The sub-agent gets a fresh
//! context (that is the whole point — the caller does not carry its intermediate
//! steps) and cannot delegate further.

use std::cell::Cell;
use std::rc::Rc;

use serde_json::{json, Value};

use crate::agent_loop::{run, Ctx, RunConfig, DEFAULT_SUB_MAX_ITER};
use crate::job::{Effort, FailureKind, Job, JobResult, Persistence};
use crate::policy::sub_policy;
use crate::{registry, thread};

/// Build a job from the tool arguments and run it as a sub-agent.
pub fn handle(ctx: &Ctx, parent_depth: usize, parent_tid: Option<&str>, input: &Value) -> JobResult {
    let task = input["task"].as_str().unwrap_or("").trim().to_string();
    if task.is_empty() {
        return JobResult::blocked(&crate::job::new_id(), FailureKind::AmbiguousRequest, 0);
    }

    // Fan-out cap: a single top-level run may only delegate so many sub-agents.
    let f = ctx.fanout.get();
    if f >= ctx.max_fanout {
        let id = crate::job::new_id();
        return JobResult::partial(&id, "fan-out limit reached".into(), FailureKind::BudgetExceeded, 0);
    }
    ctx.fanout.set(f + 1);

    let checks = input["checks"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let persistence = Persistence::parse(input["persistence"].as_str());
    let max_iter = input["max_iter"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_SUB_MAX_ITER);
    let mut job = Job::new(task, checks, persistence, max_iter);

    // Effort is per-run: a sub-agent can be given its own level via the tool
    // call; when omitted it inherits the parent's, so a high-effort planner's
    // delegated execution steps don't silently drop back to no reasoning. Stored
    // on the job (not just passed to Ctx) so a durable resume reconstructs the
    // same level via job_to_input rather than silently reverting to None.
    let effort = input
        .get("effort")
        .and_then(|v| v.as_str())
        .map(|s| Effort::parse(Some(s)))
        .unwrap_or(ctx.effort);
    job.effort = effort;

    // Durable only takes effect when the parent itself is persisted, since resume
    // is driven from the parent's registry.
    let durable = persistence == Persistence::Durable && parent_tid.is_some();
    let child_tid = if durable {
        Some(thread::child_thread_id(parent_tid.unwrap(), &job.id))
    } else {
        None
    };

    if durable {
        registry::append_issued(ctx.paths, parent_tid.unwrap(), &job);
    }

    // The sub-agent runs with its own fresh budget — independent of the caller's.
    let child_ctx = Ctx {
        client: ctx.client,
        provider: ctx.provider,
        paths: ctx.paths,
        model: ctx.model,
        effort,
        budget: Rc::new(Cell::new(ctx.sub_budget)),
        sub_budget: ctx.sub_budget,
        fanout: ctx.fanout.clone(),
        max_fanout: ctx.max_fanout,
    };
    let result = run(
        &child_ctx,
        RunConfig {
            job: job.clone(),
            policy: sub_policy(),
            depth: parent_depth + 1,
            thread_id: child_tid,
        },
    );

    if durable {
        registry::append_result(ctx.paths, parent_tid.unwrap(), &result);
    }

    result
}

/// Reconstruct the tool arguments that would produce this job — used to rebuild a
/// delegation round when resuming persisted work.
pub fn job_to_input(job: &Job) -> Value {
    json!({
        "task": job.task,
        "checks": job.checks,
        "persistence": match job.persistence {
            Persistence::Ephemeral => "ephemeral",
            Persistence::Durable => "durable",
        },
        "effort": match job.effort {
            Effort::None => "none",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        },
        "max_iter": job.max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Status;
    use crate::llm::LlmClient;
    use crate::provider::{Anthropic, Provider};
    use crate::thread::Paths;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    struct ScriptedClient {
        responses: RefCell<Vec<Value>>,
        seen_effort: RefCell<Vec<Effort>>,
    }
    impl ScriptedClient {
        fn new(responses: Vec<Value>) -> ScriptedClient {
            ScriptedClient {
                responses: RefCell::new(responses),
                seen_effort: RefCell::new(vec![]),
            }
        }
    }
    impl LlmClient for ScriptedClient {
        fn call(
            &self,
            _p: &dyn Provider,
            _m: &str,
            _msgs: &mut Value,
            _s: &str,
            _t: &Value,
            effort: Effort,
        ) -> Result<Value, String> {
            self.seen_effort.borrow_mut().push(effort);
            let mut q = self.responses.borrow_mut();
            if q.is_empty() {
                Err("no scripted response".into())
            } else {
                Ok(q.remove(0))
            }
        }
    }

    fn mk_ctx<'a>(
        client: &'a dyn LlmClient,
        provider: &'a dyn Provider,
        paths: &'a Paths,
        budget: usize,
        max_fanout: usize,
        fanout: Rc<Cell<usize>>,
    ) -> Ctx<'a> {
        Ctx {
            client,
            provider,
            paths,
            model: "m",
            effort: Effort::None,
            budget: Rc::new(Cell::new(budget)),
            sub_budget: budget,
            fanout,
            max_fanout,
        }
    }

    #[test]
    fn empty_task_is_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, &paths, 100, 8, Rc::new(Cell::new(0)));
        let r = handle(&c, 0, None, &json!({"task": "  "}));
        assert_eq!(r.status, Status::Blocked);
    }

    #[test]
    fn fanout_cap_returns_partial_without_running() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![]);
        let provider = Anthropic;
        let fanout = Rc::new(Cell::new(2));
        let c = mk_ctx(&client, &provider, &paths, 100, 2, fanout);
        let r = handle(&c, 0, None, &json!({"task": "do"}));
        assert_eq!(r.status, Status::Partial);
        assert_eq!(r.failure, Some(FailureKind::BudgetExceeded));
    }

    #[test]
    fn ephemeral_sub_runs_and_returns_result() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"sub done"}]})]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, &paths, 100, 8, Rc::new(Cell::new(0)));
        let r = handle(&c, 0, None, &json!({"task": "compute"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.output.as_deref(), Some("sub done"));
        // No registry writes for an ephemeral sub.
        assert!(registry::load(&paths, "any").is_empty());
    }

    #[test]
    fn durable_sub_records_issued_and_result() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"durable done"}]})]);
        let provider = Anthropic;
        let c = mk_ctx(&client, &provider, &paths, 100, 8, Rc::new(Cell::new(0)));
        let r = handle(&c, 0, Some("main"), &json!({"task":"long","persistence":"durable"}));
        assert_eq!(r.status, Status::Success);
        let recs = registry::load(&paths, "main");
        assert_eq!(recs.len(), 2); // issued + result
        assert!(registry::in_flight(&recs).is_empty());
    }

    #[test]
    fn sub_budget_is_independent_of_parent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![
            json!({"content":[{"type":"tool_use","id":"s1","name":"run_shell","input":{"command":"true"}}]}),
            json!({"content":[{"type":"text","text":"sub done"}]}),
        ]);
        let provider = Anthropic;
        // Parent's own budget is exhausted, but the sub-agent gets its own allowance
        // and runs to completion — no tree-wide pool couples them.
        let c = Ctx {
            client: &client,
            provider: &provider,
            paths: &paths,
            model: "m",
            effort: Effort::None,
            budget: Rc::new(Cell::new(0)),
            sub_budget: 50,
            fanout: Rc::new(Cell::new(0)),
            max_fanout: 8,
        };
        let r = handle(&c, 0, None, &json!({"task": "work"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(r.steps_taken, 1); // sub spent its own budget, not the parent's
        assert_eq!(c.budget.get(), 0); // parent counter untouched
    }

    #[test]
    fn job_to_input_round_trips_fields() {
        let mut job = Job::new("t".into(), vec!["c".into()], Persistence::Durable, 7);
        job.effort = Effort::High;
        let v = job_to_input(&job);
        assert_eq!(v["task"], "t");
        assert_eq!(v["persistence"], "durable");
        assert_eq!(v["effort"], "high");
        assert_eq!(v["max_iter"], 7);
    }

    #[test]
    fn effort_defaults_to_parent_when_omitted() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let mut c = mk_ctx(&client, &provider, &paths, 100, 8, Rc::new(Cell::new(0)));
        c.effort = Effort::High;
        let r = handle(&c, 0, None, &json!({"task": "compute"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(client.seen_effort.borrow().as_slice(), &[Effort::High]);
    }

    #[test]
    fn effort_override_wins_over_parent() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path().to_path_buf());
        let client = ScriptedClient::new(vec![json!({"content":[{"type":"text","text":"done"}]})]);
        let provider = Anthropic;
        let mut c = mk_ctx(&client, &provider, &paths, 100, 8, Rc::new(Cell::new(0)));
        c.effort = Effort::High;
        let r = handle(&c, 0, None, &json!({"task": "compute", "effort": "low"}));
        assert_eq!(r.status, Status::Success);
        assert_eq!(client.seen_effort.borrow().as_slice(), &[Effort::Low]);
    }
}
