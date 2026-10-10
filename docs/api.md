# Python API reference

Import the public API from `safety_chores_rl`. **API version 1** provides live
episodes, isolated rollout workers, budgeted planners, and serializable results.
Native dependencies load only when a native runtime is selected.

## Quick start

This complete example evaluates two actions in the portable, one-step
`ChoiceRuntime`, then executes the higher-reward action. Save it as a Python file
and run it after `python -m pip install -e .`.

```python
from safety_chores_rl import (
    RuntimeSpec, Episode, RolloutSession,
    RolloutBudget, BestOfN, reward_return,
)


def main():
    spec = RuntimeSpec.create("safety_chores_rl.toy:ChoiceRuntime")
    with Episode(spec, max_steps=1) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        plan = BestOfN(session, objective=reward_return).plan(
            root, [(0,), (7,)], sample_seeds=(0,),
            rollout_budget=RolloutBudget(max_steps=1),
        )
        if plan.status == "ok":
            transition = session.execute_first(env, plan)
            print(transition.action, transition.reward)  # 7 7.0


if __name__ == "__main__":
    main()
```

Use the `__main__` guard because rollout workers start fresh Python processes.
For Safety-CHORES, follow the [native setup and reproduction note](native.md).

## RuntimeSpec

```python
RuntimeSpec(factory: str, config_json: str = "{}")
RuntimeSpec.create(factory: str, **config) -> RuntimeSpec
```

Immutable runtime configuration, shared by live episodes and rollout workers.
[Source](https://github.com/tumeteor/safety-chores-rl/blob/main/src/safety_chores_rl/types.py).

| Parameter | Description |
| --- | --- |
| `factory` | Importable `"package.module:Factory"`; the factory receives configuration as keyword arguments. |
| `config_json` | JSON object, normalized into a stable representation. Use `create()` to pass ordinary keyword arguments instead. |

**Property:** `identity: str` hashes the API version, factory, and configuration.
Factories must be importable in a fresh interpreter; pass configuration rather
than live simulator objects or closures. Changing runtime code or artifacts
requires a new session and an updated configuration identity.

## Episode

```python
Episode(spec: RuntimeSpec, *, task_id=0, seed=0, max_steps=200)
```

Owns and resets the live runtime. Use as a context manager to close it reliably.
[Source](https://github.com/tumeteor/safety-chores-rl/blob/main/src/safety_chores_rl/runtime.py).

| Parameter | Default | Description |
| --- | --- | --- |
| `spec` | Required | Runtime factory and frozen configuration. |
| `task_id` | `0` | Nonnegative task identifier understood by the runtime. |
| `seed` | `0` | Nonnegative episode seed. |
| `max_steps` | `200` | Positive episode action cap. |

| Member | Returns | Behavior |
| --- | --- | --- |
| `step(action=None)` | `Transition` | Execute an action ID, or let the policy choose when `None`. Update actions, cost, and completion. |
| `matches(root)` | `bool` | Compare episode identity, prefix, cost, and runtime snapshot with a captured root. |
| `close()` | `None` | Close the runtime; repeated calls are safe. |
| `observation` | `dict` | Current public planning observation. |
| `actions`, `cost`, `done` | `list[int]`, `float`, `bool` | Executed prefix, cumulative cost, and episode completion. |

`step()` raises `RuntimeError` after close or completion. A runtime error during
an action marks the episode done because the simulator may already have moved.

## RolloutSession

```python
RolloutSession(spec: RuntimeSpec, *, workers=1, cache_size=4096)
```

Persistent process pool with session-owned roots and exact-request caching.
[Source](https://github.com/tumeteor/safety-chores-rl/blob/main/src/safety_chores_rl/session.py).

| Parameter | Default | Description |
| --- | --- | --- |
| `spec` | Required | Must match the live episode's specification. |
| `workers` | `1` | Positive number of isolated worker processes. |
| `cache_size` | `4096` | Maximum cached results; `0` disables reuse storage. |

Use a context manager. Concurrent operations on the same session raise
`RuntimeError`; use separate sessions for concurrent callers. `cancel()` may be
called from another thread.

### capture_root

```python
session.capture_root(episode: Episode) -> Root
```

Capture the prefix, accumulated cost, public observation, and validation snapshot
without advancing the live episode. The episode must be live, nonterminal, and
use the same `RuntimeSpec`.

**Raises:** `ValueError` for an incompatible, completed, or closed episode.
Roots are immutable, session-owned handles. Keep the returned object; copies,
deserialized roots, foreign roots, and released roots are rejected.

### rollout

```python
session.rollout(
    root, candidate, *, sample_seed=0, continuation=True,
    budget: RolloutBudget, search_budget=None, reuse=True,
) -> RolloutResult
```

Reset and replay the root in a worker, execute the candidate, then optionally
sample a policy continuation. The live episode remains unchanged.

| Parameter | Description |
| --- | --- |
| `root` | A registered `Root` from this session. |
| `candidate` | Nonempty iterable of nonnegative integer action IDs. |
| `sample_seed` | Integer in `[0, 2**32)`, applied to policy randomness after the entire candidate and its feedback. |
| `continuation` | `False` runs only the candidate; `True` continues the policy within the horizon. |
| `budget` | Candidate plus continuation horizon and per-request limits. Must fit the remaining episode cap. |
| `search_budget` | Optional `SearchBudget`; `None` uses its defaults. |
| `reuse` | Allow exact valid-result cache hits. |

**Returns:** one `RolloutResult`, including invalid outcomes such as timeout or
root mismatch. Simulator randomness is not reseeded at the continuation boundary.

### rollout_batch

```python
session.rollout_batch(
    root, candidates, *, sample_seeds=(0,), continuation=True,
    budget: RolloutBudget, search_budget=None, ledger=None, reuse=True,
) -> tuple[RolloutResult, ...]
```

Evaluate every candidate/seed pair. Parameters follow `rollout()`, with nonempty
`candidates` and `sample_seeds` iterables. Results are ordered **sample-major,
then candidate order**. Exact duplicate requests are coalesced. `ledger` is an
advanced hook for sharing the internal budget accounting used by planners;
ordinary callers should pass `search_budget`.

### Root inspection and lifecycle

| Method | Returns | Behavior |
| --- | --- | --- |
| `validate_root(root, *, timeout_seconds=60.0)` | `RolloutResult` | Independently reset/replay and validate the root; bypass the cache. |
| `inspect_candidates(root, *, timeout_seconds=60.0, ledger=None)` | `RolloutResult` | Run optional `candidate_preview()` in a disposable replay episode. |
| `release_root(root)` | `None` | Release a registered root and its cached results. |
| `capabilities()` | `dict` | Report API/runtime identity, worker/cache counts, and reconstruction capabilities. |
| `cancel()` | `None` | Interrupt the active batch; later calls can restart workers. |
| `close()` | `None` | Stop workers and clear roots and caches. |

Only valid results enter the bounded LRU cache. Its key includes the runtime,
root, candidate, seed, continuation flag, and rollout budget. A cache hit has
`cache_hit=True` and `native_steps=0`. Reuse is local to the session.

### execute_first

```python
session.execute_first(episode, plan) -> Transition
```

Check the live root, execute `plan.action`, and release the root and its cached
results after the action returns. Raises `StalePlanError` if the root or episode
has changed, or `ValueError` if no action is proposed. Inspect `plan.status`
before deciding to execute a partially evaluated plan.

## Budgets

### RolloutBudget

```python
RolloutBudget(
    max_steps: int, timeout_seconds=60.0,
    stop_on_cost=True, record_trace=False,
)
```

| Parameter | Description |
| --- | --- |
| `max_steps` | Positive candidate-plus-continuation horizon; excludes root replay. |
| `timeout_seconds` | Positive finite per-request timeout. |
| `stop_on_cost` | Stop when accumulated cost rules out zero-cost success. |
| `record_trace` | Save execution signatures; requires runtime `execution_signature()`. |

### SearchBudget

```python
SearchBudget(
    max_rollouts=128, max_native_steps=20000,
    wall_seconds=60.0, max_candidates=1024,
)
```

| Parameter | Description |
| --- | --- |
| `max_rollouts` | Maximum dispatched rollout requests. |
| `max_native_steps` | Primitive actions, **including root-prefix replay**. |
| `wall_seconds` | Search wall-time budget, including worker startup and reset. |
| `max_candidates` | Bound on candidate enumeration. |

Counts must be positive integers; time must be positive and finite. The coordinator
reserves worst-case work before dispatch and returns unused reservations after
early completion. Errors/timeouts are charged conservatively. Cleanup may extend
observed runtime beyond the deadline; action counts exclude rendering and inference.

## Planners

All planners return a `Plan`. Candidate actions are nonempty integer tuples;
sample seeds must be distinct. `admissible(observation, candidate)` optionally
filters candidates using public information.
[Source](https://github.com/tumeteor/safety-chores-rl/blob/main/src/safety_chores_rl/search.py).

### BestOfN

```python
BestOfN(session, *, objective=safe_completion, max_cost=None)
planner.plan(
    root, candidates, *, sample_seeds=(0, 1, 2, 3),
    rollout_budget, search_budget=SearchBudget(),
    continuation=True, admissible=None,
) -> Plan
```

Evaluate all candidates on the same seeds; maximize mean objective with stable
lexicographic tie-breaking. A candidate with invalid, unresolved, or over-limit
samples has no score. `max_cost` filters observed total cost, including the prefix.

### FirstSuccess

```python
FirstSuccess(session)
planner.plan(
    root, candidates, *, sample_seeds=(0, 1, 2, 3),
    rollout_budget, search_budget=SearchBudget(), admissible=None,
) -> Plan
```

Preserve caller candidate priority and stop at the first fully valid worker wave
containing a completed zero-cost success. Forces `stop_on_cost=True` and
`record_trace=True`. The returned `witness` holds the entire successful trajectory;
`plan.candidate` holds its full action sequence. Failures remain in `evaluations`.
Invalid evidence, cancellation, or exhausted budget produces no executable witness.
The stopping rule is a feasibility search, not a success-rate estimator.

### SuccessiveHalving

```python
SuccessiveHalving(
    session, *, objective=safe_completion, max_cost=None,
    initial_samples=1, reduction_factor=2,
)
planner.plan(
    root, candidates, *, sample_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
    rollout_budget, search_budget=SearchBudget(),
    continuation=True, admissible=None,
) -> Plan
```

Start with `initial_samples`, retain the strongest fraction of candidates, and
multiply the sample budget by `reduction_factor` each round. The factor must be
at least 2. `usage` includes rounds and exhaustive-rollout count. This approximate
ranking procedure provides no confidence or optimality guarantee.

### BeamSearch

```python
BeamSearch(
    session, *, depth=3, width=4, branching_factor=8,
    heuristic=reward_return, max_cost=0.0,
)
planner.plan(
    root, expander, *, rollout_budget,
    search_budget=SearchBudget(), sample_seed=0, admissible=None,
) -> Plan
```

Score prefix-only rollouts and keep a bounded frontier. `expander` is an iterable
of action IDs or a bounded callable `expander(observation, prefix)` returning
those IDs. `heuristic(result)` supplies the score. Each prefix reconstructs from
the root; `max_cost=0` prunes observed violations. `usage` reports expanded prefixes
and completed depth. Depth, width, and branching factor must be positive integers.

### RecedingHorizon

```python
RecedingHorizon(session, planner, *, fallback=None)
controller.step(episode, candidates, **options) -> tuple[Plan, Transition | None]
```

Capture a root, call `planner.plan(..., **options)`, and execute its first action.
Release the root afterward. If there is no proposed action, return `(plan, None)`
or call the supplied `fallback(observation, plan)` to choose an action. A fallback
must not mutate the live episode. For custom handling of cancellation or budget
exhaustion, call the planner and execution method separately.

### Objectives

| Function | Returns | Interpretation |
| --- | --- | --- |
| `safe_completion(result)` | `float` or `None` | Resolved safe success as 0/1; invalid or unresolved evidence returns `None`. |
| `reward_return(result)` | `float` or `None` | Sum of observed rewards; invalid evidence returns `None`. |

A custom objective accepts `RolloutResult` and returns a finite scalar or `None`.
Observed finite-horizon return does not estimate eventual task completion.

## First-success and checked execution

```python
session.execute_trajectory(episode, plan, *, on_step=None) -> tuple[Transition, ...]
```

Execute a complete traced, zero-cost successful witness. Verify the live root,
then check signatures before and after each action, exact reward/cost, and episode
completion. `on_step(episode, transition, index, matched)` is an optional observer
with a zero-based index; it is followed by another state check.

| Condition | Result |
| --- | --- |
| Live episode or root changed | Raise `StalePlanError`. |
| Missing/incomplete witness, nonzero cost, or unsuccessful result | Raise `ValueError`. |
| Observed state/transition diverges | Raise `TrajectoryMismatch`; stop before further actions. |
| Whole sequence matches | Return the executed transitions. |

Once execution begins, the root and cache are released on completion or failure.
Observers should only record evidence. Divergence cannot undo an action already
executed. Native signatures include camera, exposed physics/cost, policy-history,
and source fields, but exclude the proposal's forked policy RNG; initial root
validation still includes it.

## Data types

These records are frozen dataclasses. `API_VERSION` is `1`.

### Root

Created by `capture_root()`; retain the session's registered instance.

| Fields / properties | Meaning |
| --- | --- |
| `owner`, `episode_id`, `runtime_id`, `identity` | Session, episode, runtime, and root identities. |
| `task_id`, `seed`, `episode_limit` | Original episode configuration. |
| `prefix`, `step`, `cost` | Executed action tuple, its length, and accumulated cost. |
| `observation` | Public planning dictionary decoded from `observation_json`. |
| `snapshot_json` | Serialized runtime validation state. |

A root is a reconstruction handle, not a portable hidden-state checkpoint.

### Transition

```python
Transition(
    action: int, reward: float, cost: float,
    terminated=False, truncated=False, success=None,
)
```

Reward and cost must be finite; cost must be nonnegative. `success` is `bool | None`;
`True` requires termination. Action IDs must be nonnegative integers.

### RolloutResult

| Fields / properties | Type | Meaning |
| --- | --- | --- |
| `request_id`, `root_id`, `candidate`, `sample_seed` | Identifiers | Request provenance and candidate prefix. |
| `status`, `stop_reason`, `valid` | `str`, `str`, `bool` | Validity and stopping condition. |
| `actions`, `rewards`, `costs` | Tuples | Returned per-step evidence. |
| `root_cost`, `total_cost` | `float` | Prefix cost; prefix plus branch costs. |
| `terminated`, `truncated` | `bool` | Task termination and horizon/limit information. |
| `success`, `safe_success` | `bool` or `None` | Task completion and zero-cost completion, when resolved. |
| `replay_steps`, `native_steps`, `elapsed_seconds` | Counts/time | Reconstruction work, charged actions, and elapsed seconds. |
| `endpoint`, `checks_json`, `error` | Evidence | Public endpoint dictionary, serialized checks, optional error. |
| `trace`, `cache_hit`, `schema_version` | Metadata | Root hash plus one hash per action, reuse flag, schema version. |

`to_dict()` returns a JSON-serializable record with decoded `endpoint` and `checks`.
Runtime errors charge the reserved action upper bound; partial traces contain
only transitions that returned successfully. Invalid results are never cached.

### Plan and CandidateEvaluation

| Record | Fields / properties |
| --- | --- |
| `Plan` | `root_id`, optional `candidate`, `status`, `evaluations`, `usage`, `method`, optional `witness`. `action` is the first proposed action or `None`; `to_dict()` serializes the record. |
| `CandidateEvaluation` | `candidate`, optional mean `score`, optional `standard_error`, `samples`, and the underlying `results`. |

Inspect status and witness availability before execution. Budget exhaustion can
leave a partially evaluated plan; an absent action requires an explicit caller
decision or configured fallback.

## Status and exceptions

| Result status / stop reason | Interpretation |
| --- | --- |
| `valid` / `terminal` | Runtime terminated; inspect `success`. |
| `valid` / `horizon` or `prefix_complete` | Observed prefix only; eventual success may be unresolved. |
| `valid` / `episode_limit` | Task cap exhausted; success is false. |
| `valid` / `positive_cost` | Safe success is false; task success may remain unresolved. |
| `root_mismatch` | Reconstruction fields differ; no candidate evidence. |
| `error`, `timeout`, `cancelled`, `budget_exhausted` | Invalid or unevaluated sample; preserve diagnostics. |

`safe_success` is false after positive cost; otherwise it follows resolved task
success. Plan statuses include `ok`, `no_admissible_plan`, `insufficient_evidence`,
`cancelled`, and `budget_exhausted`. `FirstSuccess` additionally returns
`no_successful_trajectory` or `invalid_evidence`.

**Exceptions:** `StalePlanError(ValueError)` signals an invalid live execution root.
`TrajectoryMismatch(RuntimeError)` signals divergence during checked execution.
Argument/ownership errors raise `ValueError`; closed/busy sessions raise `RuntimeError`.

## Runtime extension contract

Implement `Runtime` in an importable factory selected by `RuntimeSpec`.

| Method | Returns | Required behavior |
| --- | --- | --- |
| `reset(task_id, seed, episode_limit)` | `None` | Reconstruct episode, policy/decoder, cost, and RNG state. |
| `advance(forced_action=None)` | `Transition` | Run policy inference, execute one action, and feed back the actual action. |
| `seed_suffix(seed)` | `None` | Change continuation-policy RNG streams only. |
| `snapshot()` | `dict` | JSON-serializable validation fields. |
| `observe()` | `dict` | Public planning information, excluding audit-only state. |
| `end_episode()` | `None` | Release episode resources. |
| `close()` | `None` | Release the runtime. |

`snapshot()` and `observe()` must not advance physics, run policy inference, or
consume randomness. Optional `execution_signature() -> dict` enables tracing and
checked execution. Optional `candidate_preview() -> dict` supports candidate
inspection in a disposable worker and may perform policy inference there.

Custom runtimes are responsible for reset fidelity and seeded behavior. Matching
declared fields establishes agreement on those fields. Native
`diagnostics_dir` retains per-episode camera checks, renderer metadata, and Unity
logs after shutdown; without it, diagnostics are temporary. A camera mismatch
remains invalid and is never repaired by silently re-rendering or retrying.
See [native reproduction](native.md) and [validation scope](release.md).
