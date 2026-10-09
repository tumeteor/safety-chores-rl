# Rollout and planning semantics

`API_VERSION = 1`. Results contain a `schema_version` and can be serialized with
`to_dict()`. `RuntimeSpec` freezes a factory name and JSON configuration. The
factory must be importable in a fresh Python interpreter; closures and live
simulator objects cannot be passed to workers.

## Episode and root lifecycle

1. Own the live runtime with `Episode(spec, task_id=..., seed=..., max_steps=...)`.
2. Create `RolloutSession(spec, workers=...)` as a context manager.
3. `capture_root(episode)` records the executed prefix, cumulative cost, public
   observation, and runtime validation fields without advancing the episode.
4. `validate_root(root)` independently resets and replays; it never uses the cache.
5. `rollout(root, candidate, ...)` or `rollout_batch(root, candidates, ...)` evaluates
   immutable action tuples in independent processes.
6. `execute_first(episode, plan)` checks episode identity and state, executes one
   action, and releases the root and its cached results. Otherwise call `release_root`.

The root is not a portable checkpoint. A copied/deserialized, foreign-session, or
released root is rejected. Capturing an identically seeded second episode does
not make a plan executable in it. A session serializes callers; create separate
sessions for concurrent callers. `cancel()` is callable from another thread and
interrupts the current batch. Later batches may use restarted workers.

## Requests, seeds, and outcomes

A candidate is a nonempty tuple of nonnegative integer action IDs. Its length
must fit `RolloutBudget.max_steps`, which includes the candidate and continuation
but excludes root replay. That horizon cannot exceed the remaining episode cap.
With `continuation=False`, only the candidate prefix is executed.

Each replay uses the episode's original task and seed. Policy inference happens
even for forced actions, and the actual executed action is fed back. The sample
seed forks policy randomness **after the entire candidate and its feedback**.
Simulator randomness is not reseeded at that point. Samples therefore vary
the continuation, not the candidate's pre-action randomness. Candidate order and
worker assignment do not change a deterministic runtime's seeded outcomes.

Batch results are ordered sample-major, then candidate order. Exact duplicates
are coalesced. Search requires distinct sample seeds; repeated seeds are not
independent evidence.

| Result | Interpretation |
| --- | --- |
| `status=valid`, `stop_reason=terminal` | Runtime reported termination; success comes from its outcome |
| `horizon` / `prefix_complete` | Observed prefix only; eventual success normally remains `None` |
| `episode_limit` | Full task cap exhausted; success is false |
| `positive_cost` | Nonnegative cost has ruled out safe success; eventual task success may remain unknown |
| `root_mismatch` | Declared reconstruction fields differ; no candidate evidence |
| `error` / `timeout` / `cancelled` / `budget_exhausted` | Invalid or unevaluated sample; retain status and diagnostics |

`safe_success` is false after positive cost, otherwise follows resolved task
success. `safe_completion` refuses unresolved samples. `reward_return` is observed
reward only. A custom objective must return a finite number or `None`.

## Resource limits and reuse

`SearchBudget` bounds rollout calls, primitive native actions, elapsed wall time,
and candidates. Native actions include root-prefix replay. The coordinator
reserves worst-case work before dispatching each bounded worker wave. Successful
early termination returns unused reservations. Errors and timeouts are charged
conservatively. Startup and reset count toward wall time; native-action counts
do not measure internal rendering, reset, policy inference, or cleanup cost.
Timeout cleanup may extend observed elapsed time beyond the deadline.

The session has an LRU cache bounded by `cache_size` entries. Its identity includes
runtime configuration, episode/root, candidate, sample seed, continuation, and
all rollout budget flags. Only valid results are cached. Reuse is session-local;
releasing a root removes its cache entries. `cache_hit=True` reports zero new
native steps. Registered roots still need explicit release. Large observations
and trajectories increase memory per entry; lower or disable the cache as needed.

## Planners

`BestOfN` evaluates all candidates on the requested seeds. `SuccessiveHalving`
starts with a small matched sample set, removes weak candidates, and multiplies
the surviving sample budget each round. It offers no confidence or optimality
guarantee. Record its seed set and elimination settings with each result.

`BeamSearch` scores prefix-only rollouts. Its expander receives public endpoint
observations and the action prefix; it must be cheap and bounded. A reusable
iterable of actions is also accepted. The heuristic receives a rollout result.
Default `max_cost=0` prunes observed violations; it cannot guarantee future safety.
Each deeper prefix resets and replays from the same root, rather than assuming a
hidden-state clone. Results expose expansion counts and completed depth.

All planners retain invalid sample records and explicit budget status. A partially
evaluated plan may have no action. `RecedingHorizon` executes a proposed action
only after a root check; on no plan it returns `(plan, None)` unless a fallback
callback was explicitly supplied. Callers wanting to stop on cancellation or
budget exhaustion should inspect a plan before calling `execute_first`.

## First-success and checked execution

`FirstSuccess` preserves caller candidate priority, visits distinct sample seeds
in order, and dispatches at most `session.workers` candidates per wave. It forces
`stop_on_cost=True` and `record_trace=True`. The first valid, completed zero-cost
success in a fully valid wave becomes `plan.witness`; `plan.candidate` contains
its entire action sequence. Every evaluated failure remains in `evaluations`.
Invalid evidence, cancellation, or budget exhaustion ends the search without
an executable witness. No successful sample also produces no proposed action.
This is a feasibility search; its stopping rule does not estimate success rate.

Call `session.execute_trajectory(episode, plan)` to execute the saved sequence.
It verifies the live root and checks the runtime signature before and after each
action, plus exact reward/cost and episode completion. It releases the root and
its cache on execution completion or failure. `TrajectoryMismatch` stops further
actions; it cannot undo the action that revealed divergence. The optional
`on_step(episode, transition, index, matched)` observer is followed by another
state check. Observers should only record evidence.

Tracing requires `execution_signature()`, a side-effect-free JSON-serializable
runtime extension. `RolloutResult.trace` contains one root hash followed by one
hash per returned action. Native signatures contain the full snapshot except
the policy sampling RNG intentionally forked for continuation proposals. Cameras,
exposed physics, cost state, policy history/decoder/memory, and source identity
still have to match. Initial root validation includes the RNG. Forced execution
still runs policy inference and actual-action feedback at every step. This is
agreement on declared fields, not a hidden-engine-state or future-safety proof.

## Runtime extension contract

Implement the `Runtime` protocol:

* `reset(task_id, seed, episode_limit)` reconstructs a fresh episode and all
  policy, decoder, cost, and RNG bookkeeping. Persistent weights are allowed only
  when reuse preserves this state.
* `advance(forced_action=None)` runs one policy forward, executes one primitive
  action, records actual-action feedback, and returns a validated `Transition`.
* `seed_suffix(seed)` changes the intended continuation RNG streams only.
* `snapshot()` exposes JSON-serializable validation fields. It and `observe()`
  must neither step physics nor infer policy nor consume randomness.
* `observe()` exposes public planning information, excluding audit-only state.
* `end_episode()` releases episode resources; `close()` releases the runtime.

Optional `candidate_preview()` supports `inspect_candidates()`. It is
called only in a disposable replay episode and may perform policy inference.
Custom runtimes must ensure reset fidelity and deterministic seeded behavior.
Matching declared fields is evidence of replay fidelity, not proof of equality
of unobservable engine state. Changing runtime code or artifacts requires a new
session and configuration identity.

## Native runtime and evidence

See [native validation](native.md) for the external installation requirements,
configuration command, and bounded plan/execute/replan example. The native
runtime accepts `diagnostics_dir` as an optional persistent output directory.
Each runtime creates a unique subdirectory; each reset creates an episode
directory. Camera checks, renderer metadata, and the Unity log survive normal
runtime shutdown there. Without this option, diagnostic files are temporary.

A camera mismatch remains an invalid execution. Capture records both existing
arrays and their hashes; it does not refresh observations, retry an action, or
relax the equality check. If capture itself fails, the integrity error still
propagates. Retained files can be incomplete after a hard process termination.
Diagnostics may contain observations, local paths, and scene metadata; choose
which files to publish separately from the source distribution.

For runtime errors, `native_steps` reports the reserved upper bound because an
action may have executed before the error was raised. The partial action trace
contains only transitions that returned successfully. Search-budget accounting
also charges the reservation; an invalid result is never cached as evidence.
