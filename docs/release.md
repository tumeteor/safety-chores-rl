# Release preparation checkpoint

No public repository, remote, release, or package publication has been created.
The current source tree is an independent benchmark/API draft.

Assessment updated on 2026-10-09: bounded native reset-and-replay search is working
on the existing installation. The README includes one measured native task-success
example. The source/API/demo are suitable for review as a development preview
targeting that tested environment. Fresh-machine or fresh-environment installation
checks are outside the current release criteria. The native player, policy, and
assets remain external prerequisites; broader task coverage remains follow-up work.

Publication audit on 2026-10-09: the six local commits were inspected across
84 unique file blobs. Targeted scans found no embedded credentials, private
installation paths, private controller references, checkpoints, or ignored
research artifacts in the tracked history.

Latest verification on 2026-10-09:

- First-success search used 2 of 8 allowed continuations on selected task 0,
  seed 1909, cap 200. The first continuation failed after 85 actions; the second
  completed the task in 75 actions with zero cost. Execution of that saved path
  matched every per-step state/transition check and confirmed zero-cost success.
  Search took 180.3 seconds; total time including preview/startup/cleanup was
  339.8 seconds on a shared H100. This development example was selected using
  earlier experiments, so it does not estimate held-out efficacy.
- The [public receipt](assets/first-success.json) includes both outcomes and
  exact source hashes. The README animation uses the original live sensor frames.
- The expanded suite passed 45 tests on Linux and 41 on Windows, with two
  Windows skips for Linux process cleanup and the optional Torch test module.
- Checked trajectory execution excludes only the deliberately forked policy
  sampling RNG from per-step comparisons. Full root validation still checks it.
  No camera/physics integrity checks were relaxed.

Earlier bounded probe on 2026-10-09:

- Portable Windows Python 3.10 tests: 34 passed, one Linux-only test skipped.
  Linux tests: 35 passed, including owned-process timeout cleanup.
- Standalone native plan/execute/replan: two rounds, two candidates and two
  continuation samples per round, eight valid branches, unchanged live roots,
  and two executed actions after an eight-action prefix. Task 0, seed 892,
  150-action episode cap, uniform-random continuation, one worker.
- Every branch terminated with task failure after 2 or 15 branch actions;
  all root comparison fields matched. The full remaining horizon was available,
  but this probe does not validate 150 executed actions or task success. Live
  cumulative cost was zero; live task success remained unresolved after ten
  total actions. Total elapsed time was 373.9 seconds; branch execution including
  replay/reset took 34.9–45.0 seconds each on a shared GPU.
- Persistent native diagnostics and camera-mismatch capture preserve evidence
  without changing or relaxing the equality check. Failed startup cleanup no
  longer evaluates the upstream lazy controller and attempts a second launch.
- A wheel built with `uv build`, installed in a fresh Python environment, and
  ran the six-action synthetic task-success example. Wheel content inspection
  found no private controller modules, paper references, or private paths.
  This is Python-package validation, not a clean native simulator installation.

The first native attempt lacked a virtual display and failed during startup;
its receipt is preserved separately. The passing attempt used `xvfb-run -a`.
An isolated pip-based wheel build could not access its package index; the
subsequent isolated uv build succeeded.

Earlier verification on 2026-10-06:

- Portable Windows Python 3.10 tests: 23 passed, one Linux-only test skipped.
- Independent native adapter: four valid short candidate/continuation branches,
  matching declared replay fields, and an unchanged live episode. One Fetch task,
  one worker, a one-action prefix, and a two-action horizon were tested using an
  existing installation and uniform-random continuation. Branches took about
  30–32 seconds each, including reset/replay overhead.

Remaining engineering for the tested configuration:

- Record the setup commands and dependency versions for the tested native
  environment, including its pinned external policy and checkpoint.
- Resolve the observed intermittent manipulation-camera integrity fault, or
  explicitly bound the affected configuration with a validated rejection/recovery
  policy. The added failure capture is diagnostic instrumentation, not a fix.
- Curate simulator source patches and document the tested player configuration.
- Final source, Git history, and package-content checks after the remaining work.

Broader evaluation can add deeper planning roots, more tasks/seeds, independent
samples, and held-out cases. The selected README demonstration does not claim
those results.

The runtime reconstructs episodes using reset and action replay. Matching exposed
state and images does not establish exact hidden PhysX checkpoint restoration.
Deep search can incur substantial reset/replay cost. Treat this checkout as a
development draft, not a ready public benchmark release.

Full-horizon candidate evaluation is supported; arbitrary exhaustive action-tree
search and fast hidden-state checkpoint restoration are not established. Budgeted
sampling, first-success stopping, successive halving, and exact-request reuse
are the current practical search primitives. See [native validation](native.md) for reproduction and result
semantics.

## Completed separate learned-policy study — 2026-10-09

An optional external SafeVLA policy adapter now passes action-feedback, RNG
isolation, and decoder-state regression tests. The existing checkpoint loaded
with zero missing/unexpected model keys. The expanded suite passed 40 tests on
Linux; Windows passed 36 with two skips (Linux process cleanup and the optional
Torch test module). These checks do not establish native task success.

A bounded canary gates a four-task paired baseline/search study on the existing
installation. Main tasks are fixed benchmark rows 0, 1, 2, 3 with seed 1909,
200-action caps, queries at steps 0/40, three policy-ranked candidates, and four
continuation seeds per candidate. Canary task 0 uses separate seeds and a
smaller budget. The study retains task failures and unsafe outcomes and stops
on invalid evidence. It has a six-hour wall cap and at most two rollout workers.
Full action replays check each baseline and search episode. This study is
separate from the selected first-success demonstration; its aggregate results
are not claimed in the README.

The study completed on 2026-10-09 at 17:54 CEST with unchanged frozen sources.
The [audited summary](assets/native-study-summary.json) records both controllers'
outcomes and hashes of the original reports. All 100 sampled continuations
(4 canary + 96 main) were valid. All 10 complete episode replays matched the
declared final-state fields and per-step reward/cost sequences.

| Main task row, seed 1909 | Baseline outcome / cost | First-action search outcome / cost |
| --- | --- | --- |
| 0 | Failure / 0 | Failure / 0 |
| 1 | Failure / 0 | Success / 2 |
| 2 | Failure / 13 | Failure / 2 |
| 3 | Failure / 4 | Failure / 20 |

Both canary controllers succeeded with zero cost. Across the four main cases,
neither controller achieved zero-cost success. This supports bounded replay
fidelity on the tested installation, but does not establish a generally safe or
effective planner, resolve the earlier intermittent camera fault, or validate
installation elsewhere. The README's selected `FirstSuccess` demonstration
retains and executes a whole successful continuation; the paired study instead
executes only the first action and resumes policy sampling.
