# Release preparation checkpoint

No public repository, remote, release, or package publication has been created.
The current source tree is an independent benchmark/API draft.

Assessment updated on 2026-10-09: continue release engineering. Bounded native
reset-and-replay search is working on the existing installation. A usable public
simulator release still needs task-success and independent-installation evidence.

Verified on 2026-10-09:

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

Still required before a usable simulator release:

- A reproducible native task-success example using an independently distributable
  continuation policy or explicitly labeled oracle. Add deeper execution roots,
  full successful action replays, multiple tasks/seeds, and independent samples.
- Resolve the observed intermittent manipulation-camera integrity fault, or
  explicitly bound the affected configuration with a validated rejection/recovery
  policy. The added failure capture is diagnostic instrumentation, not a fix.
- Curated simulator source patches and a tested public build/install recipe.
- Dependency/environment pins and an installation check outside the existing setup.
- Source license and redistribution review for simulator components and assets.
- Final source, Git history, and distribution review after the remaining work.

The runtime reconstructs episodes using reset and action replay. Matching exposed
state and images does not establish exact hidden PhysX checkpoint restoration.
Deep search can incur substantial reset/replay cost. Treat this checkout as a
development draft, not a ready public benchmark release.

Full-horizon candidate evaluation is supported; arbitrary exhaustive action-tree
search and fast hidden-state checkpoint restoration are not established. Budgeted
sampling, successive halving, and exact-request reuse are the current practical
search primitives. See [the native probe](native.md) for reproduction and result
semantics.

## Pending learned-policy study — 2026-10-09

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
Full action replays check each baseline and search episode. No release gate is
closed until the native receipts are audited.
