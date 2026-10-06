# Release preparation checkpoint

No public repository, remote, release, or package publication has been created.
The current source tree is an independent benchmark/API draft.

Verified on 2026-10-06:

- Portable Windows Python 3.10 tests: 23 passed, one Linux-only test skipped.
- Independent native adapter: four valid short candidate/continuation branches,
  matching declared replay fields, and an unchanged live episode. One Fetch task,
  one worker, a one-action prefix, and a two-action horizon were tested using an
  existing installation and uniform-random continuation. Branches took about
  30–32 seconds each, including reset/replay overhead.

Still required before a usable simulator release:

- A complete native plan/execute/replan validation and a reproducible task-success
  example. Neither follows from the short smoke or synthetic task examples.
- Curated simulator source patches and a tested public build/install recipe.
- Dependency/environment pins and an installation check outside the existing setup.
- Source license and redistribution review for simulator components and assets.
- Final source, Git history, archive, and wheel review.

The runtime reconstructs episodes using reset and action replay. Matching exposed
state and images does not establish exact hidden PhysX checkpoint restoration.
Deep search can incur substantial reset/replay cost. Treat this checkout as a
development draft, not a ready public benchmark release.
