# Validation and limitations

<p class="note-lead">Native execution, branch reconstruction, and environment reproducibility on the tested Safety-CHORES installation. This note records the checks behind the demonstrations and the scope of the results.</p>

<nav class="note-nav" aria-label="On this page"><a href="#checked-native-execution">Execution</a><a href="#branch-reconstruction-and-visualization">Branches</a><a href="#environment-validation">Environment</a><a href="#separate-baselinesearch-study">Paired study</a><a href="#operational-limits">Limits</a></nav>

## Checked native execution

The selected `FirstSuccess` run completed task 0, seed 1909, in **75 actions with
zero recorded cost**. All 75 per-step state and transition checks matched during
execution. The task and seed were informed by earlier experiments; this is
selected replay evidence rather than held-out efficacy evidence.

| Search | Checked execution | Search time | Total runtime |
| --- | --- | --- | --- |
| 2 of 8 allowed continuations | 75 / 75 actions matched | 180.3 s | 339.8 s |

[Run receipt](assets/first-success.json) ·
[Reproduction command](native.md#first-success-demonstration)

<details markdown="1" class="note-details">
<summary>Run conditions and state checks</summary>

The run used a 200-action episode cap and one rollout worker on a shared H100.
The first continuation failed after 85 actions; the second completed the task
in 75. The successful path was retained and executed in full. Total runtime
includes preview, startup, search, execution, and cleanup.

Checked execution compares camera and exposed physics/cost state, policy
history, rewards, costs, and completion. It excludes only the deliberately
forked policy sampling RNG from per-step comparisons; full root validation
still checks that RNG. A mismatch stops execution before the next action.

</details>

## Branch reconstruction and visualization

The two comparison animations reconstruct **five selected branches**, covering
585 continuation actions and 120 prefix actions. All **1,420 captured camera
images** passed their recorded hash checks. The historical study retained less
state evidence than the checked-execution run above.

<details markdown="1" class="note-details">
<summary>Branch coverage and verification scope</summary>

Two continuations start from task 0, seed 892; three start from a 40-action root
on task 1, seed 1909. The earlier study retained actions, rewards, costs,
candidate probabilities, and endpoint observations, but did not retain original
root snapshots or intermediate state hashes.

Full captured prefix states agree across the new reconstructions. This is
narrower evidence than the FirstSuccess example's original per-step state
checks. See the [branch comparison recipes and verification scope](native.md#branch-comparison-animations).

</details>

## Environment validation

In an isolated Python environment on Apex, **45 tests passed**, and the selected
run reproduced the same 75 actions and all **76 state signatures**. The native
stack was reused. Six dependency constraints remain unresolved, so the tested
environment does **not** pass `pip check`.

[Environment receipt](assets/apex-fresh-environment.json) ·
[Native prerequisites](native.md#setup)

<details markdown="1" class="note-details">
<summary>Isolation, dependencies, and platform coverage</summary>

System and user site packages were disabled. Native runtime was 332.1 seconds,
including 177.9 seconds of search. The player, GPU driver, external SafeVLA
sources, assets, and checkpoint were reused; this check did not install or
rebuild the native stack on a blank machine.

| Package constraint | Installed version |
| --- | --- |
| safety-gymnasium requires MuJoCo 2.3.3 | MuJoCo 3.3.7 |
| gymnasium-robotics requires MuJoCo 2.3.3 | MuJoCo 3.3.7 |
| gymnasium-robotics requires NumPy below 1.24 | NumPy 1.26.4 |
| omnisafe requires pandas 2.0.3 | pandas 2.3.3 |
| AllenAct requires Pillow below 10.3 | Pillow 11.3.0 |
| AllenAct requires torchvision at most 0.16.2 | torchvision 0.19.1+cu121 |

Native execution and tests passed in this configuration despite those conflicts.
The expanded suite also passed 41 tests on Windows, with two skips for Linux
process cleanup and optional Torch. The receipt records versions and isolation
checks.

</details>

## Separate baseline/search study

A bounded paired study produced **100 valid continuations** and **10 matching
complete episode replays**. Neither controller achieved zero-cost success on
any of the four main cases. These checks support bounded replay fidelity on
the tested installation; they do not establish general planner efficacy.

<details markdown="1" class="note-details">
<summary>Study protocol and paired outcomes</summary>

The main study used task rows 0–3 with seed 1909, 200-action caps, queries at
steps 0 and 40, three policy-ranked candidates, and four continuation seeds per
candidate. A separate canary used task 0 with a smaller budget. Of the 100
sampled continuations, four were canary and 96 were main-study continuations.

All 10 complete episode replays matched their declared final-state fields and
per-step reward/cost sequences. The [study receipt](assets/native-study-summary.json)
retains both controllers' outcomes and hashes of the original reports.

| Main task row, seed 1909 | Baseline outcome / cost | First-action search outcome / cost |
| --- | --- | --- |
| 0 | Failure / 0 | Failure / 0 |
| 1 | Failure / 0 | Success / 2 |
| 2 | Failure / 13 | Failure / 2 |
| 3 | Failure / 4 | Failure / 20 |

Both canary controllers succeeded with zero cost. This study executes only
the selected first action and resumes policy sampling; `FirstSuccess` instead
retains and executes a whole successful path.

</details>

## Operational limits

| Area | Validated scope and remaining limits |
| --- | --- |
| State reconstruction | Reset and action replay match exposed state and images in the tested cases. Restoration of hidden PhysX state is not established. |
| Search | Full remaining-horizon candidate evaluation is supported. Exhaustive action-tree search and fast hidden-state checkpoint restoration are not established; deep roots incur replay overhead. |
| Camera integrity | An intermittent manipulation-camera fault occurred in an earlier configuration. Diagnostics and strict rejection preserve evidence; they do not fix the fault or establish a validated recovery policy. |
| Portability | The player, task assets, SafeVLA source, and checkpoint remain external prerequisites. Installation portability and broader task/seed coverage need further validation. |

Budgeted sampling, first-success stopping, successive halving, and exact-request
reuse are available through the [API](api.md). Zero recorded cost in a selected
episode does not establish general safety.
