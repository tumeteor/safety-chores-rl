# Validation and limitations

Safety-CHORES-RL provides replay rollouts and budgeted search on a tested native
Safety-CHORES installation. When relevant, please cite
[A Safe Action Is Not Enough: Feasible-Future Decoding for Vision-Language-Action Policies](https://arxiv.org/abs/2610.05166)
by Nguyen et al.

## Checked native execution

The selected `FirstSuccess` example used task 0, seed 1909, with a 200-action
episode cap and one rollout worker on a shared H100. Two of eight allowed
continuations were evaluated: the first failed after 85 actions; the second
completed the task in 75 actions with zero recorded cost.

The successful path was retained and executed with all 75 per-step state and
transition checks matching. Search took 180.3 seconds; total runtime, including
preview, startup, execution, and cleanup, was 339.8 seconds. The task and seed
were informed by earlier experiments, so this is not held-out efficacy evidence.
See the [run receipt](assets/first-success.json).

Checked execution compares camera and exposed physics/cost state, policy
history, rewards, costs, and completion. It excludes only the deliberately
forked policy sampling RNG from per-step comparisons; full root validation
still checks that RNG. A mismatch stops execution before the next action.

## Branch reconstruction and visualization

The two comparison animations show five selected branches: two continuations
from task 0, seed 892, and three from a 40-action root on task 1, seed 1909.
Together they reproduced 585 continuation actions and 120 prefix actions.
All 1,420 captured camera images passed their recorded hash checks.

The earlier study retained actions, rewards, costs, candidate probabilities,
and endpoint observations, but did not retain original root snapshots or
intermediate state hashes. Full captured prefix states agree across the new
reconstructions. This is narrower evidence than the FirstSuccess example's
original per-step state checks. See the
[branch comparison verification scope](native.md#branch-comparison-animations).

## Environment validation

The selected 75-action result reproduced in an isolated Python environment on
Apex, with system and user site packages disabled. All 45 tests passed; the
successful actions and all 76 state signatures matched the original run.
Native runtime was 332.1 seconds, including 177.9 seconds of search.

The native player, GPU driver, external SafeVLA sources, assets, and checkpoint
were reused. This check did not install or rebuild the native stack on a blank
machine. The [environment receipt](assets/apex-fresh-environment.json) records
versions, isolation checks, and six unresolved dependency constraints:

| Package constraint | Installed version |
| --- | --- |
| safety-gymnasium requires MuJoCo 2.3.3 | MuJoCo 3.3.7 |
| gymnasium-robotics requires MuJoCo 2.3.3 | MuJoCo 3.3.7 |
| gymnasium-robotics requires NumPy below 1.24 | NumPy 1.26.4 |
| omnisafe requires pandas 2.0.3 | pandas 2.3.3 |
| AllenAct requires Pillow below 10.3 | Pillow 11.3.0 |
| AllenAct requires torchvision at most 0.16.2 | torchvision 0.19.1+cu121 |

Native execution and tests passed in this configuration despite those conflicts.
The dependency set does not pass `pip check`. The expanded suite also passed
41 tests on Windows with two skips for Linux process cleanup and optional Torch.
See [native setup](native.md) for required inputs and reproduction commands.

## Separate baseline/search study

A bounded paired study used task rows 0-3 with seed 1909, 200-action caps,
queries at steps 0 and 40, three policy-ranked candidates, and four continuation
seeds per candidate. A separate canary used task 0 with a smaller budget.

All 100 sampled continuations (4 canary and 96 main) were valid. All 10 complete
episode replays matched their declared final-state fields and per-step
reward/cost sequences. The [study receipt](assets/native-study-summary.json)
retains both controllers' outcomes and hashes of the original reports.

| Main task row, seed 1909 | Baseline outcome / cost | First-action search outcome / cost |
| --- | --- | --- |
| 0 | Failure / 0 | Failure / 0 |
| 1 | Failure / 0 | Success / 2 |
| 2 | Failure / 13 | Failure / 2 |
| 3 | Failure / 4 | Failure / 20 |

Both canary controllers succeeded with zero cost. Neither controller achieved
zero-cost success across the four main cases. This supports bounded replay
fidelity on the tested installation, but does not establish general planner
efficacy. This study executes only the selected first action and resumes policy
sampling; `FirstSuccess` instead retains and executes a whole successful path.

## Operational limits

- Roots are reconstructed with reset and action replay. Matching exposed state
  and images does not establish restoration of hidden PhysX state.
- Full remaining-horizon candidate evaluation is supported. Exhaustive action-tree
  search and fast hidden-state checkpoint restoration are not established.
  Deep roots incur reset/replay overhead.
- An intermittent manipulation-camera integrity fault was observed in an earlier
  configuration. Diagnostics and strict rejection preserve evidence; they do
  not fix the underlying fault or establish a validated recovery policy.
- The player, task assets, SafeVLA source, and checkpoint remain external
  prerequisites. Installation portability and broader task/seed coverage need
  further validation.

Budgeted sampling, first-success stopping, successive halving, and exact-request
reuse are the practical search primitives exposed by the [API](api.md).
Zero recorded cost in a selected episode does not establish general safety.
