# Native rollouts

<p class="note-lead">Reconstruct a native decision state, sample policy continuations, and check the selected trajectory during execution. This note covers the tested Safety-CHORES environment and the commands behind the recorded demonstrations.</p>

<nav class="note-nav" aria-label="On this page"><a href="#setup">Setup</a><a href="#first-success-demonstration">Reproduce</a><a href="#trajectory-visualization">Trajectories</a><a href="#branch-comparison-animations">Branches</a></nav>

## Setup

Use **Linux with NVIDIA/Vulkan**, the modified simulator player, task/house assets,
and a compatible **SafeVLA + PyTorch/AllenAct** environment with a checkpoint.
These native components are external to the Python package. The tested versions
and dependency constraints are listed in [validation and limitations](release.md#environment-validation).

<details markdown="1" class="note-details">
<summary>Prepare the native and policy configurations</summary>

```sh
python -m pip install -e ".[test,native]" pillow
python tools/prepare_native_config.py \
  --safevla-repo /path/to/SafeVLA \
  --asset-root /path/to/assets \
  --benchmark /path/to/fetchtype_val.jsonl.gz \
  --houses /path/to/val.jsonl.gz \
  --build-manifest /path/to/player/checkpoint-build.json \
  --player /path/to/player/thor-Linux64-local \
  --output runs/native.json
```

Create `runs/policy.json` with the checkpoint's actual path and SHA256:

```json
{
  "checkpoint": "/path/to/checkpoint.pt",
  "checkpoint_sha256": "REPLACE_WITH_CHECKPOINT_SHA256",
  "device": "cuda:0",
  "greedy": false
}
```

Obtain the digest with `sha256sum /path/to/checkpoint.pt`. The loader rejects
unexpected or missing weights unless explicitly allowed. Test-time image
augmentation is disabled.

Keep these configuration files local: they contain installation paths. Native
configuration hashes identify the player, task inputs, and external sources;
they do not constitute a complete environment or asset lockfile.

</details>

## First-success demonstration

The selected Fetch case asks the robot to navigate to a basketball and grasp it.
`FirstSuccess` evaluates candidates until a valid worker batch contains a
completed zero-cost continuation, then retains and executes the entire path.

| Task / seed | Search | Checked execution | Recorded cost |
| --- | --- | --- | --- |
| 0 / 1909 | 2 of 8 allowed rollouts | 75 of 75 actions matched | 0 |

On one shared H100 with one worker, search took **180.3 s** and the complete run
**339.8 s**. Candidate 2 failed after 85 actions; candidate 1 succeeded after 75.
Both used continuation seed 4100. [Run receipt](assets/first-success.json) ·
[Watch the demonstration](../README.md#a-practical-native-example).

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_first_success.py \
  --config runs/native.json --policy-config runs/policy.json \
  --out runs/first-success --task-id 0 --seed 1909 \
  --vulkan-device 0 --workers 1 --candidates 2 \
  --sample-seeds 4100 4101 4102 4103 --episode-limit 200 --save-frames
```

Use a **new output directory** and the physical Vulkan GPU index; CUDA visibility
alone does not select it. `xvfb-run` provides the display required by the upstream
controller. The task and seed were selected using earlier experiments.

<details markdown="1" class="note-details">
<summary>Execution contract, budgets, and outputs</summary>

- Workers reconstruct the simulator and policy history through reset and action
  replay. The API does not restore hidden physics checkpoints.
- The default request timeout is 420 s and the search budget is 1,800 s. Preview,
  live startup, and execution add work. Use an outer process timeout for
  unattended runs because initial live reset is not interruptible by the budget.
- No successful trajectory means no action fallback; the example exits with
  status 2. A mismatch during checked execution prevents the next action.
- `report.json` retains results and partial evidence. `frames/` stores the existing
  navigation/manipulation observations at the root and after each action; saving
  frames does not issue rendering calls.

The native signatures check camera images, exposed physics/cost state, policy
history, and source identity. Per-step execution comparisons exclude the policy
sampling RNG deliberately forked for proposals; full root validation includes it.
See the [checked-execution API](api.md#first-success-and-checked-execution).

</details>

<details markdown="1" class="note-details">
<summary>Other evaluation recipes: paired study and integration probe</summary>

The paired study compares ordinary policy sampling with first-action search at
declared query steps. Both full live episodes are replayed; failures, unsafe
outcomes, and invalid branches remain in the report.

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_task_study.py \
  --config runs/native.json --policy-config runs/policy.json \
  --out runs/task-study --task-id 0 --seed 1909 --episode-limit 200 \
  --vulkan-device 0 --workers 2 --query-steps 0 40 \
  --candidates 3 --sample-seeds 4100 4101 4102 4103
```

A smaller integration probe uses uniform-random continuations and executes two
live actions after a fixed prefix:

```sh
PYTHONHASHSEED=0 xvfb-run -a python examples/native_search.py \
  --config runs/native.json --out runs/native-search \
  --vulkan-device 0 --episode-limit 150 --samples 2 --live-steps 2 \
  --wall-seconds 600 --prefix 0 0 0 0 0 0 0 0
```

`native_search_validated=true` means the requested rounds reconstructed their
roots and produced valid branches before execution. Read task success, cost,
and length separately. This probe supplies no live safety filter, and its action
IDs depend on the pinned vocabulary. `--planner halving` selects successive
halving; `--policy-factory module:Class --policy-config policy.json` supplies a
custom continuation policy. Both runners retain diagnostics and need an outer
timeout for unattended operation.

</details>

## Trajectory visualization

The original report contains state hashes and camera frames. Forced replay with
the same runtime, configuration, and checkpoint recovers base positions from
cached metadata without extra render or movement calls. For the selected run,
both paths matched their original signatures and all **152 successful-path
camera images** matched the saved live frames.

<details markdown="1" class="note-details">
<summary>Recover coordinates and render the animation</summary>

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_trajectory_replay.py \
  --config runs/native.json --policy-config runs/policy.json \
  --report runs/first-success/report.json --out runs/trajectory-replay \
  --vulkan-device 0

python -m pip install matplotlib pillow
python tools/plot_trajectories.py \
  --trajectories runs/trajectory-replay/trajectories.json \
  --out runs/first-success/visuals
python tools/render_first_success.py \
  --report runs/first-success/report.json --frames runs/first-success/frames \
  --trajectories runs/trajectory-replay/trajectories.json \
  --out runs/first-success/visuals
```

Omit `--trajectories` from the renderer for a camera-only animation. Replay is
additional work, excluded from the reported search time. Every replayed camera
hash must match the original live frame before it is used in the animated map.

</details>

Maps show base centres and headings in Unity world X/Z metres. Room outlines omit
doors and furniture; rotation and arm motion may leave the base stationary.
Playback runs at eight actions per second with a final hold, rather than wall time.

## Branch comparison animations

The two comparisons use selected continuations from the `BestOfN` study.
Action counts below exclude the shared prefix.

| Case / root | Candidate / continuation seeds | Outcomes: actions, cost |
| --- | --- | --- |
| [Task 0, seed 892](assets/branches-seed892.json) / initial state | `1` / `9002`, `9001` | Success: 74, 0; limit: 150, 0 |
| [Task 1, seed 1909](assets/branches-prefix40.json) / 40 actions | `0` / `4102`, `4100`, `4101` | Success: 98, 0; success: 103, 3; limit: 160, 0 |

Each capture resets the episode, replays the live prefix and candidate, then
regenerates the continuation from its saved seed. Checks cover every action,
reward, cost, termination, root candidate probabilities, and final observation,
including both camera hashes. Full reset/prefix states agree across the new
reconstructions. The historical study did not save original root snapshots or
intermediate state hashes, so the receipts distinguish these checks from
historical per-step state equality.

<details markdown="1" class="note-details">
<summary>Capture and render the recorded branches</summary>

Use the exact sources, configuration, and checkpoint from the original report.
These clips used frozen runtime sources corresponding to `d0de2d9`, including
the original file bytes and line endings that contribute to source identity.

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_branch_replay.py \
  --config runs/native.json --policy-config runs/policy.json \
  --report runs/canary-task0-seed892/report.json --query-step 0 \
  --selection 1:9002 1:9001 --out runs/branches-seed892 --vulkan-device 0

PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_branch_replay.py \
  --config runs/native.json --policy-config runs/policy.json \
  --report runs/task1-seed1909/report.json --query-step 40 \
  --selection 0:4102 0:4100 0:4101 --out runs/branches-prefix40 --vulkan-device 0

python tools/render_branch_replay.py \
  --capture runs/branches-seed892/branches.json --out runs/branches-seed892.gif
python tools/render_branch_replay.py \
  --capture runs/branches-prefix40/branches.json --out runs/branches-prefix40.gif
```

The renderer verifies source-image hashes, preserves outcome/cost milestones,
and labels reset, replay, and continuation. It keeps every second action by
default, with holds at reset, the root, and the outcome. Columns align by action
count; captures were collected sequentially. Navigation images keep their
resolution; manipulation images are reduced to fit. Cyan marks the shared prefix.

</details>

For complete episode outcomes, environment checks, and operational limits, see
[validation and limitations](release.md). For interfaces and return types, see the
[API reference](api.md).
