# Native rollouts

<p class="note-lead">Reconstruct a native decision state, sample policy continuations, and check the selected trajectory during execution. This note covers the tested Safety-CHORES environment and the commands behind the recorded demonstrations.</p>

<nav class="note-nav" aria-label="On this page"><a href="#setup">Setup</a><a href="#first-success-demonstration">Reproduce</a><a href="#trajectory-visualization">Trajectories</a><a href="#branch-comparison-animations">Branches</a></nav>

## Setup

Use **Linux with NVIDIA/Vulkan** and activate your existing **SafeVLA Python
environment**. The setup helper locates the installed sources, modified player,
task/house data, assets, and checkpoint, then generates the configs and launcher:

```sh
python -m pip install -e ".[native]" pillow
python tools/setup_native.py
bash runs/native-setup/run_demo.sh --save-frames
```

It prints the full number of task rows in your configured benchmark. The default
asset preflight checks task 0; it does not restrict the available episode set.
The simulator and Python environment must already be installed. See the tested
[environment and dependency constraints](release.md#environment-validation).

<details markdown="1" class="note-details">
<summary>Discovery, checkpoint download, and generated files</summary>

```sh
# Use a different installation root, or reuse a previously pinned config.
python tools/setup_native.py --installation-root ../SafeVLA
python tools/setup_native.py --from-config runs/native.json

# If the task's official checkpoint is not installed, download it (~2 GB).
python tools/setup_native.py --download-checkpoint

# Check assets for more task IDs without changing the benchmark file.
python tools/setup_native.py --task fetch --task-ids 0 1 2 3 --output runs/fetch-setup
```

These are alternative setup commands. Discovery checks standard directories
under the installation root, sibling SafeVLA directories, and existing
`SAFEVLA_REPO`, `SAFEVLA_CHECKPOINT`, `OBJAVERSE_*`, `PYTHONPATH`, and
`SAFETY_CHORES_PLAYER` settings. `SAFETY_CHORES_NATIVE_CONFIG` can identify an
existing config. Ambiguous or missing inputs produce an explicit error and an
override flag; run `python tools/setup_native.py --help` for all options.

The helper writes `native.json`, `policy.json`, `env.sh`, `run_demo.sh`, and a
`setup.json` receipt under `runs/native-setup`. It hashes the checkpoint, verifies
the player's artifact manifest, checks assets for the requested task IDs, and
preserves existing source and data files. Repeating an identical setup is safe;
use a new output directory when changing inputs. For the other recipes in this
note, first run `source runs/native-setup/env.sh`.

Optional downloads use the [official SafeVLA weights](https://huggingface.co/SafetyEmbodiedAI/safety-model)
and verify their SHA256: `safe_fetch.pt`, `safe_pickup.pt`, or `safe_objnav.pt`.
The Fetch checkpoint is the same artifact used in the recorded demonstration.
Use `--checkpoint` for a custom policy. The policy loader still checks model
compatibility; test-time image augmentation is disabled.

Keep generated files local because they contain installation paths. Artifact
checks configure the installation; native execution is checked when you run
the demo. This helper does not install the simulator or rebuild the Python
environment. Proxy settings are inherited from the calling shell.

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
bash runs/native-setup/run_demo.sh \
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
  --config runs/native-setup/native.json --policy-config runs/native-setup/policy.json \
  --out runs/task-study --task-id 0 --seed 1909 --episode-limit 200 \
  --vulkan-device 0 --workers 2 --query-steps 0 40 \
  --candidates 3 --sample-seeds 4100 4101 4102 4103
```

A smaller integration probe uses uniform-random continuations and executes two
live actions after a fixed prefix:

```sh
PYTHONHASHSEED=0 xvfb-run -a python examples/native_search.py \
  --config runs/native-setup/native.json --out runs/native-search \
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
  --config runs/native-setup/native.json --policy-config runs/native-setup/policy.json \
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
  --config runs/native-setup/native.json --policy-config runs/native-setup/policy.json \
  --report runs/canary-task0-seed892/report.json --query-step 0 \
  --selection 1:9002 1:9001 --out runs/branches-seed892 --vulkan-device 0

PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_branch_replay.py \
  --config runs/native-setup/native.json --policy-config runs/native-setup/policy.json \
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
