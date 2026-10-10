# Native validation on an existing installation

The native adapter currently requires Linux, an NVIDIA/Vulkan environment,
the modified simulator player, a compatible SafeVLA installation, and local
benchmark tasks, houses, and assets. Installing this Python package does not
install those components. The current scope uses the tested existing environment;
fresh-environment installation testing is outside the release criteria.
The Python-only examples do not establish native task success.

Pin the existing installation explicitly:

```sh
python -m pip install -e ".[test,native]"
python tools/prepare_native_config.py \
  --safevla-repo /path/to/SafeVLA \
  --asset-root /path/to/assets \
  --benchmark /path/to/fetchtype_val.jsonl.gz \
  --houses /path/to/val.jsonl.gz \
  --build-manifest /path/to/player/checkpoint-build.json \
  --player /path/to/player/thor-Linux64-local \
  --output runs/native.json
```

Keep `runs/native.json` local: it contains installation paths. The command
records the benchmark, houses, manifest, player executable, and external Python
source hashes. This is not yet a complete environment or asset lockfile.

Run the bounded full-horizon search probe in the prepared native environment:

```sh
PYTHONHASHSEED=0 xvfb-run -a python examples/native_search.py \
  --config runs/native.json --out runs/native-search \
  --vulkan-device 0 --episode-limit 150 --samples 2 --live-steps 2 \
  --wall-seconds 600 --prefix 0 0 0 0 0 0 0 0
```

`xvfb-run` supplies the display required by the currently tested upstream
controller even when Vulkan renders the scene. Use the physical Vulkan GPU
index; CUDA visibility alone does not select it. Output directories must be
new. Action IDs depend on the pinned task vocabulary. The example records that
vocabulary in `report.json`; the prefix above is an integration probe, not an
expert trajectory.

For every search round, the remaining episode horizon is evaluated for each
candidate and continuation sample. A branch can terminate earlier. The
example checks that search preserved the live root, executes one selected
action, then plans again. `--planner halving` selects successive halving;
`--policy-factory module:Class --policy-config policy.json` supplies a caller's
continuation policy through the interface in `native/policy.py`.

The default uniform-random policy tests the integration. A zero-scoring
candidate can still be selected, and this example does not impose a live
safety filter. `native_search_validated=true` means the requested rounds had
valid, resolved branches and unchanged live roots followed by execution. It
does not imply task success, safety, an optimal plan, or execution of every
action up to the cap. Read branch costs, lengths, terminal outcomes, live cost,
and `task_success` separately. With only two live actions, live task success
will normally remain unknown.

The example saves a report after each round and on errors/interruption, and
retains native diagnostics beneath the output directory. A nonvalidated probe
returns a nonzero exit status. Initial live reset is not interruptible by the
in-process wall budget; use an outer process timeout for unattended runs.

Before publishing a native task-success example, freeze its task IDs, seeds,
policy/candidate source, compute budget, and success criteria. Retain failures
and refusals as well as successes. Validate later execution roots and complete
action replays on the declared setup. A declared oracle or scripted fixture must
be labeled as such.

## Learned continuation validation (experimental)

`safety_chores_rl.native.safevla_policy:SafeVLAPolicy` adapts the external
SafeVLA inference agent. It requires the existing compatible PyTorch/AllenAct
environment and a local checkpoint; neither is bundled. The policy configuration
contains `checkpoint`, its `checkpoint_sha256`, `device` (for example `cuda:0`),
and optional `greedy` (default false). The loader rejects missing or unexpected
weights unless their exact names are explicitly declared in the configuration.
Test-time image augmentation is disabled for this adapter.

Every native action runs one model forward, then feeds the actual executed
action back into the upstream agent, including search interventions. Sampling
uses isolated Python, NumPy, and Torch RNG streams; suffix reseeding preserves
the policy history and does not reseed the simulator. Replay validation includes
checkpoint identity, action history, decoder key/value caches, time counters,
memory, previous-action feedback, and RNG hashes. The independent candidate
preview ranks current policy probabilities in a disposable worker.

The experimental paired study runner is:

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_task_study.py \
  --config runs/native.json --policy-config runs/policy.json \
  --out runs/task-study --task-id 0 --seed 1909 --episode-limit 200 \
  --vulkan-device 0 --workers 2 --query-steps 0 40 \
  --candidates 3 --sample-seeds 4100 4101 4102 4103
```

It compares ordinary policy execution with bounded search at the declared
steps. Each branch can continue to the remaining task cap. Both full live
action sequences are independently replayed and compared, including final
policy/physics/image state, per-step reward/cost, and terminal outcome.
Invalid branches fail the integration gate and remain in the report. Successful
integration does not require task success: failures and unsafe outcomes must
remain visible. The program saves partial evidence on failure and needs an
outer timeout for unattended operation.

## First-success demonstration

The [README example](../README.md#a-practical-native-example) uses
`examples/native_first_success.py`. On the selected task 0 / seed 1909 case,
two continuations were evaluated: candidate action 2 failed after 85 actions;
candidate action 1 succeeded after 75, both with suffix seed 4100. The saved
75-action success then executed with zero cost and all per-step checks matching.
These outcomes are from an existing installation, not an independent clean build
or held-out evaluation. The earlier paired study is a separate experiment.

Use the README command after creating the native and policy configurations
above. The default budget is two candidates, four suffix seeds, one worker,
200 actions per branch, 420 seconds per worker request, and 1,800 seconds of
search time. Candidate preview and live startup/execution are additional work.
Use an outer process timeout for unattended runs. There is no action fallback
when a successful trajectory is absent; the example exits with status 2.

`--save-frames` needs Pillow and writes the existing navigation/manipulation
observations at the root and after each executed action. It never re-renders
for the visualization. Regenerate the animation from a completed report with:

```sh
python -m pip install pillow
python tools/render_first_success.py \
  --report runs/first-success/report.json --frames runs/first-success/frames \
  --out runs/first-success/visuals
```

The renderer preserves the camera images and adds labels and a progress strip.
Playback is eight actions per second, with a final hold; this is not wall time.

## Trajectory visualization

The original example stores state hashes and images, not world coordinates.
Recover robot base positions and headings with a forced replay of each saved
rollout. Use the same runtime package, native configuration, and policy checkpoint
as the original run: every root and action must match its original signature.
The collector reads cached event metadata without issuing extra render or move
commands. Replay is additional work; it is excluded from the README search time.

```sh
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 xvfb-run -a \
  python examples/native_trajectory_replay.py \
  --config runs/native.json --policy-config runs/policy.json \
  --report runs/first-success/report.json --out runs/trajectory-replay \
  --vulkan-device 0

python -m pip install matplotlib pillow
python tools/plot_trajectories.py \
  --trajectories runs/trajectory-replay/trajectories.json --out runs/first-success/visuals
python tools/render_first_success.py \
  --report runs/first-success/report.json --frames runs/first-success/frames \
  --trajectories runs/trajectory-replay/trajectories.json --out runs/first-success/visuals
```

The plot shows robot base centres in Unity world X/Z coordinates (metres), over
the house's room floor polygons. Doors and furniture are omitted, so the outlines
are not an occupancy or collision map. Rotation and arm motion do not necessarily
change base position. The animated map uses recovered poses and requires every
replayed camera hash to match the corresponding original live frame. In the
published selected example, both paths (85 and 75 actions) matched every saved
signature, and all 152 successful-path camera images matched the live frames.

## Branch comparison animations

The README includes two further episodes from the recorded `BestOfN` study:

| Capture | Search root | Selected candidate / suffix seeds | Recorded outcomes |
| --- | --- | --- | --- |
| [Episode 0, seed 892](assets/branches-seed892.json) | Initial state, empty prefix | `1` / `9002`, `9001` | Success: 74 actions, cost 0; limit: 150 actions, cost 0 |
| [Episode 1, seed 1909](assets/branches-prefix40.json) | After 40 live actions | `0` / `4102`, `4100`, `4101` | Success: 98 actions, cost 0; success: 103 actions, cost 3; limit: 160 actions, cost 0 |

Action counts in this table exclude the prefix. These are selected branches
from the earlier study, not additional `FirstSuccess` performance trials. The
second comparison deliberately retains a costly success and an unsuccessful
branch alongside a zero-cost success.

For every selected branch, `examples/native_branch_replay.py` starts a fresh
seeded episode, forces the recorded live prefix, forces the candidate action,
then regenerates the policy continuation using the original suffix seed. There
is no hidden physics-state clone. Prefix replay runs policy inference and feeds
back the actual actions to rebuild policy history along with the simulator.
The collector reads current observations and cached world metadata without
additional render or movement calls.

The collector checks every regenerated action, reward, cost, termination,
original root candidate probabilities, and the complete final observation
(including both camera hashes). Full reset/prefix snapshots must agree across
the newly reconstructed branches. The original study did not retain root
snapshots or intermediate state hashes, so these captures do **not** establish
per-step state equality against the historical run. Both JSON receipts retain
the original report hash, collector hash, reconstructed state hashes, world
poses, image hashes, and explicit verification scope.

Use the exact runtime sources, native configuration, and policy checkpoint
from the original report. These historical clips reused the frozen package
sources corresponding to commit `d0de2d9`, including their original file bytes
and line endings, which contribute to the runtime identity. Example capture
commands for the saved study cases:

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

python -m pip install pillow
python tools/render_branch_replay.py \
  --capture runs/branches-seed892/branches.json --out runs/branches-seed892.gif
python tools/render_branch_replay.py \
  --capture runs/branches-prefix40/branches.json --out runs/branches-prefix40.gif
```

The renderer verifies each displayed source image against the capture receipt,
shows reset/replay/continuation phases, and preserves outcome/cost milestones.
The default keeps every second action, playing at eight actions per second
with additional holds at reset, the reconstructed root, and the final outcomes.
The columns align by action count, not elapsed execution time. Branches were
collected sequentially. Navigation images retain their original resolution;
manipulation images are reduced to fit. Trajectories show base X/Z position and
heading with a one-metre scale; room outlines omit doors and furniture.
Cyan marks the shared prefix, followed by each branch's own continuation color.
