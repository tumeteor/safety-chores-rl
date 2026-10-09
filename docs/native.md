# Native validation on an existing installation

The native adapter currently requires Linux, an NVIDIA/Vulkan environment,
the modified simulator player, a compatible SafeVLA installation, and local
benchmark tasks, houses, and assets. Installing this Python package does not
install those components. A portable simulator build/install recipe remains a
release gate. The Python-only examples do not establish native task success.

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
action replays, and reproduce the result on a clean installation. A declared
oracle or scripted fixture must be labeled as such.
