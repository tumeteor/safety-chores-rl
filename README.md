# Safety-CHORES-RL

Simulator tooling, isolated replay rollouts, and budgeted search for Safety-CHORES.

This is a local release draft. The simulator uses external Safety-CHORES tasks and
assets. Generic planners and synthetic examples run with Python alone; native
integration and release validation are in progress.

## Try the portable examples

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python examples/fast_search.py --workers 2
python examples/benchmark_search.py --workers 2
```

See [API semantics](docs/api.md) for the runtime and search contracts.
