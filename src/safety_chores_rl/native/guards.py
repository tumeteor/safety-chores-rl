"""Scoped native seed acknowledgements and owned asset staging."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def checked_native_seeds(controller_class, receipts):
    original = controller_class.step

    def checked(self, *args, **kwargs):
        command = args[0] if args else kwargs.get('action')
        action = command.get('action') if isinstance(command, dict) else command
        if action != 'SetRandomSeed':
            return original(self, *args, **kwargs)
        requested = command['seed'] if isinstance(command, dict) else kwargs['seed']
        if isinstance(requested, bool) or int(requested) != requested or requested < 0 or requested >= 2**31:
            raise ValueError('Native episode seed must fit nonnegative Int32; no silent remapping')
        event = original(self, *args, **kwargs)
        if event.metadata.get('lastActionSuccess') is not True:
            raise RuntimeError('Native SetRandomSeed failed')
        receipts.append(int(requested))
        return event

    controller_class.step = checked
    try:
        yield
    finally:
        controller_class.step = original


@contextmanager
def private_asset_staging(build_class, player, directory):
    """Scope AI2-THOR's asset cache to our player without altering its build tree."""
    original = build_class.base_dir
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    player = Path(player).resolve()

    def base_dir(build):
        if Path(build.executable_path).resolve() == player:
            return str(directory)
        return original.__get__(build, build_class)

    build_class.base_dir = property(base_dir)
    try:
        yield
    finally:
        build_class.base_dir = original

