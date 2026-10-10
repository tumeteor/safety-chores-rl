"""Wire an existing SafeVLA installation into the native rollout examples.

Run with the installation's Python environment active. No upstream files are
modified. Optional checkpoint downloads use SafeVLA's official model repository.
"""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import urllib.request

from safety_chores_rl.native.config import file_hash, source_manifest, validate_config


PROJECT = Path(__file__).resolve().parents[1]
TASKS = {'fetch': 'FetchType', 'pickup': 'PickupType', 'objectnav': 'ObjectNavType'}
CHECKPOINTS = {
    'fetch': ('safe_fetch.pt', '3ae9076b27ba6564d6e7b919619ff2a84c45ebc70ead569d243f2fd91ff43e6c'),
    'pickup': ('safe_pickup.pt', 'c8ad41aef8860e2941327ef92143dc61e66b441df48be677021b1b21b54ea14b'),
    'objectnav': ('safe_objnav.pt', '05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301'),
}
UPSTREAM = 'https://github.com/PKU-Alignment/SafeVLA'
MODEL_REPO = 'https://huggingface.co/SafetyEmbodiedAI/safety-model'


def choose(label, explicit, candidates, predicate=None):
    predicate = predicate or (lambda p: p.is_file())
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not predicate(path):
            raise ValueError(f'{label} is missing or invalid: {path}')
        return path
    found = list(dict.fromkeys(Path(p).expanduser().resolve() for p in candidates if p and predicate(Path(p).expanduser())))
    if len(found) == 1:
        return found[0]
    if len(found) > 1 and all(p.is_file() for p in found):
        # Installations commonly retain duplicate benchmark/checkpoint files.
        # Identical bytes are interchangeable; different artifacts need a choice.
        if len({file_hash(p) for p in found}) == 1:
            return found[0]
    flag = '--' + label.replace('_', '-')
    if not found:
        raise ValueError(f'Cannot locate {label}. Set {flag}, or use --from-config with an existing native.json.')
    raise ValueError(f'Ambiguous {label}; select one with {flag}: ' + ', '.join(map(str, found)))


def expanded_dirs(roots, names):
    return [root / name for root in roots for name in names]


def discover(args, environ):
    """Prefer explicit inputs and environment settings; never guess between copies."""
    existing = args.from_config or environ.get('SAFETY_CHORES_NATIVE_CONFIG')
    if not existing and (args.output / 'native.json').is_file():
        existing = args.output / 'native.json'
    if existing:
        if any(getattr(args, name) for name in ('safevla_repo', 'asset_root', 'benchmark', 'houses', 'player', 'build_manifest')):
            raise ValueError('Use --from-config or individual native paths, not both; existing pins must remain intact.')
        config, rows = validate_config(json.loads(Path(existing).expanduser().read_text(encoding='utf-8')))
        task = next(key for key, value in TASKS.items() if value == config['task_type'])
        if args.task and args.task != task:
            raise ValueError('--task differs from the pinned native configuration')
        return config, rows, task

    task = args.task or 'fetch'
    base = args.installation_root.expanduser().resolve(strict=True)
    roots = [base, base / 'data', base.parent / 'data']
    repo = choose('safevla_repo', args.safevla_repo or environ.get('SAFEVLA_REPO'),
        expanded_dirs([base, base.parent, Path.home()], ['', 'SafeVLA', 'safevla', 'source/SafeVLA'])
        + [Path(p) for p in environ.get('PYTHONPATH', '').split(os.pathsep) if p],
        lambda p: (p / 'tasks/multi_task_eval_sampler.py').is_file()
                  and (p / 'training/online/dinov2_vits_tsfm_base.py').is_file())
    roots = list(dict.fromkeys([repo, *roots]))
    stem = TASKS[task].lower()
    benchmark = choose('benchmark', args.benchmark,
        expanded_dirs(roots, [f'benchmark/{stem}_val.jsonl.gz', f'benchmark/{stem}.jsonl.gz',
                              f'safety-chores/{stem}.jsonl.gz']))
    house_dirs = expanded_dirs(roots, ['houses_2023_07_28', 'objaverse_houses/houses_2023_07_28',
                                      'objaverse_houses', 'houses'])
    houses = choose('houses', args.houses or
        (Path(environ['OBJAVERSE_HOUSES_DIR']) / 'val.jsonl.gz' if environ.get('OBJAVERSE_HOUSES_DIR') else None),
        [p / 'val.jsonl.gz' for p in house_dirs])
    data_dirs = expanded_dirs(roots, ['objaverse_assets/2023_07_28', 'assets/2023_07_28', '2023_07_28'])
    data_dirs.append(Path.home() / '.objathor-assets/2023_07_28')
    if environ.get('OBJAVERSE_DATA_DIR'):
        path = Path(environ['OBJAVERSE_DATA_DIR']).expanduser()
        data_dirs = [path if path.name == '2023_07_28' else path / '2023_07_28']
    assets = choose('asset_root', args.asset_root or environ.get('OBJAVERSE_ASSETS_DIR'),
                    [p / 'assets' for p in data_dirs], lambda p: p.is_dir())
    player_dirs = expanded_dirs(roots, ['player', 'native/player', '.native/player', 'builds'])
    player = choose('player', args.player or environ.get('SAFETY_CHORES_PLAYER'),
        [p / 'thor-Linux64-local' for p in player_dirs]
        + [f for p in player_dirs if p.is_dir() for f in p.glob('*/thor-Linux64-local')])
    manifest = choose('build_manifest', args.build_manifest,
                      [player.parent / 'checkpoint-build.json'])
    config = dict(schema_version=1, task_type=TASKS[task], split='val', asset_root=str(assets),
                  safevla=dict(repo_path=str(repo), source_files=source_manifest(repo)))
    for name, path in dict(benchmark=benchmark, houses=houses, player=player, build_manifest=manifest).items():
        config[name] = dict(path=str(path), sha256=file_hash(path))
    config, rows = validate_config(config)
    return config, rows, task


def verify_player(config):
    player = Path(config['player']['path'])
    manifest = json.loads(Path(config['build_manifest']['path']).read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'safety_chores_checkpoint_build_v1'
            or manifest.get('executableSha256') != config['player']['sha256']
            or not manifest.get('artifacts')):
        raise ValueError('The player needs its matching Safety-CHORES checkpoint-build.json; a stock player is insufficient.')
    for name, expected in manifest['artifacts'].items():
        target = (player.parent / name).resolve()
        if not target.is_relative_to(player.parent) or not target.is_file() or file_hash(target) != expected:
            raise ValueError(f'Player artifact missing, changed, or outside its directory: {name}')
    if sys.platform == 'linux' and not os.access(player, os.X_OK):
        raise ValueError(f'Player is not executable: {player}')
    return len(manifest['artifacts'])


def verify_assets(config, rows, task_ids):
    if any(i < 0 or i >= len(rows) for i in task_ids):
        raise ValueError(f'--task-ids must be in 0..{len(rows)-1}')
    required_houses = {int(rows[i]['house_index']) for i in task_ids}
    found, assets = set(), set()

    def collect(objects):
        for obj in objects:
            uid = obj.get('assetId', '')
            if re.fullmatch(r'[a-fA-F0-9]{32}', uid):
                assets.add(uid)
            collect(obj.get('children', []))

    with gzip.open(config['houses']['path'], 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index in required_houses:
                collect(json.loads(line)['objects'])
                found.add(index)
            if found == required_houses:
                break
    if found != required_houses:
        raise ValueError(f'Houses file lacks indices: {sorted(required_houses - found)}')
    base = Path(config['asset_root'])
    missing = [uid for uid in sorted(assets) if any(not (base / uid / name).is_file()
               for name in (f'{uid}.pkl.gz', 'thor_metadata.json', 'albedo.jpg', 'normal.jpg', 'emission.jpg'))]
    if missing:
        raise ValueError(f'{len(missing)} Objaverse assets missing for task IDs {task_ids}; first: {missing[0]}. Set --asset-root to the populated asset directory.')
    return dict(task_ids=task_ids, houses=sorted(found), objaverse_assets=len(assets))


def download_checkpoint(url, target, expected):
    """Only promote a complete, hash-verified download; keep existing files intact."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != expected:
            raise ValueError(f'Existing checkpoint has a different SHA256: {target}')
        return target
    partial = target.with_name(target.name + '.download')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        if file_hash(partial) != expected:
            raise ValueError('Downloaded checkpoint SHA256 differs from the official pinned weights')
        os.link(partial, target)  # Atomic promotion without replacing an existing file.
        partial.unlink()
    except BaseException:
        # Only remove this invocation's incomplete file, never an existing checkpoint.
        if partial.exists() and 'out' in locals():
            partial.unlink()
        raise
    return target


def policy_config(args, config, task, environ):
    filename, expected = CHECKPOINTS[task]
    repo = Path(config['safevla']['repo_path'])
    roots = list(dict.fromkeys([repo, args.installation_root.resolve(), repo.parent, args.output.resolve()]))
    candidates = expanded_dirs(roots, [f'checkpoints/{filename}', f'checkpoints/{Path(filename).stem}/{filename}',
                                      f'ckpt/{filename}', f'data/safe_aligned_ckpt/{task}/{filename}'])
    # Already-generated policy pins take precedence and are verified again.
    existing = args.output / 'policy.json'
    explicit = args.checkpoint or environ.get('SAFEVLA_CHECKPOINT')
    if not explicit and existing.is_file():
        policy = json.loads(existing.read_text(encoding='utf-8'))
        if file_hash(policy['checkpoint']) != policy['checkpoint_sha256']:
            raise ValueError('Existing policy checkpoint SHA256 mismatch')
        policy['device'] = args.device
        return policy
    if not explicit and not any(p.is_file() for p in candidates) and args.download_checkpoint:
        target = args.output.resolve() / 'checkpoints' / filename
        print(f'Downloading the official {filename} (~2 GB) from {MODEL_REPO}', flush=True)
        explicit = download_checkpoint(f'{MODEL_REPO}/resolve/main/{filename}', target, expected)
    path = choose('checkpoint', explicit, candidates)
    actual = file_hash(path)
    # An explicit path allows a custom checkpoint, while automatic discovery
    # must select the named official weights rather than another controller.
    if not explicit and actual != expected:
        raise ValueError(f'Automatically found {filename} differs from the official checkpoint. Select a custom checkpoint explicitly with --checkpoint.')
    return dict(checkpoint=str(path), checkpoint_sha256=actual, device=args.device, greedy=False)


def environment(config, args, environ):
    assets = Path(config['asset_root'])
    annotations = choose('annotations', args.annotations or environ.get('OBJAVERSE_ANNOTATIONS_PATH'),
                         [assets.parent / 'annotations.json.gz'])
    with gzip.open(annotations, 'rt', encoding='utf-8') as stream:
        if not isinstance(json.load(stream), dict):
            raise ValueError('Objaverse annotations must contain a JSON object')
    values = dict(OBJAVERSE_ASSETS_DIR=str(assets), OBJAVERSE_DATA_DIR=str(assets.parent),
        OBJAVERSE_HOUSES_DIR=str(Path(config['houses']['path']).parent),
        OBJAVERSE_ANNOTATIONS_PATH=str(annotations), PYTHONHASHSEED='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        values[name] = environ.get(name, '1')
    for name in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'TORCH_HOME', 'HF_HOME', 'HF_HUB_CACHE', 'NLTK_DATA'):
        if name in environ:
            values[name] = environ[name]
    torch_home = Path(environ.get('TORCH_HOME', str(Path.home() / '.cache/torch')))
    dino = environ.get('DINOV2_REPO_DIR') or (torch_home / 'hub/facebookresearch_dinov2_main')
    if Path(dino).is_dir():
        values['DINOV2_REPO_DIR'] = str(Path(dino).resolve())
    t5 = environ.get('SAFEVLA_T5_PATH')
    if not t5:
        hf_home = Path(environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
        hub = Path(environ.get('HF_HUB_CACHE', str(hf_home / 'hub')))
        for name in ('models--google-t5--t5-small', 'models--t5-small'):
            ref = hub / name / 'refs/main'
            if ref.is_file():
                revision = ref.read_text().strip()
                if re.fullmatch(r'[a-f0-9]{40}', revision):
                    candidate = hub / name / 'snapshots' / revision
                    if (candidate / 'config.json').is_file():
                        t5 = str(candidate)
                        break
    if t5:
        values['SAFEVLA_T5_PATH'] = t5
        values['SAFEVLA_LOCAL_FILES_ONLY'] = environ.get('SAFEVLA_LOCAL_FILES_ONLY', '1')
    return values


def write_outputs(args, config, policy, values, summary):
    out = args.output.expanduser().resolve()
    files = {'native.json': json.dumps(config, indent=2) + '\n',
             'policy.json': json.dumps(policy, indent=2) + '\n',
             'setup.json': json.dumps(summary, indent=2) + '\n'}
    paths = str(PROJECT / 'src') + ':' + config['safevla']['repo_path']
    files['env.sh'] = '# Generated for this installation. Source from bash.\n' + '\n'.join(
        f'export {key}={shlex.quote(value)}' for key, value in sorted(values.items())) + '\n'
    files['env.sh'] += f'export PYTHONPATH={shlex.quote(paths)}${{PYTHONPATH:+:"$PYTHONPATH"}}\n'
    command = [sys.executable, str(PROJECT / 'examples/native_first_success.py'),
               '--config', str(out / 'native.json'), '--policy-config', str(out / 'policy.json'),
               '--out', str(out / 'first-success'), '--task-id', str(args.task_ids[0]),
               '--vulkan-device', str(args.vulkan_device)]
    files['run_demo.sh'] = ('#!/usr/bin/env bash\nset -euo pipefail\n'
        f'source {shlex.quote(str(out / "env.sh"))}\n'
        'display=()\nif [[ -z "${DISPLAY:-}" ]]; then\n'
        '  command -v xvfb-run >/dev/null || { echo "Install xvfb and xauth, or set DISPLAY." >&2; exit 1; }\n'
        '  display=(xvfb-run -a)\nfi\n'
        f'exec "${{display[@]}}" {shlex.join(command)} "$@"\n')
    # Refuse a changed configuration before writing any file. Repeating the
    # identical setup is a no-op; use another output directory for new inputs.
    for name, content in files.items():
        path = out / name
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Refusing to replace existing setup: {path}. Use a new --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = out / name
        if not path.exists():
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
    (out / 'run_demo.sh').chmod(0o755)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--installation-root', type=Path, default=PROJECT.parent,
                   help='Existing installation root; also checks standard sibling directories and environment variables')
    p.add_argument('--from-config', type=Path, help='Reuse and verify an already-pinned native.json')
    p.add_argument('--task', choices=TASKS, help='Default: fetch, or the task in --from-config')
    p.add_argument('--task-ids', type=int, nargs='+', default=[0], help='Check asset availability for these rows; does not filter the benchmark')
    p.add_argument('--output', type=Path, default=Path('runs/native-setup'))
    for name in ('safevla-repo', 'asset-root', 'benchmark', 'houses', 'player', 'build-manifest', 'checkpoint', 'annotations'):
        p.add_argument('--' + name, type=Path, help='Override automatic discovery')
    p.add_argument('--download-checkpoint', action='store_true', help='Download official task weights if absent (~2 GB)')
    p.add_argument('--device', default='cuda:0', help='Policy CUDA device')
    p.add_argument('--vulkan-device', type=int, default=0, help='Physical simulator GPU index')
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        if sys.platform != 'linux':
            raise ValueError('Native setup requires Linux; activate the existing SafeVLA Python environment there.')
        if args.vulkan_device < 0:
            raise ValueError('--vulkan-device must be nonnegative')
        missing = [name for name in ('torch', 'numpy', 'ai2thor', 'allenact', 'objathor', 'transformers')
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError('Activate the existing SafeVLA Python environment; missing packages: ' + ', '.join(missing))
        config, rows, task = discover(args, os.environ)
        print(f'Found {len(rows)} {config["task_type"]} benchmark rows; checking the installation.', flush=True)
        artifacts = verify_player(config)
        coverage = verify_assets(config, rows, args.task_ids)
        values = environment(config, args, os.environ)
        policy = policy_config(args, config, task, os.environ)
        summary = dict(schema_version=1, status='configured', upstream=UPSTREAM,
            benchmark_task_type=config['task_type'], benchmark_rows=len(rows),
            asset_preflight=coverage, player_artifacts_checked=artifacts,
            checkpoint_sha256=policy['checkpoint_sha256'], python=sys.executable,
            scope='Local artifact checks only; run the generated demo for native execution checks.')
        output = write_outputs(args, config, policy, values, summary)
        print(json.dumps({**summary, 'output': str(output), 'run': f'bash {shlex.quote(str(output / "run_demo.sh"))}'}))
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f'Setup incomplete: {exc}\n')


if __name__ == '__main__':
    main()
