"""Explicit local artifact configuration; no downloads or policy checkpoints."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

TASK_TYPES = {'FetchType', 'PickupType', 'ObjectNavType'}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_manifest(repo):
    repo = Path(repo)
    return {p.relative_to(repo).as_posix(): file_hash(p) for p in sorted(repo.rglob('*.py'))
            if not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(repo).parts)}


def validate_config(config):
    config = copy.deepcopy(config)
    if config.get('schema_version') != 1:
        raise ValueError('Native config requires schema_version=1')
    if config.get('task_type') not in TASK_TYPES:
        raise ValueError('Unsupported benchmark task_type')
    if config.get('split', 'val') not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val, or test')
    for name in ('benchmark', 'houses', 'build_manifest', 'player'):
        path = Path(config[name]['path']).expanduser().resolve(strict=True)
        if file_hash(path) != config[name]['sha256']:
            raise ValueError(f'{name} SHA256 mismatch')
        config[name]['path'] = str(path)
    repo = Path(config['safevla']['repo_path']).expanduser().resolve(strict=True)
    expected = config['safevla'].get('source_files')
    if not expected or source_manifest(repo) != expected:
        raise ValueError('External benchmark source differs from the declared Python source manifest')
    config['safevla']['repo_path'] = str(repo)
    config['asset_root'] = str(Path(config['asset_root']).expanduser().resolve(strict=True))
    with gzip.open(config['benchmark']['path'], 'rt', encoding='utf-8') as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    if not tasks or any(row.get('task_type') != config['task_type'] for row in tasks):
        raise ValueError('Benchmark rows must match the declared task_type')
    return config, tasks
