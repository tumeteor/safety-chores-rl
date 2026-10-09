from types import SimpleNamespace

import pytest

from examples.native_task_study import run_arm
from safety_chores_rl import RuntimeSpec


@pytest.mark.parametrize('search', [False, True])
def test_study_accounts_for_full_execution_and_exact_replay(search):
    spec = RuntimeSpec.create('tests.runtime_fixtures:InspectableGridRuntime', size=2)
    args = SimpleNamespace(task_id=0, seed=892, episode_limit=6, workers=1,
        query_steps=[0, 2], branch_seconds=10, candidates=2, sample_seeds=[71, 72], query_seconds=20)
    result = dict(arm='search' if search else 'baseline', actions=[], costs=[], rewards=[],
                  queries=[], success=None, valid=False, replay_exact=False)
    saved = []
    run_arm(spec, args, search, result, lambda: saved.append(len(result['actions'])))
    assert result['valid'] and result['replay_exact'] and not result['replay_mismatches']
    assert len(result['actions']) == len(result['costs']) == len(result['rewards'])
    assert result['success'] is not None and saved
    assert bool(result['queries']) == search
    assert all(row['live_unchanged'] for row in result['queries'])
