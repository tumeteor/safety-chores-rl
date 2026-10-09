from dataclasses import replace
import json
import os
from pathlib import Path
import random
import threading
import time

import pytest

from safety_chores_rl import (Episode, RolloutSession, RuntimeSpec, RolloutBudget,
    SearchBudget, BestOfN, StalePlanError, reward_return)
from safety_chores_rl.types import canonical


def grid(**kwargs):
    return RuntimeSpec.create('safety_chores_rl.toy:GridRuntime', **kwargs)


def bandit(**kwargs):
    return RuntimeSpec.create('tests.runtime_fixtures:BanditRuntime', **kwargs)


def test_replay_isolation_cache_dedup_and_serialization():
    spec = grid()
    with Episode(spec, seed=19, max_steps=15) as env, RolloutSession(spec, workers=2) as session:
        env.step(0)
        before = canonical(env.runtime.snapshot())
        global_rng = random.getstate()
        root = session.capture_root(env)
        assert session.validate_root(root).valid
        results = session.rollout_batch(root, [(0,), (1,), (0,)], sample_seeds=[5, 6],
            budget=RolloutBudget(5), search_budget=SearchBudget(wall_seconds=20))
        assert len(results) == 6 and all(r.valid for r in results)
        assert sum(r.cache_hit for r in results) == 2
        again = session.rollout(root, (0,), sample_seed=5, budget=RolloutBudget(5))
        assert again.cache_hit and again.native_steps == 0
        assert canonical(env.runtime.snapshot()) == before and random.getstate() == global_rng
        json.dumps(again.to_dict(), allow_nan=False)
        # Reuse does not skip independent root validation.
        assert not session.validate_root(root).cache_hit


def test_prefix_full_episode_and_positive_cost_have_distinct_outcomes():
    spec = grid(hazards=[[1, 0]])
    with Episode(spec, max_steps=3) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        prefix = session.rollout(root, (1,), continuation=False, budget=RolloutBudget(1))
        assert prefix.success is None and prefix.safe_success is None and prefix.truncated
        unsafe = session.rollout(root, (0,), budget=RolloutBudget(3))
        assert unsafe.safe_success is False and unsafe.success is None
        assert unsafe.stop_reason == 'positive_cost' and len(unsafe.actions) == 1
        capped = session.rollout(root, (1, 1, 1), continuation=False, budget=RolloutBudget(3))
        assert capped.success is False and capped.safe_success is False
        assert capped.stop_reason == 'episode_limit'


def test_root_mismatch_is_invalid_and_not_a_negative_vote():
    spec = bandit(drift=True)
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        result = session.rollout(root, (0,), budget=RolloutBudget(2))
        assert result.status == 'root_mismatch' and result.safe_success is None
        assert 'process' in json.loads(result.checks_json)['mismatches']
        assert not session.rollout(root, (0,), budget=RolloutBudget(2)).cache_hit


def test_partial_error_does_not_erase_good_candidates():
    spec = bandit(fail_action=2)
    with Episode(spec) as env, RolloutSession(spec, workers=2) as session:
        root = session.capture_root(env)
        plan = BestOfN(session, objective=reward_return).plan(root, [(1,), (2,), (3,)],
            sample_seeds=[0], rollout_budget=RolloutBudget(2))
        assert plan.action == 3
        failed = next(e for e in plan.evaluations if e.candidate == (2,))
        assert failed.score is None and failed.results[0].success is None


def test_runtime_errors_report_reserved_work_and_are_not_cached():
    spec = bandit(fail_action=2)
    with Episode(spec, max_steps=4) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        for _ in range(2):
            result = session.rollout(root, (2,), budget=RolloutBudget(4))
            assert result.status == 'error' and result.success is None
            assert result.native_steps == 4 and not result.cache_hit
        plan = BestOfN(session).plan(root, [(2,), (3,)], sample_seeds=[0],
            rollout_budget=RolloutBudget(4), search_budget=SearchBudget(max_native_steps=4))
        assert plan.status == 'budget_exhausted' and plan.action is None
        assert plan.usage['native_steps'] == 4


def test_foreign_released_and_stale_roots_rejected():
    spec = grid()
    with Episode(spec) as env, RolloutSession(spec) as session, RolloutSession(spec) as other:
        root = session.capture_root(env)
        with pytest.raises(ValueError, match='Foreign'):
            other.rollout(root, (0,), budget=RolloutBudget(1))
        plan = BestOfN(session, objective=reward_return).plan(root, [(0,)], sample_seeds=[0],
            continuation=False, rollout_budget=RolloutBudget(1))
        env.step(1)
        with pytest.raises(StalePlanError):
            session.execute_first(env, plan)
        session.release_root(root)
        with pytest.raises(ValueError, match='released'):
            session.rollout(root, (0,), budget=RolloutBudget(1))


def test_episode_prefix_consumes_budget_and_rejected_request_does_no_work():
    spec = grid()
    with Episode(spec, max_steps=4) as env, RolloutSession(spec) as session:
        env.step(0)
        root = session.capture_root(env)
        with pytest.raises(ValueError, match='remaining'):
            session.rollout(root, (0,), budget=RolloutBudget(4))
        result = session.rollout(root, (1, 1, 1), continuation=False, budget=RolloutBudget(3))
        assert result.replay_steps == 1 and result.native_steps == 4
        assert result.stop_reason == 'episode_limit'


def test_global_budget_bounds_parallel_admission():
    spec = grid()
    with Episode(spec) as env, RolloutSession(spec, workers=4) as session:
        root = session.capture_root(env)
        plan = BestOfN(session, objective=reward_return).plan(root, [(0,), (1,), (2,), (3,)],
            sample_seeds=[0, 1], rollout_budget=RolloutBudget(4),
            search_budget=SearchBudget(max_rollouts=3, max_native_steps=8))
        assert plan.usage['rollouts'] <= 3 and plan.usage['native_steps'] <= 8
        assert plan.status == 'budget_exhausted'


def test_seed_forks_after_all_candidate_feedback():
    spec = RuntimeSpec.create('tests.runtime_fixtures:SeedTimingRuntime')
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        result = session.rollout(root, (0, 1), continuation=False, budget=RolloutBudget(2))
        assert result.valid, result.error


def test_timeout_kills_worker_and_later_request_still_runs():
    spec = bandit(delay=0.3)
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        result = session.rollout(root, (0,), budget=RolloutBudget(1, timeout_seconds=0.05))
        assert result.status == 'timeout' and result.success is None
        result = session.rollout(root, (1,), budget=RolloutBudget(1, timeout_seconds=10))
        assert result.valid
        assert session._pool.slots[0].process.is_alive()
    assert session._pool.slots[0].process is None


def test_cancellation_retains_records_and_stops_owned_worker():
    spec = bandit(delay=2)
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        timer = threading.Timer(0.2, session.cancel)
        timer.start()
        try:
            results = session.rollout_batch(root, [(0,), (1,)], budget=RolloutBudget(1))
        finally:
            timer.cancel()
        assert len(results) == 2 and all(r.status == 'cancelled' for r in results)


def test_cache_identity_includes_root_seed_horizon_and_stop_rule():
    spec = grid()
    with Episode(spec, seed=1) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        first = session.rollout(root, (0,), sample_seed=1, budget=RolloutBudget(2))
        variants = [session.rollout(root, (0,), sample_seed=2, budget=RolloutBudget(2)),
                    session.rollout(root, (0,), sample_seed=1, budget=RolloutBudget(3)),
                    session.rollout(root, (0,), sample_seed=1, budget=RolloutBudget(2, stop_on_cost=False))]
        assert all(not r.cache_hit and r.request_id != first.request_id for r in variants)


@pytest.mark.parametrize('value', [True, -1, 1.1])
def test_reject_invalid_seed(value):
    spec = grid()
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        with pytest.raises(ValueError):
            session.rollout(root, (0,), sample_seed=value, budget=RolloutBudget(1))


def test_identical_new_episode_cannot_execute_another_episodes_plan():
    spec = grid()
    with Episode(spec) as env, Episode(spec) as other, RolloutSession(spec) as session:
        root = session.capture_root(env)
        plan = BestOfN(session, objective=reward_return).plan(root, [(0,)], sample_seeds=[0],
            continuation=False, rollout_budget=RolloutBudget(1))
        with pytest.raises(StalePlanError):
            session.execute_first(other, plan)
        assert not other.actions
        assert session.execute_first(env, plan).action == 0


def test_early_terminal_refunds_admit_pending_wave_work():
    spec = bandit()
    with Episode(spec) as env, RolloutSession(spec, workers=4) as session:
        root = session.capture_root(env)
        plan = BestOfN(session, objective=reward_return).plan(root, [(i,) for i in range(4)],
            sample_seeds=[0], rollout_budget=RolloutBudget(4),
            search_budget=SearchBudget(max_rollouts=4, max_native_steps=7))
        assert plan.status == 'ok' and plan.action == 3
        assert plan.usage['rollouts'] == 4 and plan.usage['native_steps'] == 4


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process-group regression')
def test_timeout_stops_owned_posix_descendant(tmp_path):
    pid_path = tmp_path/'child.pid'
    spec = RuntimeSpec.create('tests.runtime_fixtures:DescendantRuntime', pid_path=str(pid_path))
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        result = session.rollout(root, (0,), budget=RolloutBudget(1, timeout_seconds=2))
        assert result.status == 'timeout'
        pid = int(pid_path.read_text())
        stat = Path(f'/proc/{pid}/stat')
        for _ in range(100):
            if not stat.exists() or stat.read_text().split(') ')[1][0] == 'Z':
                break
            time.sleep(0.02)
        else:
            os.kill(pid, 9)
            pytest.fail('Owned descendant survived worker timeout')
