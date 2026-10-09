from dataclasses import replace

import pytest

from safety_chores_rl import (Episode, RuntimeSpec, RolloutSession, RolloutBudget,
    SearchBudget, FirstSuccess, TrajectoryMismatch, StalePlanError)


def spec(**config):
    return RuntimeSpec.create('safety_chores_rl.toy:GridRuntime', size=2, **config)


def test_first_success_stops_after_one_wave_and_executes_checked_full_path():
    runtime = spec()
    with Episode(runtime, max_steps=6) as env, RolloutSession(runtime, workers=2) as session:
        root = session.capture_root(env)
        plan = FirstSuccess(session).plan(root, [(0, 1), (1, 0), (0, 0)],
            sample_seeds=[40, 41, 42, 43], rollout_budget=RolloutBudget(6))
        assert plan.status == 'ok' and plan.candidate == (0, 1)
        assert plan.usage['rollouts'] == 2 < plan.usage['exhaustive_rollouts'] == 12
        assert len(plan.witness.trace) == 3 and env.matches(root)
        checks = []
        steps = session.execute_trajectory(env, plan,
            on_step=lambda episode, transition, index, matched: checks.append(matched))
        assert steps[-1].success and env.done and env.cost == 0
        assert checks == [True, True] and not session._roots


def test_no_safe_trajectory_returns_no_action_and_preserves_live_episode():
    runtime = spec(hazards=[[1, 1]])
    with Episode(runtime, max_steps=2) as env, RolloutSession(runtime) as session:
        root = session.capture_root(env)
        plan = FirstSuccess(session).plan(root, [(0, 1), (1, 0)],
            sample_seeds=[1], rollout_budget=RolloutBudget(2))
        assert plan.status == 'no_successful_trajectory' and plan.witness is None
        assert plan.action is None and env.matches(root)
        with pytest.raises(ValueError, match='witness'):
            session.execute_trajectory(env, plan)
        assert not env.actions


def test_execution_drift_stops_before_next_action():
    runtime = spec()
    with Episode(runtime, max_steps=4) as env, RolloutSession(runtime) as session:
        root = session.capture_root(env)
        plan = FirstSuccess(session).plan(root, [(0, 1)], sample_seeds=[1], rollout_budget=RolloutBudget(4))
        def perturb(episode, *args):
            episode.runtime.pos = (0, 0)
        with pytest.raises(TrajectoryMismatch, match='after action 0'):
            session.execute_trajectory(env, plan, on_step=perturb)
        assert env.actions == [0] and not session._roots


def test_stale_or_forged_trace_is_rejected_before_execution():
    runtime = spec()
    with Episode(runtime, max_steps=4) as env, RolloutSession(runtime) as session:
        root = session.capture_root(env)
        plan = FirstSuccess(session).plan(root, [(0, 1)], sample_seeds=[1], rollout_budget=RolloutBudget(4))
        altered = replace(plan, witness=replace(plan.witness, trace=()))
        with pytest.raises(ValueError, match='witness'):
            session.execute_trajectory(env, altered)
        assert not env.actions
        env.step(1)
        with pytest.raises(StalePlanError):
            session.execute_trajectory(env, plan)


def test_trace_flag_changes_cache_identity_and_budget_exhaustion_is_explicit():
    runtime = spec()
    with Episode(runtime, max_steps=4) as env, RolloutSession(runtime) as session:
        root = session.capture_root(env)
        normal = session.rollout(root, (0, 1), budget=RolloutBudget(4))
        traced = session.rollout(root, (0, 1), budget=RolloutBudget(4, record_trace=True))
        assert not normal.trace and traced.trace and not traced.cache_hit
        assert normal.request_id != traced.request_id
        plan = FirstSuccess(session).plan(root, [(0, 0), (1, 0)], sample_seeds=[7],
            rollout_budget=RolloutBudget(4), search_budget=SearchBudget(max_native_steps=1))
        assert plan.status == 'budget_exhausted' and plan.action is None
