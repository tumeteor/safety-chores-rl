from safety_chores_rl import (Episode, RolloutSession, RuntimeSpec, RolloutBudget,
    SearchBudget, BestOfN, SuccessiveHalving, BeamSearch, RecedingHorizon, reward_return)
from safety_chores_rl.toy import goal_distance
import pytest


def test_halving_finds_known_best_with_fewer_native_calls():
    spec = RuntimeSpec.create('tests.runtime_fixtures:BanditRuntime')
    candidates = [(i,) for i in range(8)]
    with Episode(spec) as env, RolloutSession(spec, workers=2) as session:
        root = session.capture_root(env)
        fast = SuccessiveHalving(session, objective=reward_return).plan(root, candidates,
            rollout_budget=RolloutBudget(1), sample_seeds=range(8))
        assert fast.action == 7 and fast.status == 'ok'
        assert fast.usage['rollouts'] == 20 < 64
        assert fast.usage['rounds'] == 4
        full = BestOfN(session, objective=reward_return).plan(root, candidates,
            rollout_budget=RolloutBudget(1), sample_seeds=range(8))
        assert full.action == fast.action and full.usage['cache_hits'] == 20


def test_candidate_order_and_parallelism_preserve_seeded_results():
    spec = RuntimeSpec.create('safety_chores_rl.toy:GridRuntime', size=3)
    results = []
    with Episode(spec) as env:
        for workers, candidates in [(1, [(0,), (1,)]), (2, [(1,), (0,)])]:
            with RolloutSession(spec, workers=workers) as session:
                root = session.capture_root(env)
                plan = BestOfN(session, objective=reward_return).plan(root, candidates,
                    rollout_budget=RolloutBudget(10), sample_seeds=[5, 9, 12])
                results.append([(e.candidate, e.score, [(r.actions, r.rewards, r.costs) for r in e.results])
                                for e in plan.evaluations])
    assert results[0] == results[1]


def test_beam_routes_around_blocked_direct_path_and_replans():
    spec = RuntimeSpec.create('safety_chores_rl.toy:GridRuntime', size=3, obstacles=[[1, 0]])
    with Episode(spec, max_steps=8) as env, RolloutSession(spec, workers=2) as session:
        planner = BeamSearch(session, depth=3, width=3, branching_factor=4, heuristic=goal_distance)
        controller = RecedingHorizon(session, planner)
        first, step = controller.step(env, range(4), rollout_budget=RolloutBudget(3))
        assert first.action == 1 and step.action == 1
        assert not session._roots
        for _ in range(6):
            if env.done:
                break
            remaining = env.max_steps-len(env.actions)
            plan, step = controller.step(env, range(4), rollout_budget=RolloutBudget(min(3, remaining)))
            assert step is not None
        assert env.runtime.pos == (2, 2)


def test_horizon_unknown_not_ranked_as_zero_safe_completion():
    spec = RuntimeSpec.create('safety_chores_rl.toy:GridRuntime')
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        plan = BestOfN(session).plan(root, [(0,), (1,)], rollout_budget=RolloutBudget(1),
                                   continuation=False, sample_seeds=[0])
        assert plan.status == 'insufficient_evidence' and plan.action is None


def test_admissibility_and_bounded_candidate_generator():
    spec = RuntimeSpec.create('tests.runtime_fixtures:BanditRuntime')
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        planner = BestOfN(session)
        plan = planner.plan(root, [(0,)], rollout_budget=RolloutBudget(1), admissible=lambda *_: False)
        assert plan.status == 'no_admissible_plan' and plan.usage['rollouts'] == 0
        with pytest.raises(ValueError, match='max_candidates'):
            planner.plan(root, ((i,) for i in range(100)), rollout_budget=RolloutBudget(1),
                         search_budget=SearchBudget(max_candidates=3))


def test_duplicate_samples_are_not_independent_evidence():
    spec = RuntimeSpec.create('tests.runtime_fixtures:BanditRuntime')
    with Episode(spec) as env, RolloutSession(spec) as session:
        root = session.capture_root(env)
        with pytest.raises(ValueError, match='distinct'):
            BestOfN(session).plan(root, [(0,)], sample_seeds=[0, 0], rollout_budget=RolloutBudget(1))


def test_receding_horizon_no_plan_has_no_implicit_live_action():
    spec = RuntimeSpec.create('safety_chores_rl.toy:GridRuntime')
    with Episode(spec) as env, RolloutSession(spec) as session:
        controller = RecedingHorizon(session, BestOfN(session))
        plan, step = controller.step(env, [(0,)], rollout_budget=RolloutBudget(1))
        assert step is None and not env.actions and not session._roots
        controller.fallback = lambda observation, plan: 1
        plan, step = controller.step(env, [(0,)], rollout_budget=RolloutBudget(1))
        assert step.action == 1 and env.actions == [1] and not session._roots
