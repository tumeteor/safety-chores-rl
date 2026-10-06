"""Run: python examples/fast_search.py --workers 2"""
import argparse
import json
from safety_chores_rl import (RuntimeSpec, Episode, RolloutSession, RolloutBudget,
    SearchBudget, BestOfN, SuccessiveHalving, BeamSearch, RecedingHorizon, reward_return)
from safety_chores_rl.toy import goal_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    spec = RuntimeSpec.create('safety_chores_rl.toy:GridRuntime', size=4, obstacles=[[1,0],[1,1]])
    with Episode(spec, seed=19, max_steps=16) as env, RolloutSession(spec, workers=args.workers) as session:
        root = session.capture_root(env)
        race = SuccessiveHalving(session, objective=reward_return).plan(root,
            [(0,), (1,), (2,), (3,)], sample_seeds=range(8), rollout_budget=RolloutBudget(8))
        print(json.dumps(dict(stage='racing', action=race.action, status=race.status, usage=race.usage)))
        session.release_root(root)
        planner = BeamSearch(session, depth=3, width=3, heuristic=goal_distance, branching_factor=4)
        controller = RecedingHorizon(session, planner)
        while not env.done:
            plan, step = controller.step(env, range(4), rollout_budget=RolloutBudget(min(3,16-len(env.actions))),
                search_budget=SearchBudget(max_rollouts=80, max_native_steps=2000, wall_seconds=30))
            print(json.dumps(dict(stage='replan', action=plan.action, position=env.observation['position'],
                                  status=plan.status, usage=plan.usage)))
            if step is None:
                break
        print(json.dumps(dict(success=env.runtime.pos == env.runtime.goal, steps=len(env.actions), cost=env.cost)))


if __name__ == '__main__':
    main()
