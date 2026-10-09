import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')

from safety_chores_rl.native.safevla_policy import SafeVLAPolicy, _get_rng


def fixture_policy():
    policy = SafeVLAPolicy.__new__(SafeVLAPolicy)
    cache = SimpleNamespace(cache_k=torch.zeros(1, 4, 2), cache_v=torch.zeros(1, 4, 2))
    model = SimpleNamespace(named_modules=lambda: [('decoder', cache)])
    agent = SimpleNamespace(last_action_flat=torch.tensor([[0]]), actor_critic=model,
                            steps_taken_in_task=0, num_evaluated_traj=1, memory=None)
    def forward(sensors, instruction):
        random.random()
        np.random.random()
        torch.rand(1)
        agent.last_action_flat.fill_(0)
        agent.steps_taken_in_task += 1
        return 'left', torch.tensor([0.75, 0.25])
    agent.get_action = forward
    policy.agent, policy.rng = agent, _get_rng()
    policy.actions = ['left', 'right']
    policy.history, policy.pending = [], False
    policy.checkpoint_sha256, policy.load_status = 'fixture', dict(missing=[], unexpected=[])
    return policy, cache


def assert_same_rng(first, second):
    assert first[0] == second[0]
    assert first[1][0] == second[1][0] and np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])
    assert all(torch.equal(a, b) for a, b in zip(first[3], second[3]))


def test_learned_feedback_records_executed_action_and_isolates_rng():
    policy, _ = fixture_policy()
    external = _get_rng()
    policy.seed_suffix(192)
    assert_same_rng(external, _get_rng())
    assert policy.propose(dict(sensors={}, instruction='go')) == 0
    assert_same_rng(external, _get_rng())
    policy.feedback(1)
    assert policy.agent.last_action_flat.item() == 1
    assert policy.history == [1]
    with pytest.raises(ValueError):
        policy.feedback(0)


def test_learned_snapshot_detects_decoder_mutation_and_forward_without_feedback():
    policy, cache = fixture_policy()
    before = policy.snapshot()
    cache.cache_k[0, 0, 0] = 1
    assert policy.snapshot()['decoder'] != before['decoder']
    policy.propose(dict(sensors={}, instruction='go'))
    with pytest.raises(RuntimeError, match='feedback'):
        policy.propose(dict(sensors={}, instruction='go'))


def test_seed_fork_preserves_history_and_replays_policy_rng():
    first, _ = fixture_policy()
    second, _ = fixture_policy()
    for policy in (first, second):
        policy.seed_suffix(33)
        policy.propose(dict(sensors={}, instruction='go'))
        policy.feedback(1)
    assert first.snapshot() == second.snapshot()
    first.seed_suffix(44)
    assert first.snapshot()['history'] == second.snapshot()['history']
    assert first.snapshot()['rng'] != second.snapshot()['rng']
