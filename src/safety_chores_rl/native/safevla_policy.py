"""Optional adapter for the external SafeVLA inference agent and local weights.

No checkpoint or external model implementation is bundled. The caller pins the
checkpoint and external Python sources; redistribution remains a separate gate.
"""
from contextlib import contextmanager
import copy
import hashlib
import random

from .config import file_hash
from ..types import integer


def _tensor(value):
    import torch
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def _memory(value):
    import torch
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if torch.is_tensor(value):
        return _tensor(value)
    if isinstance(value, dict):
        return {str(k): _memory(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_memory(v) for v in value]
    raise TypeError(f'Unsupported policy memory: {type(value).__name__}')


def _get_rng():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _set_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


class SafeVLAPolicy:
    """Fresh upstream agent per episode, with explicit executed-action feedback."""
    def __init__(self, *, checkpoint, checkpoint_sha256, device='cuda:0',
                 greedy=False, expected_missing_keys=(), expected_unexpected_keys=()):
        if file_hash(checkpoint) != checkpoint_sha256:
            raise ValueError('Policy checkpoint SHA256 mismatch')
        self.checkpoint, self.checkpoint_sha256 = checkpoint, checkpoint_sha256
        self.device, self.greedy = device, greedy
        self.expected_missing = sorted(expected_missing_keys)
        self.expected_unexpected = sorted(expected_unexpected_keys)
        self.agent = None

    def reset(self, action_names, seed):
        import torch
        from architecture.models.allenact_transformer_models.inference_agent import InferenceAgentVIDA
        from training.online.dinov2_vits_tsfm_base import DinoV2ViTSTSFMBase, DinoV2ViTSTSFMBaseParams
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        params = DinoV2ViTSTSFMBaseParams()
        params.num_train_processes = 0
        params.use_bbox = False
        params.full_sensor = True
        params.use_text_goal = True
        params.use_data_augmentation = False
        self.agent = InferenceAgentVIDA.from_experiment_config(
            exp_config=DinoV2ViTSTSFMBase(params=params), mode='test', device=torch.device(self.device))
        self.agent.img_encoder_rgb_mean = (0.48145466, 0.4578275, 0.40821073)
        self.agent.img_encoder_rgb_std = (0.26862954, 0.26130258, 0.27577711)
        self.agent.greedy_sampling = self.greedy
        self.agent.test_augmentation = False
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        weights = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        mapped = {}
        for key, value in weights.items():
            key = key.removeprefix('model.')
            key = {'actor.weight': 'actor.linear.weight', 'actor.bias': 'actor.linear.bias'}.get(key, key)
            mapped[key] = value
        status = self.agent.actor_critic.load_state_dict(mapped, strict=False)
        self.load_status = dict(missing=sorted(status.missing_keys), unexpected=sorted(status.unexpected_keys))
        if self.load_status != dict(missing=self.expected_missing, unexpected=self.expected_unexpected):
            raise RuntimeError(f'Checkpoint/model compatibility differs from declaration: {self.load_status}')
        self.agent.actor_critic.eval()
        self.agent.steps_before_rollout_refresh = 10000
        self.agent.reset()
        self.actions = list(action_names)
        if self.agent.get_action_list() != self.actions:
            raise ValueError('Policy vocabulary differs from native task vocabulary')
        self.history, self.pending, self.probabilities = [], False, None
        self.rng = _get_rng()
        self.seed_suffix(seed)

    @contextmanager
    def _policy_rng(self):
        external = _get_rng()
        _set_rng(self.rng)
        try:
            yield
        finally:
            self.rng = _get_rng()
            _set_rng(external)

    def propose(self, observation):
        import torch
        if self.pending:
            raise RuntimeError('Every policy forward requires actual-action feedback')
        with self._policy_rng(), torch.inference_mode():
            name, probabilities = self.agent.get_action(
                copy.deepcopy(observation['sensors']), observation['instruction'])
        self.probabilities = probabilities.detach().cpu().tolist()
        self.pending = True
        return self.actions.index(name)

    def feedback(self, action):
        import torch
        integer(action, 'executed action')
        if not self.pending or action >= len(self.actions):
            raise ValueError('Invalid action feedback or missing policy forward')
        # The next upstream forward reads last_action_flat into rollout storage.
        # Store what the simulator executed, including search interventions.
        self.agent.last_action_flat = torch.full_like(self.agent.last_action_flat, action)
        self.history.append(action)
        self.pending = False

    def seed_suffix(self, seed):
        import numpy as np
        import torch
        integer(seed, 'policy seed')
        external = _get_rng()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.rng = _get_rng()
        finally:
            _set_rng(external)

    def candidate_preview(self, observation):
        self.propose(observation)
        order = sorted(range(len(self.actions)), key=lambda i: (-self.probabilities[i], i))
        return dict(actions=order, action_names=self.actions, probabilities=self.probabilities)

    def snapshot(self):
        state = {}
        for name, module in self.agent.actor_critic.named_modules():
            row = {}
            if hasattr(module, 'time_step_counter'):
                row['time_step_counter'] = int(module.time_step_counter)
            for key in ('cache_k', 'cache_v'):
                if hasattr(module, key):
                    row[key] = _tensor(getattr(module, key))
            if row:
                state[name] = row
        numpy_rng = self.rng[1]
        return dict(checkpoint_sha256=self.checkpoint_sha256, load_status=self.load_status,
                    history=list(self.history), pending=self.pending,
                    steps=int(self.agent.steps_taken_in_task),
                    trajectory=int(self.agent.num_evaluated_traj), memory=_memory(self.agent.memory),
                    last_action=_memory(getattr(self.agent, 'last_action_flat', None)), decoder=state,
                    rng=dict(python=self.rng[0], numpy=[numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                             torch=_tensor(self.rng[2]), cuda=[_tensor(t) for t in self.rng[3]]))

    def close(self):
        self.agent = None
