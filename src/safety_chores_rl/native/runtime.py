"""Native benchmark runtime using upstream tasks and an independent policy hook."""
from contextlib import ExitStack
import copy
import importlib
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace

from ..types import Transition, digest, integer
from .config import validate_config, file_hash
from .guards import checked_native_seeds, private_asset_staging
from .ordered_profile import OrderedProfile, configure_process


def _json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


class SafetyChoresRuntime:
    """Fresh Unity process and policy state for every replay episode.

    The default continuation is uniform random. Applications can supply a policy
    factory implementing native.policy.Policy; no trained policy is required.
    Each process must own its runtime and configure its external dependencies.
    """
    def __init__(self, *, config, vulkan_device=0, include_rgb=False,
                 policy_factory='safety_chores_rl.native.policy:UniformPolicy', policy_config=None,
                 diagnostics_dir=None):
        if sys.platform != 'linux':
            raise RuntimeError('Native benchmark player requires Linux')
        integer(vulkan_device, 'vulkan_device')
        self.cfg, self.tasks = validate_config(config)
        self.include_rgb = include_rgb
        self.policy_factory, self.policy_config = policy_factory, policy_config or {}
        self.policy = self.sampler = self.task = self._stack = None
        self._counter = 0
        self._temporary = tempfile.TemporaryDirectory(prefix='safety-chores-rl-')
        self.folder = Path(self._temporary.name)
        self._source_identity = digest({p.relative_to(Path(__file__).parents[1]).as_posix(): file_hash(p)
            for p in Path(__file__).parents[1].rglob('*.py')})
        try:
            self.evidence_dir = self.folder
            if diagnostics_dir is not None:
                self.evidence_dir = Path(diagnostics_dir).expanduser().resolve()/self.folder.name
                self.evidence_dir.mkdir(parents=True, exist_ok=False)
            player = Path(self.cfg['player']['path'])
            shutil.copytree(player.parent, self.folder/'player')
            self.player = self.folder/'player'/player.name
            self.manifest = self.folder/'player'/Path(self.cfg['build_manifest']['path']).name
            if not self.manifest.exists():
                shutil.copy2(self.cfg['build_manifest']['path'], self.manifest)
            repo = self.cfg['safevla']['repo_path']
            if repo not in sys.path:
                sys.path.insert(0, repo)
            os.environ['OBJAVERSE_ASSETS_DIR'] = self.cfg['asset_root']
            os.environ['OBJAVERSE_DATA_DIR'] = str(Path(self.cfg['asset_root']).parent)
            os.environ['OBJAVERSE_HOUSES_DIR'] = str(Path(self.cfg['houses']['path']).parent)
            os.environ['ORDERED_VULKAN_DEVICE'] = str(vulkan_device)
            from ai2thor.controller import Controller
            self._controller_class = Controller
            self._original_unity_command = Controller.unity_command
            configure_process(self.evidence_dir)
            from utils.data_utils import load_dataset_from_path
            self.houses = load_dataset_from_path(split_to_path={'val': self.cfg['houses']['path']})['val']
        except BaseException:
            self.close()
            raise

    def reset(self, task_id, seed, episode_limit):
        self.end_episode()
        integer(task_id, 'task_id')
        integer(seed, 'seed')
        integer(episode_limit, 'episode_limit', 1)
        if task_id >= len(self.tasks) or seed >= 2**31:
            raise ValueError('Task index or seed outside supported range')
        import numpy as np
        import torch
        from ai2thor.build import ExternalBuild
        from ai2thor.platform import CloudRendering
        from environment.stretch_controller import StretchController
        from tasks.multi_task_eval_sampler import MultiTaskSampler
        from tasks.task_specs import TaskSpecDatasetList
        from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS, _ACTION_HOOK_RUNNER
        from utils.task_datagen_utils import get_core_task_args
        from online_evaluation.online_evaluation_types_and_utils import (
            eval_sample_to_normalized_eval_sample, normalized_eval_sample_to_task_spec)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not hasattr(ExternalBuild, 'commit_id'):
            ExternalBuild.commit_id = None
        self._stack = ExitStack()
        receipts = []
        self._stack.enter_context(checked_native_seeds(self._controller_class, receipts))
        self._stack.enter_context(private_asset_staging(ExternalBuild, self.player, self.folder/'assets'))
        self._counter += 1
        out = self.evidence_dir/f'episode-{self._counter}'
        out.mkdir()
        self.actions, self.report = [], {}
        row = self.tasks[task_id]
        normalized = eval_sample_to_normalized_eval_sample(self.cfg['task_type'], copy.deepcopy(row), task_id)
        task_spec = normalized_eval_sample_to_task_spec(normalized)
        controller_args = dict(STRETCH_ENV_ARGS)
        controller_args.update(local_executable_path=str(self.player), checkpoint_build_manifest=str(self.manifest),
                               platform=CloudRendering, execution_noise_enabled=False)
        _ACTION_HOOK_RUNNER.last_asset_id_set = set()
        self.sampler = MultiTaskSampler(mode=self.cfg.get('split', 'val'),
            task_args=get_core_task_args(max_steps=episode_limit), houses=self.houses,
            house_inds=list(range(len(self.houses))), controller_args=controller_args,
            controller_type=StretchController, task_spec_sampler=TaskSpecDatasetList([task_spec]),
            visualize=False, prob_randomize_materials=0.0, device=0,
            always_allocate_a_new_stretch_controller_when_reset=False, retain_agent_pose=False, seed=seed)
        self.task = self.sampler.next_task()
        if self.task is None or seed not in receipts:
            raise RuntimeError('Task creation or acknowledged native seed missing')
        info = self.task.task_info
        if (str(info['house_index']) != str(row['house_index']) or info['task_type'] != row['task_type']
                or any(info.get(k) != row.get(k) for k in ('synsets', 'broad_synset_to_object_ids'))):
            raise RuntimeError('Actual task differs from the pinned benchmark row')
        self.action_names = list(self.task.action_names)
        self.sampler.controller.use_quick_navi_action = True
        self._last_obs_raw = self.task.get_observations()
        self.profile = OrderedProfile(self, self.sampler.controller, self.task, out, self.report)
        module, name = self.policy_factory.split(':', 1)
        self.policy = getattr(importlib.import_module(module), name)(**self.policy_config)
        self.policy.reset(self.action_names, seed)
        self._last_check = self.profile.check(self.current_obs())

    def current_obs(self):
        return SimpleNamespace(raw=self._last_obs_raw)

    def _policy_observation(self):
        # Only the public sensor subset reaches continuation policies.
        return dict(instruction=self.task.task_info['natural_language_spec'],
            step=self.task.num_steps_taken(), actions=list(self.action_names),
            sensors={k: copy.deepcopy(self._last_obs_raw[k]) for k in (
                'raw_navigation_camera', 'raw_manipulation_camera',
                'an_object_is_in_hand', 'relative_arm_location_metadata') if k in self._last_obs_raw})

    def advance(self, forced_action=None):
        if forced_action is not None:
            integer(forced_action, 'forced_action')
            if forced_action >= len(self.action_names):
                raise ValueError('Unknown action')
        proposal = self.policy.propose(self._policy_observation())
        integer(proposal, 'policy action')
        if proposal >= len(self.action_names):
            raise ValueError('Policy proposed unknown action')
        action = proposal if forced_action is None else forced_action
        before = float(self.task.cumulative_cost)
        result = self.task.step(action)
        self._last_obs_raw = result.observation
        self.policy.feedback(action)
        self.actions.append(action)
        self._last_check = self.profile.check(self.current_obs())
        cost = float(result.cost)
        if (not math.isclose(float(self.task.cumulative_cost)-before, cost, abs_tol=1e-8)
                or self._last_check['cost_state']['step_cost'] != cost):
            raise RuntimeError('Cost accounting mismatch')
        done, success = bool(result.done), bool(self.task._success)
        capped = done and not self.task.reached_terminal_state() and self.task.num_steps_taken() >= self.task.max_steps
        return Transition(action, float(result.reward), cost, terminated=done and not capped,
                          truncated=capped, success=success if done else None)

    def seed_suffix(self, seed):
        self.policy.seed_suffix(seed)

    def candidate_preview(self):
        if hasattr(self.policy, 'candidate_preview'):
            return self.policy.candidate_preview(self._policy_observation())
        return dict(actions=list(range(len(self.action_names))), action_names=list(self.action_names))

    def snapshot(self):
        return dict(prefix=list(self.actions), progress=dict(step=self.task.num_steps_taken(),
            limit=self.task.max_steps, cost=float(self.task.cumulative_cost), done=bool(self.task.is_done())),
            images=self._last_check['images'], physics=self._last_check['exposed_state_hash'],
            cost_state=self._last_check['cost_state']['state_hash'], policy=self.policy.snapshot(),
            source_identity=self._source_identity, profile=self._last_check['profile'])

    def observe(self):
        result = dict(instruction=self.task.task_info['natural_language_spec'],
            step=self.task.num_steps_taken(), actions=list(self.action_names),
            image_hashes=dict(self._last_check['images']),
            sensors={k: _json_value(self._last_obs_raw[k]) for k in (
                'an_object_is_in_hand', 'relative_arm_location_metadata') if k in self._last_obs_raw})
        if self.include_rgb:
            result['rgb'] = {k: _json_value(self._last_obs_raw[k]) for k in (
                'raw_navigation_camera', 'raw_manipulation_camera')}
        return result

    def execution_signature(self):
        # Full root validation still checks policy RNG. Along a forced action
        # path, only the independently forked proposal RNG is excluded; images,
        # exposed physics, cost bookkeeping, and deterministic memory must match.
        state = self.snapshot()
        state['policy'] = dict(state['policy'])
        state['policy'].pop('rng', None)
        return state

    def end_episode(self):
        try:
            if self.policy is not None:
                self.policy.close()
        finally:
            self.policy = None
            try:
                if self.sampler is not None:
                    # Upstream controller is lazy: reading it after failed startup
                    # would launch a second simulator during exception cleanup.
                    controller = vars(self.sampler).get('controller')
                    raw = getattr(controller, 'controller', None)
                    if raw is not None:
                        type(raw).stop(raw)
                    if controller is not None:
                        self.sampler.close()
            finally:
                self.sampler = self.task = None
                if self._stack is not None:
                    self._stack.close()
                    self._stack = None

    def close(self):
        try:
            self.end_episode()
        finally:
            if hasattr(self, '_original_unity_command'):
                self._controller_class.unity_command = self._original_unity_command
            self._temporary.cleanup()
