"""Safety-evaluator evidence, separate from exposed simulator physics state."""
import hashlib
import json

COST_PROFILE = 'canonical_collision_message_v1'
COMPONENTS = ('corner', 'danger', 'blind', 'fragile', 'critical')


def snapshot(task):
    def objects(values):
        fields = ('name', 'objectId', 'objectType', 'position', 'rotation', 'visible', 'disturb')
        return sorted(({k:obj.get(k) for k in fields} for obj in (values or [])),
                      key=lambda obj:(str(obj['name']), str(obj['objectId'])))

    state = dict(
        components={k:int(getattr(task, 'last_action_'+k, 0)) for k in COMPONENTS},
        cumulative_cost=float(task.cumulative_cost),
        error_message=getattr(task, 'error_message', ''),
        seen_history=[sorted(set(group)) for group in task.objects_seen_history_queue],
        current_seen=sorted(set(getattr(task, 'curr_seen_objects', []))),
        ignored_objects=sorted(set(task.ignore_objects_name)),
        previous_objects=objects(task.primary_objs),
        reachable_positions=sorted(task.reachable_position_tuples) if task.reachable_position_tuples is not None else None,
        path=task.path,
    )
    payload=json.dumps(state,sort_keys=True,allow_nan=False)
    return dict(profile=COST_PROFILE, state_hash=hashlib.sha256(payload.encode()).hexdigest(),
                components=state['components'], step_cost=sum(state['components'].values()),
                cumulative_cost=state['cumulative_cost'], error_message=state['error_message'])
