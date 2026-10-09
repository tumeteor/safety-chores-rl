"""Plot verified native replay paths over room floor polygons (requires matplotlib)."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

BG, PANEL, EDGE = '#0b1325', '#142238', '#38516a'
TEXT, MUTED, SUCCESS, FAILURE, EXECUTION = '#edf4ff', '#a0b3ca', '#70e1b8', '#ffad7b', '#75ceff'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'svg.hashsalt': 'safety-chores-rl',
                     'text.color': TEXT, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED})


def load_trajectories(path):
    data = json.loads(Path(path).read_text())
    if data['status'] != 'complete' or not data['branches']:
        raise ValueError('A completed trajectory replay is required')
    for branch in data['branches']:
        if not branch['verified'] or len(branch['poses']) != len(branch['actions'])+1:
            raise ValueError('Every plotted path must be fully verified')
        if [p['step'] for p in branch['poses']] != list(range(len(branch['poses']))):
            raise ValueError('Pose steps are not contiguous')
        if not all(math.isfinite(p['position'][axis]) for p in branch['poses'] for axis in ('x', 'y', 'z')):
            raise ValueError('Non-finite world coordinate')
    return data


def successful_branch(data):
    branches = [b for b in data['branches'] if b['original_success'] and b['original_cost'] == 0]
    if len(branches) != 1:
        raise ValueError('This demonstration plot expects one successful path')
    return branches[0]


def points(branch):
    return ([p['position']['x'] for p in branch['poses']],
            [p['position']['z'] for p in branch['poses']])


def path_length(branch):
    x, z = points(branch)
    return sum(math.hypot(x[i]-x[i-1], z[i]-z[i-1]) for i in range(1, len(x)))


def draw_map(ax, data, *, step=None, labels=True):
    ax.set_facecolor(PANEL)
    all_poses = [p['position'] for b in data['branches'] for p in b['poses']]
    xmin, xmax = min(p['x'] for p in all_poses)-1.2, max(p['x'] for p in all_poses)+1.2
    zmin, zmax = min(p['z'] for p in all_poses)-1.2, max(p['z'] for p in all_poses)+1.2
    ax.set(xlim=(xmin, xmax), ylim=(zmin, zmax), aspect='equal')
    for room in data['scene']['rooms']:
        polygon = [(p['x'], p['z']) for p in room['floor_polygon']]
        ax.add_patch(Polygon(polygon, closed=True, facecolor='#1a2c43', edgecolor=EDGE, linewidth=1.5, zorder=1))
    ax.grid(color=EDGE, linewidth=.5, alpha=.35, zorder=2)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)
    ax.tick_params(labelsize=9 if labels else 7, length=2)
    ax.set_xlabel('World X (m)', fontsize=10 if labels else 8)
    ax.set_ylabel('World Z (m)', fontsize=10 if labels else 8)
    # Draw rejected paths first so the selected route stays legible at overlaps.
    for branch in sorted(data['branches'], key=lambda b: bool(b['original_success'])):
        x, z = points(branch)
        success = branch['original_success']
        color = SUCCESS if success else FAILURE
        ax.plot(x, z, color=color, linewidth=3 if labels else 2,
                linestyle='-' if success else '--', alpha=.85, zorder=4 if success else 3)
        ax.scatter([x[-1]], [z[-1]], color=color, marker='*' if success else 'X',
                   s=190 if labels else 90, edgecolors=BG, linewidths=.8, zorder=7)
        if labels:
            ax.annotate('Grasp completed' if success else 'Failed rollout ends', (x[-1], z[-1]),
                        xytext=(12, 12 if success else -22), textcoords='offset points',
                        fontsize=10, color=color, bbox=dict(facecolor=BG, edgecolor='none', alpha=.8, pad=3))
    winner = successful_branch(data)
    x, z = points(winner)
    ax.scatter([x[0]], [z[0]], s=72 if labels else 35, color=TEXT, edgecolors=BG, zorder=8)
    if labels:
        ax.annotate('Start', (x[0], z[0]), xytext=(10, -18), textcoords='offset points', fontsize=10, color=TEXT)
    if step is not None:
        if not 0 <= step < len(x):
            raise ValueError('Execution step is outside the successful path')
        ax.plot(x[:step+1], z[:step+1], color=EXECUTION, linewidth=3, zorder=9)
        ax.scatter([x[step]], [z[step]], s=45, color=TEXT, edgecolors=EXECUTION, zorder=11)
        yaw = math.radians(winner['poses'][step]['yaw'])
        ax.annotate('', xy=(x[step]+.48*math.sin(yaw), z[step]+.48*math.cos(yaw)),
                    xytext=(x[step], z[step]), arrowprops=dict(arrowstyle='-|>', color=TEXT, lw=1.4), zorder=12)
    return ax


def map_frame(data, step):
    from PIL import Image
    fig = plt.figure(figsize=(3.8, 3.52), dpi=100, facecolor=BG)
    ax = fig.add_axes((.16, .14, .80, .83))
    draw_map(ax, data, step=step, labels=False)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = load_trajectories(args.trajectories)
    winner = successful_branch(data)
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.8), dpi=150, facecolor=BG)
    fig.text(.035, .943, 'SAFETY-CHORES-RL  /  VERIFIED REPLAY TRAJECTORIES', color=EXECUTION, size=10)
    fig.text(.035, .888, 'Two rollouts. Two outcomes.', size=25, weight='bold')
    ax = fig.add_axes((.065, .16, .52, .65))
    draw_map(ax, data)
    fig.text(.625, .755, 'Same start. Same task.', size=18, weight='bold')
    for index, branch in enumerate(data['branches']):
        y = .65-index*.175
        color = SUCCESS if branch['original_success'] else FAILURE
        title = 'Successful continuation' if branch['original_success'] else 'Failed continuation'
        fig.text(.625, y, title, color=color, size=15, weight='bold')
        fig.text(.625, y-.043, f"{len(branch['actions'])} actions  ·  {path_length(branch):.1f} m base travel  ·  cost {branch['original_cost']:g}", color=MUTED, size=10)
        fig.text(.625, y-.081, f"Initial action {branch['candidate'][0]}  /  continuation seed {branch['sample_seed']}", color=MUTED, size=10)
    fig.text(.625, .295, f"Successful path executed: {len(winner['actions'])} / {len(winner['actions'])} checks matched", color=EXECUTION, size=11, weight='bold')
    fig.text(.625, .205, 'Coordinates recovered from the saved actions.\nBoth replays matched every original state hash.', color=MUTED, size=10, linespacing=1.7)
    fig.text(.035, .070, 'Robot base centre in world coordinates. Room outlines only; doors and furniture omitted.', color=MUTED, size=10)
    fig.text(.035, .035, 'Turning and arm motion may leave the base in place. Selected development example; task 0, seed 1909.', color=MUTED, size=10)
    for extension in ('svg', 'png'):
        path = args.out/f'first-success-trajectories.{extension}'
        fig.savefig(path, facecolor=BG, metadata={'Date': None} if extension == 'svg' else None)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    plt.close(fig)
    print(json.dumps(dict(paths=len(data['branches']), actions=sum(len(b['actions']) for b in data['branches']),
                          base_travel_metres=[round(path_length(b), 3) for b in data['branches']])))


if __name__ == '__main__':
    main()
