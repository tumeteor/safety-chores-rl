"""Render reset, prefix replay, and divergent native branches side by side.

Requires Pillow. Uses only hash-checked replay images and recorded world poses.
Branches are aligned by action count; capture was sequential, not simultaneous.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from render_first_success import font

BG, PANEL, EDGE = '#0b1325', '#142238', '#38516a'
TEXT, MUTED, CYAN = '#edf4ff', '#a0b3ca', '#75ceff'
GREEN, ORANGE, PURPLE = '#70e1b8', '#ffad7b', '#bda7ff'


def load_capture(path):
    data = json.loads(path.read_text())
    if data['status'] != 'complete' or data['kind'] != 'native_branch_replay':
        raise ValueError('A completed branch replay is required')
    root = data['query_step']
    if len(data['prefix_actions']) != root or not 2 <= len(data['branches']) <= 4:
        raise ValueError('Invalid prefix/branch count')
    reference = data['branches'][0]['state_hashes'][:root+1]
    for branch in data['branches']:
        if not all(branch.get(k) for k in ('verified', 'endpoint_matched', 'original_candidate_probabilities_matched')):
            raise ValueError('Branch evidence is incomplete')
        count = root+len(branch['actions'])+1
        if (len(branch['poses']) != count or len(branch['state_hashes']) != count
                or branch['transitions_matched'] != len(branch['actions'])
                or branch['state_hashes'][:root+1] != reference
                or branch['root_snapshot_sha256'] != reference[-1]
                or branch['original_endpoint_sha256'] != branch['replay_endpoint_sha256']):
            raise ValueError('Branch state/transition evidence differs')
        if [p['step'] for p in branch['poses']] != list(range(count)):
            raise ValueError('Noncontiguous pose sequence')
        if not all(math.isfinite(p['position'][a]) for p in branch['poses'] for a in ('x', 'z')):
            raise ValueError('Nonfinite trajectory coordinate')
    return data


def branch_color(branch):
    return (GREEN if branch['original_cost'] == 0 else ORANGE) if branch['original_success'] else PURPLE


def outcome(branch):
    if branch['original_success']:
        return 'SUCCESS / ZERO COST' if branch['original_cost'] == 0 else 'SUCCESS / POSITIVE COST'
    return 'LIMIT / NO SUCCESS' if branch['stop_reason'] == 'episode_limit' else 'TASK FAILED'


def map_panel(data, branch, index):
    image = Image.new('RGB', (184, 210), PANEL)
    draw = ImageDraw.Draw(image)
    positions = [p['position'] for b in data['branches'] for p in b['poses']]
    xmin, xmax = min(p['x'] for p in positions)-.7, max(p['x'] for p in positions)+.7
    zmin, zmax = min(p['z'] for p in positions)-.7, max(p['z'] for p in positions)+.7
    scale = min(164/(xmax-xmin), 174/(zmax-zmin))
    cx, cz = (xmin+xmax)/2, (zmin+zmax)/2

    def xy(position):
        return (92+(position['x']-cx)*scale, 92-(position['z']-cz)*scale)

    for room in data['scene']['rooms']:
        draw.polygon([xy(p) for p in room['floor_polygon']], fill='#1a2c43', outline=EDGE)
    root = data['query_step']
    prefix = [xy(p['position']) for p in branch['poses'][:min(index, root)+1]]
    if len(prefix) > 1:
        draw.line(prefix, fill=CYAN, width=2)
    suffix = [xy(p['position']) for p in branch['poses'][root:index+1]]
    if len(suffix) > 1:
        draw.line(suffix, fill=branch_color(branch), width=3)
    for step, color in [(0, MUTED), (root, CYAN)]:
        x, y = xy(branch['poses'][step]['position'])
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color, outline=BG)
    current = branch['poses'][index]
    x, y = xy(current['position'])
    yaw = math.radians(current['yaw'])
    draw.line((x, y, x+12*math.sin(yaw), y-12*math.cos(yaw)), fill=TEXT, width=2)
    draw.ellipse((x-3, y-3, x+3, y+3), fill=TEXT)
    draw.line((12, 190, 12+scale, 190), fill=MUTED, width=2)
    draw.text((12, 194), '1 m', font=font(10), fill=MUTED)
    draw.text((112, 194), 'X/Z metres', font=font(10), fill=MUTED)
    return image


def timeline(data, stride):
    root = data['query_step']
    result = [('reset', 0, 1400)]
    previous = 0
    for index in sorted(set(range(stride, root+1, stride)) | ({root} if root else set())):
        result.append(('replay', index, round((index-previous)*125)))
        previous = index
    result.append(('root', root, 1800))
    maximum = max(len(b['actions']) for b in data['branches'])
    milestones = {len(b['actions']) for b in data['branches']}
    milestones.update(i+1 for b in data['branches'] for i, c in enumerate(b['costs']) if c)
    previous = 0
    for index in sorted(set(range(stride, maximum+1, stride)) | milestones | {maximum}):
        result.append(('branch', root+index, round((index-previous)*125)))
        previous = index
    result[-1] = (*result[-1][:2], 2800)
    return result


def render(data, folder, phase, absolute_step):
    width, height = len(data['branches'])*416+32, 860
    image = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(image)
    small, body, title = font(12), font(17), font(28, True)
    root = data['query_step']
    draw.text((32, 18), 'SAFETY-CHORES-RL   /   RESET, REPLAY, BRANCH', font=small, fill=CYAN)
    draw.text((30, 39), 'Same reconstructed state. Different continuations.', font=title, fill=TEXT)
    draw.text((32, 78), f"Task {data['task_id']} · episode seed {data['seed']} · {data['instruction']}", font=small, fill=MUTED)
    active = 0 if phase == 'reset' else 1 if phase in ('replay', 'root') else 2
    card_width = (width-88)//3
    labels = ['01  RESET', '02  REPLAY PREFIX', '03  CONTINUE BRANCHES']
    descriptions = [f"Fresh episode · seed {data['seed']}",
                    f"{min(absolute_step, root)} / {root} saved live actions" if root else 'Empty prefix · already at the root',
                    'Force candidate, then sample policy']
    for number, (label, description) in enumerate(zip(labels, descriptions)):
        x = 32+number*(card_width+12)
        draw.rounded_rectangle((x, 112, x+card_width, 177), radius=10,
                               fill=PANEL, outline=CYAN if number == active else EDGE, width=2 if number == active else 1)
        draw.text((x+12, 123), label, font=font(13, True), fill=CYAN if number == active else MUTED)
        draw.text((x+12, 148), description, font=font(11), fill=TEXT if number == active else MUTED)
    messages = dict(reset='Each column starts from its own freshly reset simulator.',
                    replay='Rebuild the search state by replaying the same saved actions in every worker.',
                    root='Search root rebuilt: captured state fields match across these workers.',
                    branch='Continuation seeds differ. Completed branches hold their final frame.')
    draw.text((32, 189), messages[phase], font=font(13), fill=CYAN)
    for number, branch in enumerate(data['branches']):
        x = 32+number*416
        color = branch_color(branch)
        index = min(absolute_step, len(branch['poses'])-1)
        suffix_step = max(0, index-root)
        finished = phase == 'branch' and suffix_step == len(branch['actions'])
        same_candidate = all(b['candidate'] == branch['candidate'] for b in data['branches'])
        candidate = 'same first action' if same_candidate else f"candidate {branch['candidate']}"
        draw.text((x, 222), f"BRANCH {chr(65+number)}", font=font(18, True), fill=color)
        draw.text((x+127, 227), f"{candidate} · seed {branch['sample_seed']}", font=small, fill=MUTED)
        label = outcome(branch) if finished else ('ROOT MATCHED' if phase == 'root' else 'RESETTING' if phase == 'reset' else 'REPLAYING PREFIX' if phase == 'replay' else 'CONTINUING')
        draw.rounded_rectangle((x, 253, x+384, 280), radius=6, fill=PANEL)
        draw.text((x+9, 258), label, font=font(12, True), fill=color if finished else CYAN)
        draw.text((x, 291), 'NAVIGATION CAMERA', font=small, fill=MUTED)
        for name, target, size in [('nav', (x, 312), (384, 224)), ('manip', (x, 568), (184, 107))]:
            with Image.open(folder/branch['frames']/f'{name}-{index:04d}.png') as source:
                camera = source.convert('RGB')
                if camera.size != (384, 224) or hashlib.sha256(camera.tobytes()).hexdigest() != branch['poses'][index]['image_hashes'][name]:
                    raise ValueError(f'Camera hash/size mismatch for branch {number}, step {index}, {name}')
                if size != camera.size:
                    camera = camera.resize(size, Image.Resampling.LANCZOS)
                image.paste(camera, target)
        draw.text((x, 548), 'MANIPULATION CAMERA', font=font(11), fill=MUTED)
        draw.text((x+200, 548), 'BASE TRAJECTORY', font=font(11), fill=MUTED)
        image.paste(map_panel(data, branch, index), (x+200, 568))
        cost = sum(data['prefix_costs'][:min(index, root)])+sum(branch['costs'][:suffix_step])
        draw.text((x, 689), f'{cost:g}', font=font(28, True), fill=ORANGE if cost else GREEN)
        draw.text((x+44, 702), 'RECORDED COST', font=font(10), fill=MUTED)
        draw.text((x, 731), f"{suffix_step} / {len(branch['actions'])} branch actions", font=font(12), fill=TEXT)
        draw.text((x, 753), f"Episode step: {index} / {data['episode_limit']}", font=font(11), fill=MUTED)
        draw.rounded_rectangle((x, 789, x+384, 794), radius=2, fill=PANEL)
        if suffix_step:
            draw.rectangle((x, 789, x+int(384*suffix_step/len(branch['actions'])), 794), fill=color)
    draw.text((32, 809), 'Selected recorded branches · replay capture · 8 actions/s · columns aligned by action count', font=small, fill=MUTED)
    draw.text((32, 832), 'Branches captured sequentially. Room outlines omit obstacles; arrows show base heading.', font=small, fill=MUTED)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='branches.json from native_branch_replay.py')
    parser.add_argument('--out', type=Path, required=True, help='GIF destination; also writes a PNG poster')
    parser.add_argument('--stride', type=int, default=2, help='Keep every Nth action plus outcome/cost milestones')
    args = parser.parse_args()
    if not 1 <= args.stride <= 8:
        parser.error('Stride must be between 1 and 8')
    data = load_capture(args.capture)
    schedule = timeline(data, args.stride)
    images = [render(data, args.capture.parent, phase, step) for phase, step, _ in schedule]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[-1].save(args.out.with_suffix('.png'))
    # Shared palette, including reserved UI/path colors to avoid flickering labels.
    strip = Image.new('RGB', (images[0].width//4*len(images), images[0].height//4))
    for index, item in enumerate(images):
        strip.paste(item.resize((item.width//4, item.height//4)), (item.width//4*index, 0))
    palette = strip.quantize(colors=192)
    fixed = [component for color in (BG, PANEL, EDGE, TEXT, MUTED, CYAN, GREEN, ORANGE, PURPLE, '#1a2c43')
             for component in ImageColor.getrgb(color)]
    entries = palette.getpalette()[:192*3]+fixed
    palette.putpalette(entries+[0]*(768-len(entries)))
    frames = [item.quantize(palette=palette, dither=Image.Dither.NONE) for item in images]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], loop=0, optimize=True,
                   disposal=1, duration=[duration for _, _, duration in schedule])
    print(json.dumps(dict(gif=str(args.out), frames=len(frames), bytes=args.out.stat().st_size,
                          playback_seconds=sum(d for _, _, d in schedule)/1000)))


if __name__ == '__main__':
    main()
