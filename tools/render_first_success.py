"""Render a completed native_first_success report using its original camera frames.

Requires Pillow. Does not run the simulator or generate replacement observations.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size, bold=False):
    names = (["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
             if bold else ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, help='Optional verified replay coordinates; requires matplotlib')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'complete' or report['success'] is not True:
        parser.error('This renderer requires a completed successful report')
    records = report['execution']
    if not records or not all(step['matched'] for step in records):
        parser.error('Every recorded execution step must match')
    trajectories = None
    if args.trajectories:
        from plot_trajectories import load_trajectories, successful_branch, map_frame
        trajectories = load_trajectories(args.trajectories)
        if trajectories['original_report_sha256'] != hashlib.sha256(args.report.read_bytes()).hexdigest():
            parser.error('Trajectory coordinates belong to a different report')
        winner = successful_branch(trajectories)
        if winner['actions'] != report['plan']['witness']['actions']:
            parser.error('Trajectory does not match the executed witness')
    args.out.mkdir(parents=True, exist_ok=True)
    usage = report['plan']['usage']
    total = len(records)
    colors = dict(bg='#0b1325', panel='#142238', edge='#283c56', text='#edf4ff',
                  muted='#a0b3ca', cyan='#75ceff', green='#70e1b8')
    small, body, title, value = font(13), font(17), font(29, True), font(27, True)
    images = []
    for index in range(total+1):
        image = Image.new('RGB', (1280 if trajectories else 880, 556), colors['bg'])
        draw = ImageDraw.Draw(image)
        draw.text((32, 21), 'SAFETY-CHORES-RL   /   NATIVE EXECUTION', font=small, fill=colors['cyan'])
        instruction = report['instruction'].capitalize()
        while draw.textbbox((0, 0), instruction, font=title)[2] > 816:
            instruction = instruction[:-2].rstrip() + '…'
        draw.text((30, 45), instruction, font=title, fill=colors['text'])
        cost = sum(step['cost'] for step in records[:index])
        cards = [(32, f"{usage['rollouts']} / {usage['exhaustive_rollouts']}", 'ROLLOUTS USED'),
                 (312, f'{index} / {total}', 'ACTIONS CHECKED'),
                 (592, f'{cost:g}', 'RECORDED COST')]
        for x, number, label in cards:
            draw.rounded_rectangle((x, 101, x+256, 178), radius=12, fill=colors['panel'], outline=colors['edge'])
            draw.text((x+17, 107), number, font=value, fill=colors['green'] if x==592 else colors['text'])
            draw.text((x+18, 149), label, font=small, fill=colors['muted'])
        for name, x, label in [('nav', 32, '01  NAVIGATION CAMERA'), ('manip', 464, '02  MANIPULATION CAMERA')]:
            draw.text((x, 200), label, font=small, fill=colors['muted'])
            with Image.open(args.frames/f'{name}-{index:04d}.png') as camera:
                if camera.size != (384, 224):
                    raise ValueError(f'Expected 384x224 original camera, got {camera.size}')
                if trajectories and hashlib.sha256(camera.convert('RGB').tobytes()).hexdigest() != winner['poses'][index]['image_hashes'][name]:
                    raise ValueError(f'Replay camera differs from original live frame {name}/{index}')
                image.paste(camera.convert('RGB'), (x, 226))
            draw.rectangle((x-1, 225, x+384, 450), outline=colors['edge'])
        draw.rounded_rectangle((32, 470, 848, 477), radius=3, fill=colors['panel'])
        if index:
            draw.rounded_rectangle((32, 470, 32+int(816*index/total), 477), radius=3, fill=colors['green'])
        status = 'Root captured' if not index else ('Task success confirmed' if index==total else 'Execution matches simulated path')
        draw.text((32, 490), status, font=body, fill=colors['green'])
        draw.text((32, 526), 'Selected example  ·  Original sensor frames  ·  Accelerated playback: 8 actions/s', font=small, fill=colors['muted'])
        if trajectories:
            draw.line((880, 24, 880, 530), fill=colors['edge'])
            draw.text((904, 21), '03 / TOP-DOWN TRAJECTORY', font=small, fill=colors['cyan'])
            draw.text((902, 45), 'World paths', font=title, fill=colors['text'])
            for y, color, label in [(95, '#ffad7b', 'Failed rollout'), (120, colors['green'], 'Successful rollout'),
                                     (145, colors['cyan'], 'Execution progress (replayed)')]:
                draw.line((906, y+8, 930, y+8), fill=color, width=3)
                draw.text((940, y), label, font=small, fill=colors['muted'])
            image.paste(map_frame(trajectories, index), (892, 170))
            draw.text((904, 526), 'Base position + heading · room outlines', font=small, fill=colors['muted'])
        images.append(image)
    images[-1].save(args.out/'first-success.png')
    # Use one palette across frames to keep labels and dark backgrounds stable.
    thumbnail_width = images[0].width//4
    strip = Image.new('RGB', (thumbnail_width*len(images), 139))
    for i, frame in enumerate(images):
        strip.paste(frame.resize((thumbnail_width, 139)), (thumbnail_width*i, 0))
    palette = strip.quantize(colors=192)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in images]
    durations = [120 if i % 2 == 0 else 130 for i in range(len(frames))]
    durations[-1] = 2000
    frames[0].save(args.out/'first-success.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    print(json.dumps(dict(frames=len(frames), seconds=sum(durations)/1000,
                          gif_bytes=(args.out/'first-success.gif').stat().st_size)))


if __name__ == '__main__':
    main()
