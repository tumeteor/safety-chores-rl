"""Build the static project website using only curated public repository files.

Install site/requirements.txt first. Output defaults to dist/site and works at
both a domain root and a GitHub Pages project subpath.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import markdown

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'docs/assets'


def explorer_scene(filename):
    receipt = json.loads((ASSETS/filename).read_text())
    if receipt['status'] != 'complete' or not all(b['verified'] for b in receipt['branches']):
        raise ValueError('Only verified completed branch captures may be published')
    return dict(task=receipt['task_id'], seed=receipt['seed'], root=receipt['query_step'],
        limit=receipt['episode_limit'], prefixCosts=receipt['prefix_costs'],
        rooms=[[[p['x'], p['z']] for p in room['floor_polygon']] for room in receipt['scene']['rooms']],
        branches=[dict(seed=b['sample_seed'], success=b['original_success'], costs=b['costs'],
            poses=[[p['position']['x'], p['position']['z'], p['yaw']] for p in b['poses']])
            for b in receipt['branches']])


def guide_document(name, text):
    content = markdown.markdown(text, extensions=['fenced_code', 'tables', 'toc'])
    content = re.sub(r'href="([^"#]+)\.md([^"]*)"', r'href="\1.html\2"', content)
    content = content.replace('href="../README.html', 'href="../index.html')
    content = content.replace('#a-practical-native-example', '#demo')
    content = content.replace('href="../THIRD_PARTY_NOTICES.html', 'href="third-party.html')
    content = content.replace('href="assets/', 'href="../assets/').replace('src="assets/', 'src="../assets/')
    title = re.search(r'^# (.+)', text, flags=re.M).group(1)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Safety-CHORES-RL</title><link rel="icon" href="../favicon.svg"><link rel="stylesheet" href="../style.css"></head>
<body><a class="skip-link" href="#main">Skip to content</a><header class="site-header"><a class="brand" href="../index.html"><img src="../favicon.svg" width="30" height="30" alt="">Safety-CHORES / RL</a><nav aria-label="Documentation"><a href="native.html">Native setup</a><a href="api.html">API</a><a href="release.html">Release status</a></nav></header>
<main id="main" class="wrap guide"><a class="guide-back" href="../index.html">← Project website</a>{content}</main>
<footer class="wrap site-footer"><span>Safety-CHORES-RL · Development preview</span><a href="https://github.com/tumeteor/safety-chores-rl">Source on GitHub ↗</a></footer></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'dist/site')
    parser.add_argument('--site-url', default='', help='Optional canonical HTTPS URL, including the project subpath')
    args = parser.parse_args()
    output = args.out.resolve()
    if output == ROOT or ROOT in output.parents and not output.is_relative_to(ROOT/'dist'):
        parser.error('Use dist/ or a separate output directory; do not overwrite source directories')
    if args.site_url and not args.site_url.startswith('https://'):
        parser.error('The public canonical URL must use HTTPS')
    output.mkdir(parents=True, exist_ok=True)
    expected = set()

    def write(name, content):
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8', newline='\n')
        expected.add(name)

    def copy(source, name):
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected.add(name)

    for name in ('index.html', 'style.css', 'app.js', 'favicon.svg'):
        text = (ROOT/'site'/name).read_text(encoding='utf-8')
        if name == 'index.html':
            base = html.escape(args.site_url.rstrip('/')+'/', quote=True)
            canonical = (f'<link rel="canonical" href="{base}">\n'
                         f'  <meta property="og:url" content="{base}">\n'
                         f'  <meta property="og:image" content="{base}assets/first-success.png">') if args.site_url else ''
            text = text.replace('<!-- canonical -->', canonical)
        write(name, text)
    for source in sorted(ASSETS.iterdir()):
        if source.is_file() and source.suffix in {'.json', '.png', '.svg', '.gif', '.mp4'}:
            copy(source, 'assets/'+source.name)
    data = dict(prefix40=explorer_scene('branches-prefix40.json'), seed892=explorer_scene('branches-seed892.json'))
    write('assets/explorer.json', json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
    for name in ('native', 'api', 'release'):
        write(f'guides/{name}.html', guide_document(name, (ROOT/f'docs/{name}.md').read_text(encoding='utf-8')))
    write('guides/third-party.html', guide_document('third-party', (ROOT/'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')))
    for source in sorted((ROOT/'examples').glob('*.py')):
        copy(source, 'examples/'+source.name)
    write('.nojekyll', '')
    # Refuse unknown stale files rather than allowing them into a deployment.
    extras = [p.relative_to(output).as_posix() for p in output.rglob('*')
              if p.is_file() and p.relative_to(output).as_posix() not in expected | {'build-manifest.json'}]
    if extras:
        raise ValueError(f'Unexpected files in website output: {extras}; use a fresh output directory')
    manifest = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in sorted(expected)}
    write('build-manifest.json', json.dumps(dict(files=manifest), indent=2)+'\n')
    print(json.dumps(dict(output=str(output), files=len(expected), bytes=sum((output/name).stat().st_size for name in expected))))


if __name__ == '__main__':
    main()
