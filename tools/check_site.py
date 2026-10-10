"""Check generated website links, fragments, asset hashes, and explorer evidence."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids = [], set()
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if 'id' in values:
            if values['id'] in self.ids:
                raise ValueError(f'Duplicate HTML id: {values["id"]}')
            self.ids.add(values['id'])
        for key in ('href', 'src', 'poster'):
            if values.get(key):
                self.links.append(values[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('dist/site'))
    args = parser.parse_args()
    root = args.directory.resolve()
    documents = {p.resolve(): Document(p.read_text(encoding='utf-8')) for p in root.rglob('*.html')}
    checked = 0
    for path, document in documents.items():
        for link in document.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent/unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= 'index.html'
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f'Broken local link from {path.name}: {link}')
            if url.fragment and target.suffix == '.html' and unquote(url.fragment) not in documents[target].ids:
                raise ValueError(f'Unknown fragment from {path.name}: {link}')
            checked += 1
    manifest = json.loads((root/'build-manifest.json').read_text())['files']
    for name, digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Website asset differs from build manifest: {name}')
    for scene in json.loads((root/'assets/explorer.json').read_text()).values():
        assert len(scene['prefixCosts']) == scene['root']
        assert all(len(b['poses']) == scene['root']+len(b['costs'])+1 for b in scene['branches'])
        assert all(b['poses'][:scene['root']+1] == scene['branches'][0]['poses'][:scene['root']+1]
                   for b in scene['branches'])
    print(json.dumps(dict(html_pages=len(documents), local_links_checked=checked,
                          asset_hashes_checked=len(manifest), explorer_scenes_checked=2)))


if __name__ == '__main__':
    main()
