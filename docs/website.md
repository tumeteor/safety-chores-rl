# Project website

The project website is built for **https://tumeteor.github.io/safety-chores-rl/**.
It presents the native execution demo, two branch comparison videos, a
trajectory explorer with timeline controls, and the native/API/release documentation.

Source files live in `site/`. The small Python build copies curated public
assets, derives the explorer coordinates from verified replay receipts, and
renders Markdown documentation as HTML. The website needs no runtime server,
accounts, analytics, external fonts, or JavaScript dependencies.

## Build and preview

```sh
python -m pip install -r site/requirements.txt
python tools/build_site.py --site-url https://tumeteor.github.io/safety-chores-rl/
python tools/check_site.py
python -m http.server 8765 --bind 127.0.0.1 --directory dist/site
```

Open `http://127.0.0.1:8765/`. Build output is ignored by Git. The check verifies
local links and fragment targets, the generated asset manifest, and the shared
prefixes used by the interactive explorer. Unknown files in the output directory
cause the builder to refuse publication; use a fresh output directory if needed.

## Hosting

The GitHub repository is `tumeteor/safety-chores-rl`. Configure its Pages source
as **GitHub Actions**. `.github/workflows/pages.yml` builds, validates, and deploys
the site on relevant pushes to `main`, or through a manual workflow run. Relative
asset links support both the project subpath and a domain-root deployment.

The exact address `safety-chores-rl.github.io` would instead require a GitHub
user or organization named `safety-chores-rl`, with a repository named
`safety-chores-rl.github.io`. See [GitHub Pages site types](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## Media and evidence

The three MP4s are smaller browser-playable encodings of the existing GIFs.
They retain the images, overlays, and accelerated timeline. No new simulator
observations were generated for this website. Videos load on demand and expose
native playback controls; the trajectory animation starts only on user input.

The interactive explorer uses base X/Z coordinates, headings, prefix costs,
and branch costs from `docs/assets/branches-*.json`. It is a visualization of
recorded evidence, not a live simulator or a new search. Its outcome labels and
cost counters are computed from those receipts. The two comparison cases retain
their explicit selection and verification limits.

For each original GIF, the MP4 was encoded with:

```sh
ffmpeg -ignore_loop 1 -i INPUT.gif -vf fps=30 -c:v libx264 -crf 18 \
  -preset medium -threads 2 -pix_fmt yuv420p -movflags +faststart -an OUTPUT.mp4
```

Browser checks covered desktop, tablet, and mobile widths; all video sources;
timeline play/pause/reset; root and terminal outcomes; keyboard tabs; copyable
commands; and documentation pages. The site credits the authors of
[A Safe Action Is Not Enough](https://arxiv.org/abs/2610.05166), offers a BibTeX
download, and compares this tooling with LIBERO and Safety-CHORES using primary
sources. Experiment scope is documented in [validation and limitations](release.md).
