'use strict';

const svgNS = 'http://www.w3.org/2000/svg';
const colors = ['#087c65', '#b15c2a', '#7560a5'];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied';
      const status = document.getElementById('copy-status');
      if (status) status.textContent = 'Commands copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy'; }, 2000);
    } catch {
      button.textContent = 'Select text to copy';
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  });
});

document.getElementById('watch-demo')?.addEventListener('click', () => {
  const video = document.getElementById('main-video');
  video.play().catch(() => { video.focus(); });
});
document.querySelectorAll('video').forEach(video => {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
  });
});

function svgElement(name, attributes = {}, text) {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

async function startExplorer() {
  const map = document.getElementById('trajectory-map');
  if (!map) return;
  const response = await fetch(new URL('assets/explorer.json', document.baseURI));
  if (!response.ok) throw new Error('Trajectory data could not be loaded.');
  const scenes = await response.json();
  const slider = document.getElementById('trajectory-step');
  const playButton = document.getElementById('play-trajectory');
  const resetButton = document.getElementById('reset-trajectory');
  const tabs = [...document.querySelectorAll('[data-scene]')];
  const state = { scene: scenes.prefix40, step: 0, timer: null, phase: '' };
  let currentPaths, markers, prefixPath, project, readouts;

  function color(index) {
    const branch = state.scene.branches[index];
    return branch.success ? (branch.costs.some(cost => cost > 0) ? colors[1] : colors[0]) : colors[2];
  }

  function pause() {
    clearInterval(state.timer);
    state.timer = null;
    playButton.replaceChildren(Object.assign(document.createElement('span'), { textContent: '▶' }), document.createTextNode(' Play'));
    playButton.setAttribute('aria-label', 'Play trajectories');
  }

  function makeMap() {
    const scene = state.scene;
    const positions = scene.branches.flatMap(branch => branch.poses);
    const xmin = Math.min(...positions.map(p => p[0])) - 1.3;
    const xmax = Math.max(...positions.map(p => p[0])) + 1.3;
    const zmin = Math.min(...positions.map(p => p[1])) - 1.3;
    const zmax = Math.max(...positions.map(p => p[1])) + 1.3;
    const scale = Math.min(650 / (xmax - xmin), 355 / (zmax - zmin));
    project = p => [380 + (p[0] - (xmin + xmax) / 2) * scale, 212 - (p[1] - (zmin + zmax) / 2) * scale];
    map.replaceChildren(svgElement('title', { id: 'map-title' }, 'Recorded robot base trajectories'), svgElement('desc', { id: 'map-description' }, 'Cyan shows the shared replay prefix. Branch colors show the sampled continuations. Move the timeline to inspect an action step.'));
    const defs = svgElement('defs');
    const clip = svgElement('clipPath', { id: 'map-clip' });
    clip.append(svgElement('rect', { x: 40, y: 15, width: 685, height: 380, rx: 6 }));
    defs.append(clip); map.append(defs);
    const background = svgElement('g', { 'clip-path': 'url(#map-clip)' });
    scene.rooms.forEach(room => background.append(svgElement('polygon', { points: room.map(p => project(p).join(',')).join(' '), fill: '#e8eee2', stroke: '#bfcbb9', 'stroke-width': 1.2 })));
    const spacing = Math.max(xmax - xmin, zmax - zmin) > 15 ? 4 : 2;
    for (let x = Math.ceil(xmin / spacing) * spacing; x <= xmax; x += spacing) {
      const px = project([x, 0])[0];
      background.append(svgElement('line', { x1: px, x2: px, y1: 15, y2: 395, stroke: '#ccd6c6', 'stroke-dasharray': '2 5', 'stroke-width': .7 }));
      if (px >= 42 && px <= 720) map.append(svgElement('text', { x: px, y: 413, fill: '#6c7a6c', 'font-size': 11, 'text-anchor': 'middle' }, x.toFixed(0)));
    }
    for (let z = Math.ceil(zmin / spacing) * spacing; z <= zmax; z += spacing) {
      const py = project([0, z])[1];
      background.append(svgElement('line', { x1: 40, x2: 725, y1: py, y2: py, stroke: '#ccd6c6', 'stroke-dasharray': '2 5', 'stroke-width': .7 }));
      if (py >= 15 && py <= 395) map.append(svgElement('text', { x: 28, y: py + 4, fill: '#6c7a6c', 'font-size': 11, 'text-anchor': 'end' }, z.toFixed(0)));
    }
    map.append(background);
    map.append(svgElement('text', { x: 720, y: 435, fill: '#6c7a6c', 'font-size': 11, 'text-anchor': 'end' }, 'World X / Z · metres'));
    const line = poses => poses.map((p, i) => `${i ? 'L' : 'M'}${project(p).map(v => v.toFixed(2)).join(',')}`).join(' ');
    state.path = line;
    const prefix = scene.branches[0].poses.slice(0, scene.root + 1);
    map.append(svgElement('path', { d: line(prefix), fill: 'none', stroke: '#376e91', opacity: .25, 'stroke-width': 3, 'stroke-dasharray': '4 5' }));
    scene.branches.forEach((branch, i) => map.append(svgElement('path', { d: line(branch.poses.slice(scene.root)), fill: 'none', stroke: color(i), opacity: .19, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' })));
    prefixPath = svgElement('path', { fill: 'none', stroke: '#376e91', 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
    map.append(prefixPath);
    currentPaths = scene.branches.map((_, i) => {
      const path = svgElement('path', { fill: 'none', stroke: color(i), 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
      map.append(path); return path;
    });
    const start = project(prefix[0]);
    map.append(svgElement('circle', { cx: start[0], cy: start[1], r: 4, fill: '#556858', stroke: '#fcfcf8', 'stroke-width': 2 }));
    map.append(svgElement('text', { x: start[0] + 10, y: start[1] + 17, fill: '#556858', 'font-size': 12 }, 'Reset'));
    if (scene.root) {
      const root = project(prefix[scene.root]);
      map.append(svgElement('circle', { cx: root[0], cy: root[1], r: 6, fill: '#fcfcf8', stroke: '#376e91', 'stroke-width': 2 }));
      map.append(svgElement('text', { x: root[0] + 12, y: root[1] - 12, fill: '#376e91', 'font-size': 12 }, `Search root · ${scene.root} actions`));
    }
    markers = scene.branches.map((_, i) => {
      const group = svgElement('g');
      group.append(svgElement('circle', { r: 7, fill: color(i), stroke: '#fcfcf8', 'stroke-width': 2 }));
      group.append(svgElement('path', { d: 'M0,-9 L-4,-2 L4,-2 Z', fill: color(i), transform: 'translate(0,-10)' }));
      map.append(group); return group;
    });
    const container = document.getElementById('branch-readouts');
    container.replaceChildren();
    container.style.setProperty('--branch-count', scene.branches.length);
    readouts = scene.branches.map((branch, i) => {
      const article = document.createElement('div'); article.className = 'branch-readout';
      const heading = document.createElement('div'); heading.className = 'branch-readout-heading';
      const title = document.createElement('strong'); title.textContent = `Branch ${String.fromCharCode(65 + i)}`; title.style.color = color(i);
      const seed = document.createElement('span'); seed.textContent = `suffix seed ${branch.seed}`;
      heading.append(title, seed);
      const status = document.createElement('div'); status.className = 'branch-status';
      const stats = document.createElement('div'); stats.className = 'branch-statline';
      const cost = document.createElement('span'); const actions = document.createElement('span');
      stats.append(cost, actions); article.append(heading, status, stats); container.append(article);
      return { status, cost, actions };
    });
  }

  function update() {
    const scene = state.scene;
    slider.value = state.step;
    document.getElementById('step-output').textContent = `${state.step} / ${scene.limit}`;
    slider.setAttribute('aria-valuetext', `Episode action ${state.step} of ${scene.limit}`);
    prefixPath.setAttribute('d', state.path(scene.branches[0].poses.slice(0, Math.min(state.step, scene.root) + 1)));
    scene.branches.forEach((branch, i) => {
      const index = Math.min(state.step, branch.poses.length - 1);
      const suffix = Math.max(0, index - scene.root);
      const pose = branch.poses[index];
      const point = project(pose);
      markers[i].setAttribute('transform', `translate(${point.join(',')}) rotate(${pose[2]})`);
      markers[i].style.display = state.step <= scene.root && i > 0 ? 'none' : '';
      [...markers[i].children].forEach(node => node.setAttribute('fill', state.step <= scene.root ? '#376e91' : color(i)));
      currentPaths[i].setAttribute('d', state.path(branch.poses.slice(scene.root, index + 1)));
      const cost = scene.prefixCosts.slice(0, Math.min(index, scene.root)).reduce((a, b) => a + b, 0) + branch.costs.slice(0, suffix).reduce((a, b) => a + b, 0);
      const finished = index === branch.poses.length - 1;
      readouts[i].status.textContent = finished ? (branch.success ? (cost === 0 ? 'Zero-cost success' : 'Success with cost') : 'No success at limit') : (state.step === 0 ? 'Initial state' : state.step < scene.root ? 'Replaying prefix' : state.step === scene.root ? 'Root reconstructed' : 'Continuing');
      readouts[i].status.style.color = finished ? color(i) : '#192a27';
      readouts[i].cost.textContent = `Cost ${cost}`;
      readouts[i].actions.textContent = `${suffix} / ${branch.costs.length} branch actions`;
    });
    let phase, description;
    if (state.step === 0) {
      phase = '01 / RESET'; description = scene.root ? `Fresh seeded episodes. Each worker must replay ${scene.root} saved live actions before branching.` : 'The prefix is empty: the reset state is already the search root. The branches use the same first candidate action.';
    } else if (state.step < scene.root) {
      phase = '02 / REPLAY PREFIX'; description = `Replaying action ${state.step} of ${scene.root}. Every branch follows the same live prefix; no continuation has started.`;
    } else if (state.step === scene.root) {
      phase = '02 / ROOT RECONSTRUCTED'; description = 'The full captured root states match across these reconstructed workers. Continuations can now diverge.';
    } else {
      phase = '03 / CONTINUE BRANCHES'; description = 'The candidate is the same; the policy continuation seeds differ. Completed branches stay at their final state.';
    }
    document.getElementById('phase-label').textContent = phase;
    // Announce phase changes without flooding screen readers on every playback tick.
    if (state.phase !== phase || !state.timer) document.getElementById('phase-description').textContent = description;
    state.phase = phase;
  }

  function setScene(key) {
    pause(); state.scene = scenes[key]; state.step = 0; state.phase = '';
    slider.max = state.scene.limit;
    document.getElementById('episode-label').textContent = `Task ${state.scene.task} · seed ${state.scene.seed}`;
    document.getElementById('explorer-panel').setAttribute('aria-labelledby', `tab-${key}`);
    tabs.forEach(tab => { const active = tab.dataset.scene === key; tab.setAttribute('aria-selected', active); tab.tabIndex = active ? 0 : -1; });
    makeMap(); update();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setScene(tab.dataset.scene));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus(); setScene(tabs[next].dataset.scene);
    });
  });
  slider.addEventListener('input', () => { pause(); state.step = Number(slider.value); update(); });
  resetButton.addEventListener('click', () => { pause(); state.step = 0; update(); });
  playButton.addEventListener('click', () => {
    if (state.timer) { pause(); return; }
    if (state.step >= state.scene.limit) state.step = 0;
    playButton.replaceChildren(Object.assign(document.createElement('span'), { textContent: 'Ⅱ' }), document.createTextNode(' Pause'));
    playButton.setAttribute('aria-label', 'Pause trajectories');
    state.timer = setInterval(() => { state.step = Math.min(state.scene.limit, state.step + 1); update(); if (state.step >= state.scene.limit) pause(); }, 125);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  [slider, playButton, resetButton].forEach(element => { element.disabled = false; });
  setScene('prefix40');
  // Playback is always user-initiated, including when reduced motion is requested.
  if (reducedMotion) map.setAttribute('data-reduced-motion', 'true');
}

startExplorer().catch(error => {
  const description = document.getElementById('phase-description');
  if (description) description.textContent = 'The interactive data could not load. The recorded videos and downloadable receipts below are still available.';
  console.error(error);
});
