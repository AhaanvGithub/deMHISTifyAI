// Cut-off settings: two overlapping score curves (HP, SSL) from the model's out-of-fold training scores,
// a draggable cut-off line, presets, and a live readout. Lives behind the gear button in the panel header.
// Data: cutoff_data.json = { model, source, hp: [sorted P(SSL)], ssl: [sorted P(SSL)] } (training images only).
(function () {
  const STORE = 'cutoff.v1';
  const W = 352, H = 150, PAD = { l: 8, r: 8, t: 10, b: 22 };
  let D = null, cut = 0.5, preset = 'balanced', score = null, listeners = [];

  const frac = (arr, t, above) => {           // share of a sorted array above (or at/below) t
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m + 1; else hi = m; }
    return above ? (arr.length - lo) / arr.length : lo / arr.length;
  };
  const rates = t => ({ ssl: frac(D.ssl, t, true), hp: frac(D.hp, t, false) });
  const balanced = t => { const r = rates(t); return (r.ssl + r.hp) / 2; };

  // cut-off for each preset, all computed from the training scores
  function presetCut(id) {
    const grid = Array.from({ length: 999 }, (_, i) => (i + 1) / 1000);
    if (id === 'balanced') return D.default_cut ?? grid.reduce((b, t) => (balanced(t) > balanced(b) ? t : b), 0.5);
    const [side, target] = id.split('-');  // e.g. "ssl-95": highest cut-off that still catches >=95% of SSLs
    const ok = grid.filter(t => (side === 'ssl' ? rates(t).ssl : rates(t).hp) >= target / 100);
    return side === 'ssl' ? Math.max(...ok) : Math.min(...ok);
  }
  const PRESETS = [
    ['balanced', 'Balanced', 'Equal weight to missed SSLs and over-called HPs'],
    ['ssl-95', 'SSL ≥ 95%', 'Screening: catch at least 95% of SSLs'],
    ['ssl-99', 'SSL ≥ 99%', 'Catch nearly every SSL; many HPs flagged'],
    ['hp-95', 'HP ≥ 95%', 'Few false alarms: clear at least 95% of HPs'],
    ['hp-99', 'HP ≥ 99%', 'Almost no false alarms; misses more SSLs'],
  ];

  // smooth curve per class (Gaussian kernel on the 0-1 score), each scaled to its own peak
  function density(arr, bw = 0.065, n = 400) {
    const xs = Array.from({ length: n + 1 }, (_, i) => i / n);
    const ys = xs.map(x => arr.reduce((s, v) => s + Math.exp(-0.5 * ((x - v) / bw) ** 2), 0));
    const mx = Math.max(...ys);
    return xs.map((x, i) => [x, ys[i] / mx]);
  }
  const X = v => PAD.l + v * (W - PAD.l - PAD.r);
  const Y = v => H - PAD.b - v * (H - PAD.t - PAD.b);
  const path = (pts, close) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join('') +
    (close ? `L${X(1)},${Y(0)}L${X(0)},${Y(0)}Z` : '');

  function sheet() {
    if (document.getElementById('cutSheet')) return document.getElementById('cutSheet');
    const el = document.createElement('div');
    el.id = 'cutSheet'; el.className = 'cut-sheet hidden';
    el.innerHTML = `
      <div class="cut-head">
        <div><h3>Decision cut-off</h3><div class="cut-sub">State-of-the-art on the MHIST benchmark</div></div>
        <button class="cut-close" title="Close (Esc)">✕</button>
      </div>
      <svg class="cut-chart" viewBox="0 0 ${W} ${H}" tabindex="0" aria-label="Cut-off chart; arrow keys move the line">
        <path class="c-hp-fill"/><path class="c-ssl-fill"/><path class="c-hp"/><path class="c-ssl"/>
        <rect class="c-zone" y="${PAD.t}" height="${H - PAD.t - PAD.b}"/>
        <line class="c-snip hidden" y1="${PAD.t}" y2="${H - PAD.b}"/>
        <line class="c-cut" y1="${PAD.t - 4}" y2="${H - PAD.b}"/>
        <circle class="c-knob" r="6" cy="${PAD.t - 2}"/>
        <g class="c-axis">${[0, .25, .5, .75, 1].map(v => `<text x="${X(v)}" y="${H - 6}">${v * 100}%</text>`).join('')}</g>
        <text class="c-lab c-lab-hp" x="${X(.06)}" y="${PAD.t + 10}">HP</text>
        <text class="c-lab c-lab-ssl" x="${X(.94)}" y="${PAD.t + 10}" text-anchor="end">SSL</text>
      </svg>
      <div class="cut-read">
        <div><b class="r-ssl">–</b><span>SSLs caught</span></div>
        <div><b class="r-hp">–</b><span>HPs cleared</span></div>
        <div><b class="r-bal">–</b><span>balanced acc.</span></div>
      </div>
      <div class="cut-plain"></div>
      <div class="cut-presets">${PRESETS.map(([id, label, tip]) => `<button data-p="${id}" title="${tip}">${label}</button>`).join('')}</div>
      <div class="cut-foot"></div>`;
    document.body.appendChild(el);

    el.querySelector('.cut-close').onclick = close;
    el.querySelectorAll('.cut-presets button').forEach(b => (b.onclick = () => set(presetCut(b.dataset.p), b.dataset.p)));
    const svg = el.querySelector('.cut-chart');
    const toCut = e => { const r = svg.getBoundingClientRect(); return (e.clientX - r.left) / r.width * W; };
    const drag = e => set(Math.min(.99, Math.max(.01, (toCut(e) - PAD.l) / (W - PAD.l - PAD.r))), 'custom');
    svg.addEventListener('pointerdown', e => { svg.setPointerCapture(e.pointerId); drag(e); svg.onpointermove = drag; });
    svg.addEventListener('pointerup', () => (svg.onpointermove = null));
    svg.addEventListener('keydown', e => {
      const step = e.shiftKey ? .05 : .01;
      if (e.key === 'ArrowLeft') set(Math.max(.01, cut - step), 'custom');
      if (e.key === 'ArrowRight') set(Math.min(.99, cut + step), 'custom');
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !el.classList.contains('hidden')) { e.stopPropagation(); close(); } }, true);

    el.querySelector('.c-hp').setAttribute('d', path(density(D.hp)));
    el.querySelector('.c-ssl').setAttribute('d', path(density(D.ssl)));
    el.querySelector('.c-hp-fill').setAttribute('d', path(density(D.hp), true));
    el.querySelector('.c-ssl-fill').setAttribute('d', path(density(D.ssl), true));
    el.querySelector('.cut-foot').textContent =
      `${D.model}. Curves: ${D.ssl.length} SSL and ${D.hp.length} HP training images, each scored by a model that never trained on it; ` +
      `the test set is not used. Rates on new slides will be close to these, not exact.`;
    return el;
  }

  function draw() {
    const el = document.getElementById('cutSheet');
    if (!el || !D) return;
    const x = X(cut), r = rates(cut), pct = v => `${(v * 100).toFixed(1)}%`;
    el.querySelector('.c-cut').setAttribute('x1', x); el.querySelector('.c-cut').setAttribute('x2', x);
    el.querySelector('.c-knob').setAttribute('cx', x);
    const z = el.querySelector('.c-zone'); z.setAttribute('x', x); z.setAttribute('width', X(1) - x);
    el.querySelector('.r-ssl').textContent = pct(r.ssl);
    el.querySelector('.r-hp').textContent = pct(r.hp);
    el.querySelector('.r-bal').textContent = ((r.ssl + r.hp) / 2).toFixed(3);
    el.querySelector('.cut-plain').innerHTML =
      `Calls <b>SSL above ${(cut * 100).toFixed(0)}%</b>. Out of 100 SSLs it misses about <b>${Math.round((1 - r.ssl) * 100)}</b>; ` +
      `out of 100 HPs it flags about <b>${Math.round((1 - r.hp) * 100)}</b> as SSL.`;
    el.querySelectorAll('.cut-presets button').forEach(b => b.classList.toggle('on', b.dataset.p === preset));
    const s = el.querySelector('.c-snip');
    s.classList.toggle('hidden', score == null);
    if (score != null) { s.setAttribute('x1', X(score)); s.setAttribute('x2', X(score)); }
  }

  function set(t, p) {
    cut = Math.round(t * 1000) / 1000; preset = p;
    try { localStorage.setItem(STORE, JSON.stringify({ cut, preset })); } catch {}
    draw();
    listeners.forEach(f => f(api.state()));
  }
  function open(anchor) {  // anchor: a DOMRect to open over (e.g. the results panel)
    const el = sheet();
    if (anchor && anchor.width) Object.assign(el.style, { left: Math.max(12, anchor.left) + 'px', top: Math.max(12, anchor.top) + 'px', right: 'auto' });
    el.classList.remove('hidden'); draw(); el.querySelector('.cut-chart').focus();
  }
  function close() { const el = document.getElementById('cutSheet'); if (el) el.classList.add('hidden'); }

  const api = {
    async load(url = 'cutoff_data.json') {
      D = window.CUTOFF_DATA || await (await fetch(url)).json();  // Electron pages load the .js copy
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(STORE) || 'null'); } catch {}
      if (saved && saved.preset && saved.preset !== 'custom') set(presetCut(saved.preset), saved.preset);
      else if (saved) set(saved.cut, 'custom');
      else set(presetCut('balanced'), 'balanced');
      return api.state();
    },
    open, close,
    setScore(p) { score = p; draw(); },     // mark the current snip's P(SSL) on the chart
    onChange(f) { listeners.push(f); },
    state() {
      const label = (PRESETS.find(p => p[0] === preset) || [0, 'Custom'])[1];
      return { cut, preset, label, ...(D ? rates(cut) : {}) };
    },
    presetCut,
  };
  window.Cutoff = api;
})();
