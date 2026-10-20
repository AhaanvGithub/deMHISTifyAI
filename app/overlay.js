// Snip -> tissue box -> analyze -> heatmap + side panel.
const API = window.demhist.backend;
const LABEL = { SSA: 'SSL', HP: 'HP' };
const FULL = { SSA: 'Sessile serrated lesion', HP: 'Hyperplastic polyp' };
const $ = id => document.getElementById(id);

const shot = $('shot'), dim = $('dim'), sel = $('sel'), tissue = $('tissue'), heat = $('heat');
let img, scale = 1, researcherText = '';
let selRect = null;         // selection in CSS px (screen)
let tissueBox = null;       // tissue box in CSS px, relative to selection
let state = 'select';       // select | region | results
let result = null, parsed = null, heatRef = 'ai';  // AI's map first, then compare with your call

window.demhist.onInit(({ screenshot, researcher }) => {
  researcherText = researcher;
  $('chip').textContent = `“${researcher}”`;
  img = new Image();
  img.onload = () => { scale = img.width / window.innerWidth; drawShot(); drawDim(); };
  // raw PNG bytes from the main process (a data URL string from older builds)
  img.src = typeof screenshot === 'string' ? screenshot : URL.createObjectURL(new Blob([screenshot], { type: 'image/png' }));
});

function fit(c) { c.width = window.innerWidth; c.height = window.innerHeight; }
// draw the screenshot at full device resolution (not CSS px) so the tissue keeps its native detail
function drawShot() {
  const dpr = window.devicePixelRatio || 1;
  shot.width = Math.round(window.innerWidth * dpr); shot.height = Math.round(window.innerHeight * dpr);
  const g = shot.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(img, 0, 0, shot.width, shot.height);
}
function drawDim(r) {
  fit(dim);
  const g = dim.getContext('2d');
  g.fillStyle = 'rgba(22,8,32,0.62)';  // plum dim (sunset theme)
  g.fillRect(0, 0, dim.width, dim.height);
  if (r) g.clearRect(r.x, r.y, r.w, r.h);
}

// ---------- 1. selection ----------
let drag = null;
dim.addEventListener('mousedown', e => {
  if (state !== 'select') return;
  drag = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mousemove', e => {
  if (state === 'select' && drag) {
    selRect = norm(drag.x, drag.y, e.clientX, e.clientY);
    drawDim(selRect); place(sel, selRect); sel.classList.remove('hidden');
    const tag = $('sizeTag');
    tag.classList.remove('hidden');
    tag.textContent = `${Math.round(selRect.w * scale)} × ${Math.round(selRect.h * scale)}`;
    tag.style.left = selRect.x + 'px'; tag.style.top = (selRect.y + selRect.h + 8) + 'px';
  }
  if (boxDrag) moveBox(e);
});
window.addEventListener('mouseup', () => {
  if (state === 'select' && drag) {
    drag = null;
    if (selRect && selRect.w > 40 && selRect.h > 40) startRegion();
  }
  boxDrag = null;
});
function norm(x0, y0, x1, y1) { return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) }; }
function place(el, r) { Object.assign(el.style, { left: r.x + 'px', top: r.y + 'px', width: r.w + 'px', height: r.h + 'px' }); }

function cropBlob(r) {
  const c = document.createElement('canvas');
  c.width = Math.round(r.w * scale); c.height = Math.round(r.h * scale);
  c.getContext('2d').drawImage(img, r.x * scale, r.y * scale, r.w * scale, r.h * scale, 0, 0, c.width, c.height);
  return new Promise(res => c.toBlob(res, 'image/png'));
}
const selCrop = () => cropBlob(selRect);
const tissueCrop = () => cropBlob({ x: selRect.x + tissueBox.x, y: selRect.y + tissueBox.y, w: tissueBox.w, h: tissueBox.h });

// ---------- 2. tissue region (instant guess, then vision-model refine) ----------
async function postImage(path, blob, extra = {}) {
  const fd = new FormData();
  fd.append('image', blob, 'snip.png');
  for (const [k, v] of Object.entries(extra)) fd.append(k, v);
  return fetch(API + path, { method: 'POST', body: fd });
}
// The tissue box is always square: the models were trained on square fields, and a wide box would be centre-cropped
// by every model (its ends silently dropped). A proposed box becomes the largest square centred inside it.
function squareIn(b) {
  const side = Math.max(32, Math.min(b.w, b.h, selRect.w, selRect.h));
  const x = Math.min(Math.max(0, b.x + (b.w - side) / 2), selRect.w - side);
  const y = Math.min(Math.max(0, b.y + (b.h - side) / 2), selRect.h - side);
  return { x, y, w: side, h: side };
}
let fixedField = false;  // true when the viewer's scale bar was read: the square is exactly one 448 um field
function setTissue(boxPx, label) {
  // box comes in image pixels of the selection crop
  const raw = { x: boxPx[0] / scale, y: boxPx[1] / scale, w: (boxPx[2] - boxPx[0]) / scale, h: (boxPx[3] - boxPx[1]) / scale };
  if (fixedField) { tissueBox = raw; place(tissue, tissueBox); tissue.classList.remove('hidden'); $('tissueTag').textContent = label; return; }
  tissueBox = squareIn({ x: boxPx[0] / scale, y: boxPx[1] / scale, w: (boxPx[2] - boxPx[0]) / scale, h: (boxPx[3] - boxPx[1]) / scale });
  place(tissue, tissueBox); tissue.classList.remove('hidden');
  $('tissueTag').textContent = label;
}
async function startRegion() {
  state = 'region';
  sel.classList.add('snap');  // flash when the selection lands
  $('hint').textContent = 'Adjust the tissue box if needed, then press Enter';
  $('sizeTag').classList.add('hidden');
  showActions();
  const blob = await selCrop();
  setTissue([0, 0, selRect.w * scale, selRect.h * scale], 'detecting tissue…');
  try {
    const g = await (await postImage('/region/guess', blob)).json();
    if (state === 'region') {
      tissue.classList.add('anim');
      fixedField = !!g.fixed;
      tissue.classList.toggle('fixed', fixedField);
      const advice = { zoom_out: ' · zoom out: one field is bigger than your selection',
                       zoom_in: ' · zoom in: one field is under 224 screen pixels' }[g.advice] || '';
      setTissue(g.box, fixedField
        ? `one 448 µm field (scale bar ${g.scale.label_um} µm read)${advice} · drag to move · Enter to analyze`
        : 'one field on the tissue · drag or resize · Enter to analyze');
      setTimeout(() => tissue.classList.remove('anim'), 400);
    }
  } catch { $('tissueTag').textContent = 'backend offline? drag to adjust'; }
}
function showActions() {
  const a = $('actions'); a.classList.remove('hidden');
  a.style.left = selRect.x + 'px';
  a.style.top = Math.min(selRect.y + selRect.h + 12, window.innerHeight - 56) + 'px';
}

// drag / resize the tissue box
let boxDrag = null;
tissue.addEventListener('mousedown', e => {
  if (state !== 'region') return;
  e.stopPropagation();
  const h = e.target.classList.contains('handle') ? [...e.target.classList].find(c => c.length === 2) : 'move';
  boxDrag = { mode: h, x: e.clientX, y: e.clientY, start: { ...tissueBox } };
});
function moveBox(e) {
  const dx = e.clientX - boxDrag.x, dy = e.clientY - boxDrag.y, s = boxDrag.start, m = boxDrag.mode, b = { ...s };
  if (m === 'move' && fixedField) {  // fixed-size field: move anywhere on screen, never resize
    b.x = Math.max(-selRect.x, Math.min(s.x + dx, window.innerWidth - selRect.x - s.w));
    b.y = Math.max(-selRect.y, Math.min(s.y + dy, window.innerHeight - selRect.y - s.h));
  } else if (m === 'move') {
    b.x = Math.max(0, Math.min(s.x + dx, selRect.w - s.w)); b.y = Math.max(0, Math.min(s.y + dy, selRect.h - s.h));
  } else if (fixedField) {
    return;
  } else {  // corner handle: resize as a square, the opposite corner stays put
    const left = m.includes('w'), top = m.includes('n');
    const ax = left ? s.x + s.w : s.x, ay = top ? s.y + s.h : s.y;  // anchor = opposite corner
    const want = Math.max(left ? s.w - dx : s.w + dx, top ? s.h - dy : s.h + dy);
    const room = Math.min(left ? ax : selRect.w - ax, top ? ay : selRect.h - ay);
    const side = Math.max(32, Math.min(want, room));
    b.w = b.h = side; b.x = left ? ax - side : ax; b.y = top ? ay - side : ay;
  }
  tissueBox = b; place(tissue, b);
  $('tissueTag').textContent = 'adjusted · press Enter to analyze';
}

// ---------- 3. analyze ----------
$('analyzeBtn').onclick = () => state === 'region' && analyze();
$('resetBtn').onclick = () => location.reload();
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') window.demhist.close();
  if (e.key === 'Enter' && state === 'region') analyze();
});

async function analyze() {
  state = 'results';
  $('actions').classList.add('hidden'); $('hint').classList.add('hidden');
  tissue.classList.add('locked'); $('tissueTag').textContent = 'image panel looking…';
  layoutPanel();
  const res = await postImage('/analyze', await tissueCrop(), { researcher: researcherText, cut: Cutoff.state().cut });
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = /event: (\w+)/.exec(chunk)?.[1], data = /data: (.*)/s.exec(chunk)?.[1];
      if (ev) handle(ev, JSON.parse(data));
    }
  }
}

function layoutPanel() {
  const p = $('panel'), W = 400, gap = 20;
  const right = window.innerWidth - (selRect.x + selRect.w), left = selRect.x;
  p.classList.remove('hidden');
  p.style.width = W + 'px';
  if (right >= left) { p.style.left = Math.min(selRect.x + selRect.w + gap, window.innerWidth - W - 12) + 'px'; }
  else { p.style.left = Math.max(selRect.x - W - gap, 12) + 'px'; }
  p.style.transformOrigin = right >= left ? 'top left' : 'top right';  // grow out from the selection side
  // line the panel up with the selection: same top edge, and (once the verdict is in) the same bottom edge,
  // so snip + panel read as one clean rectangle. Small snips get a minimum height, shifted up if needed.
  const MIN_H = 360;
  panelBox.h = Math.min(Math.max(selRect.h, MIN_H), window.innerHeight - 24);
  panelBox.top = Math.min(selRect.y, window.innerHeight - 12 - panelBox.h);
  p.style.top = panelBox.top + 'px';
  growPanel();
  const hc = $('heatControls');
  hc.style.left = (selRect.x + tissueBox.x) + 'px';
  hc.style.top = Math.min(selRect.y + selRect.h + 36, window.innerHeight - 52) + 'px';
}

// the panel starts compact and its height eases to fit its content as each section fills in
// (capped at the selection's height); when the verdict is in it settles at exactly the selection's height
let panelRO = null, panelDone = false, fitPanel = () => {};
const panelBox = { top: 64, h: 600 };
function growPanel() {
  if (panelRO) return;
  const p = $('panel'), inner = $('panelInner');
  fitPanel = () => {
    const want = inner.scrollHeight + 34;  // 2 x 16px padding + border
    const h = panelDone ? panelBox.h : Math.min(want, panelBox.h);
    p.style.height = h + 'px';
    p.style.overflowY = want > h ? 'auto' : 'hidden';
  };
  panelRO = new ResizeObserver(fitPanel);
  panelRO.observe(inner);
  fitPanel();
}
function finishPanel() { panelDone = true; fitPanel(); }

function handle(ev, d) {
  if (ev === 'parsed') return renderYou(d);
  if (ev === 'conch') return renderConch(d);
  if (ev === 'heat') {
    result.heat = d;
    $('tissueTag').textContent = d.kind === 'evidence'
      ? `what drives the call · ${d.level} × ${d.level} per field${d.level < 128 ? ' · sharpening…' : ''} · hover for detail`
      : `${result.grid.cols} × ${result.grid.rows} tiles · ${d.cols} × ${d.rows} heatmap cells (sharpened) · hover for detail`;
    return drawHeat();
  }
  if (ev === 'judge') return renderJudge(d);
  if (ev === 'flags') return renderFlags(d);
  if (ev === 'verdict') return renderVerdict(d);
  if (ev === 'error') {
    $('panelSummary').innerHTML = `<span class="bad">${d.message === 'no_tissue' ? 'No tissue found in the box.' : 'Error: ' + d.message}</span>`;
    $('litSummary').textContent = '—'; $('verdict').textContent = '—';
    $('litCard').classList.remove('pending'); $('verdictCard').classList.remove('pending');
    finishPanel();
  }
}

const pct = (label, n, cls = '') => `<span class="call ${cls}">${label} <span class="num" data-to="${Number(n) || 0}">0</span>%</span>`;
// numbers count up when they appear
function countUp(root = document) {
  root.querySelectorAll('.num[data-to]').forEach(el => {
    const to = +el.dataset.to; el.removeAttribute('data-to');
    const t0 = performance.now(), dur = 700;
    const step = t => { const k = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - k, 3); el.textContent = Math.round(to * e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });
}
new MutationObserver(() => countUp()).observe(document.body, { childList: true, subtree: true });

function renderYou(p) {
  parsed = p;
  $('youCard').innerHTML = `<h3>You</h3>` + (p.call
    ? `<div class="summary">${pct(LABEL[p.call], p.certainty ?? '–', 'you')}<div class="sub">${FULL[p.call]}</div></div>`
    : `<div class="summary"><span class="warn">“${p.text}”: outside the validated scope</span></div>`);
}

function renderConch(r) {
  result = r;
  heat.classList.add('reveal');
  $('litCard').classList.remove('pending');  // panel grows one section at a time as results arrive  // heatmap opens out from the centre the first time
  const mz = r.magnification || {};
  $('tissueTag').textContent = r.heat.kind === 'evidence' ? `what drives the call · ${r.heat.level} × ${r.heat.level} per field · sharpening…`
    : `${r.grid.cols} × ${r.grid.rows} tiles${mz.auto ? ` · zoom ×${mz.zoom_vs_training} vs training` : ''} · ${r.heat.cols} × ${r.heat.rows} cells · hover for detail`;
  $('heatControls').classList.remove('hidden');
  drawHeat();
  renderPanelSummary();
  if (r.panel.p_ssl != null) Cutoff.setScore(r.panel.p_ssl);
  const box = $('judges'); box.innerHTML = '';
  r.judges.forEach((j, i) => setTimeout(() => box.appendChild(judgeEl(j)), 90 * i));
}

// the call follows the cut-off setting (gear): re-derive it from the panel's P(SSL) whenever the setting moves
function applyCut(o) {
  const cut = Cutoff.state().cut, top = o.p_ssl > cut ? 'SSA' : 'HP';
  return { ...o, top, top_conf: Math.round(100 * (top === 'SSA' ? o.p_ssl : 1 - o.p_ssl)),
           alt: top === 'SSA' ? 'HP' : 'SSA', alt_conf: Math.round(100 * (top === 'SSA' ? 1 - o.p_ssl : o.p_ssl)) };
}
function renderPanelSummary() {
  const r = result; if (!r) return;
  const panelMode = r.judge_mode === 'panel';
  const judges = panelMode ? r.judges.map(applyCut) : r.judges;
  const votes = panelMode ? { SSA: judges.filter(j => j.top === 'SSA').length, HP: judges.filter(j => j.top === 'HP').length } : r.panel.votes;
  const pnl = panelMode ? { ...applyCut(r.panel), votes } : r.panel;
  $('judgesNote').textContent = panelMode ? 'ConvNeXt + MUSK, combined' : '4 vote · 4 colour checks';
  const agree = parsed?.call && pnl.top === parsed.call;
  $('panelSummary').classList.remove('muted');
  $('panelSummary').innerHTML =
    `${pct(LABEL[pnl.top], pnl.top_conf, agree || !parsed?.call ? 'good' : 'bad')}
     <div class="alt">- alt: ${LABEL[pnl.alt]} ${pnl.alt_conf}%</div>
     <div class="votes">${Object.entries(pnl.votes).map(([c, n]) => `${LABEL[c]} ${n}/${pnl.n_voters || r.judges.length}`).join(' · ')}</div>
     ${panelMode ? `<div class="sub">${Math.round(100 * pnl.p_ssl)}% chance a 7-pathologist majority says SSL · calls SSL above ${Math.round(100 * Cutoff.state().cut)}%</div>` : ''}`;
  countUp($('panelSummary'));
  if (panelMode) document.querySelectorAll('#judges .judge').forEach((el, i) => el.replaceWith(judgeEl(judges[i], true)));
}
function judgeEl(j0, still = false) {
  const panelMode = result.judge_mode === 'panel', j = panelMode ? applyCut(j0) : j0;
  const el = document.createElement('div');
  el.className = 'judge' + (still ? '' : ' pop') + (j.votes === false ? ' nonvoting' : '');
  el.title = panelMode ? `${j.prompts.SSA} · ${j.prompts.HP}`
    : `${j.prompts.SSA}\nvs\n${j.prompts.HP}` + (j.votes === false ? '\n\ncolour check only: does not vote (these versions were trained on raw images)' : '');
  el.innerHTML = `<span class="jid">${j.id}</span>${panelMode ? '' : `<span class="cond">${j.condition === 'raw' ? 'raw' : 'norm'}</span>`}
                  <b class="${j.top === 'SSA' ? 'ssl' : 'hp'}">${LABEL[j.top]} ${j.top_conf}</b>
                  <span class="jalt">${LABEL[j.alt]} ${j.alt_conf}</span>
                  <span class="jbar ${j.top === 'SSA' ? 'ssl' : 'hp'}"></span>`;
  requestAnimationFrame(() => requestAnimationFrame(() => { el.querySelector('.jbar').style.width = j.top_conf + '%'; }));
  return el;
}

// jet-style colour scale like pathology probability maps: blue -> cyan -> green -> yellow -> red
const JET = [[0, [40, 60, 220]], [0.3, [40, 200, 240]], [0.5, [70, 220, 120]], [0.7, [250, 225, 60]], [0.85, [250, 130, 40]], [1, [220, 30, 40]]];
function jet(v) {
  for (let i = 1; i < JET.length; i++) if (v <= JET[i][0]) {
    const [a, ca] = JET[i - 1], [b, cb] = JET[i], t = (v - a) / (b - a);
    return ca.map((x, k) => Math.round(x + (cb[k] - x) * t));
  }
  return JET[JET.length - 1][1];
}
const smooth = (e0, e1, x) => { const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); };
const refClass = () => heatRef === 'ai' ? 'SSA' : (parsed?.call || result.panel.top);

function drawHeat() {
  if (!result) return;
  const ref = refClass(), H = result.heat, dpr = window.devicePixelRatio || 1;
  const rect = tissue.getBoundingClientRect();
  for (const c of [heat, $('feat')]) { c.width = Math.round(rect.width * dpr); c.height = Math.round(rect.height * dpr); }
  const g = heat.getContext('2d'), a = $('opacity').value / 100;
  g.clearRect(0, 0, heat.width, heat.height);

  // 1. probability of the reference call per fine cell; low values fade to transparent
  const small = document.createElement('canvas');
  small.width = H.cols; small.height = H.rows;
  const sg = small.getContext('2d'), im = sg.createImageData(H.cols, H.rows);
  const ai = heatRef === 'ai';
  const ev = H.kind === 'evidence';  // cover-up map: 0.5 = no effect, above = pushed toward SSL, below = toward HP
  const c0 = result.judge_mode === 'panel' && !ev ? Cutoff.state().cut : 0.5;
  const centre = p => (p <= c0 ? 0.5 * p / c0 : 0.5 + 0.5 * (p - c0) / (1 - c0));  // cut-off -> middle of the scale
  // A whole-snip field gives (nearly) one value everywhere: a uniform tint says nothing about *where* and only
  // washes out the tissue, so a flat map is not painted (the call and colour bar still show the value).
  const tv = H.p_ssa.filter((_, i) => H.tissue[i] >= 0.25);
  const flat = tv.length && Math.max(...tv) - Math.min(...tv) < 0.03;
  for (let i = 0; i < H.p_ssa.length; i++) {
    const p = ai ? centre(H.p_ssa[i]) : H.p_ssa[i];
    // AI view: fixed P(SSL) scale, HP-leaning tissue blue, SSL-leaning red, undecided (~0.5) fades out.
    // Your-call view: support for your call, clear when unlikely, then cyan -> green -> yellow -> red.
    const v = ai ? p : (ref === 'SSA' ? p : 1 - p);
    // soft fade from 'no effect' (transparent) to strong evidence, so empty cells don't cut hard edges
    const k = ai ? smooth(0.0, 0.4, Math.abs(p - 0.5)) : ev ? smooth(0.0, 0.4, v - 0.5) : smooth(0.22, 0.6, v);
    const alpha = H.tissue[i] < 0.25 || flat ? 0 : (ai ? 0.55 : 0.8) * k;  // never opaque: tissue must stay readable
    im.data.set([...jet(v), Math.round(255 * a * alpha)], i * 4);
  }
  sg.putImageData(im, 0, 0);
  // two-step upscale + blur gives soft blob edges like slide-level probability maps
  const mid = document.createElement('canvas');
  const cp = Math.max(8, Math.round(1024 / Math.max(H.cols, H.rows)));  // px per cell on the blur canvas
  mid.width = H.cols * cp; mid.height = H.rows * cp;
  const mg = mid.getContext('2d');
  mg.imageSmoothingEnabled = true; mg.imageSmoothingQuality = 'high';
  mg.filter = `blur(${Math.round(cp * 1.2)}px)`;  // ~1.2 cells at every refinement level
  mg.drawImage(small, 0, 0, mid.width, mid.height);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(mid, 0, 0, heat.width, heat.height);

  // 2. optional grid: faint sub-cells, stronger lines per judge tile
  const show = $('showGrid').checked;
  if (show) {
    const cw = heat.width / H.cols, ch = heat.height / H.rows;
    const line = (x0, y0, x1, y1) => { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
    g.lineWidth = 1; g.strokeStyle = 'rgba(255,255,255,0.06)';
    // fine lines no closer than ~12 device px, so a 64-cell map doesn't turn into a grey mesh
    const every = Math.max(1, Math.ceil(12 * dpr / Math.min(cw, ch)));
    for (let c = every; c < H.cols; c += every) if (c % H.sub) line(Math.round(c * cw) + .5, 0, Math.round(c * cw) + .5, heat.height);
    for (let r = every; r < H.rows; r += every) if (r % H.sub) line(0, Math.round(r * ch) + .5, heat.width, Math.round(r * ch) + .5);
    g.strokeStyle = 'rgba(255,255,255,0.28)'; g.lineWidth = Math.max(1, dpr);
    for (let c = H.sub; c < H.cols; c += H.sub) line(Math.round(c * cw), 0, Math.round(c * cw), heat.height);
    for (let r = H.sub; r < H.rows; r += H.sub) line(0, Math.round(r * ch), heat.width, Math.round(r * ch));
  }
  const tw = rect.width / result.grid.cols, th = rect.height / result.grid.rows;
  $('colLabels').innerHTML = show ? Array.from({ length: result.grid.cols }, (_, i) => `<span style="width:${tw}px">${i + 1}</span>`).join('') : '';
  $('rowLabels').innerHTML = show ? Array.from({ length: result.grid.rows }, (_, i) => `<span style="height:${th}px">${String.fromCharCode(65 + (i % 26))}</span>`).join('') : '';

  $('colorbar').classList.remove('hidden');
  $('colorbar').classList.toggle('diverging', ai);
  if (ev) {
    $('cbLabel').textContent = ai ? 'what drives the call' : `support for ${LABEL[ref]} · your call`;
    $('cbTop').textContent = ai ? '→ SSL' : 'strong'; $('cbBottom').textContent = ai ? '→ HP' : '';
    $('cbMid').textContent = ai ? 'no effect' : '';
  } else {
    $('cbLabel').textContent = ai ? 'P(SSL)' : `P(${LABEL[ref]}) · your call`;
    $('cbTop').textContent = ai ? '1.0 SSL' : '1.0'; $('cbBottom').textContent = ai ? '0 HP' : '0';
    $('cbMid').textContent = ai && result.judge_mode === 'panel' ? `${Math.round(Cutoff.state().cut * 100)}% cut-off` : '0.5';
  }
  drawFeatures();
}

// ---------- feature outlines from the literature judge ----------
const FEAT_COLORS = ['#ffc857', '#c79bff', '#ffffff', '#5eead4'];  // outlines must stand out on tissue and on the jet heatmap
let keyFeatures = [];
// ---------- measured crypts (backend/glands.py): outlines follow real glands, labels come from measurements ----------
const GLAND_COLORS = { 'dilated crypt': '#ffc857', 'serrated lumen': '#c79bff', 'round regular crypt': '#5eead4' };
function drawGlands(fc, g) {
  const W = fc.width, Hh = fc.height, dpr = window.devicePixelRatio || 1, rect = tissue.getBoundingClientRect();
  const placed = [];
  const free = (x, y, w, h) => !placed.some(b => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
  // strongest evidence first, so the glands that drove the call get their labels before any collision
  [...result.glands].sort((a, b) => !!b.label - !!a.label || Math.abs(b.evidence) - Math.abs(a.evidence)).forEach(gl => {
    const col = GLAND_COLORS[gl.label];
    if (!col) return;  // only crypts with a clear ring and a clear measurement are drawn
    g.beginPath();
    gl.outline.forEach(([x, y], i) => (i ? g.lineTo(x * W, y * Hh) : g.moveTo(x * W, y * Hh)));
    g.closePath();
    g.lineJoin = 'round'; g.shadowColor = 'rgba(0,0,0,.55)'; g.shadowBlur = 3 * dpr;
    g.strokeStyle = col; g.lineWidth = 2 * dpr;
    g.stroke(); g.shadowBlur = 0;
    // label just above the gland (or below if that spot is taken); skipped if both collide - hover still shows it
    const top = gl.outline.reduce((m, p) => (p[1] < m[1] ? p : m)), bot = gl.outline.reduce((m, p) => (p[1] > m[1] ? p : m));
    const text = (gl.tags && gl.tags.length > 1) ? gl.tags.join(' · ') : gl.label, w = text.length * 6.2 + 18, h = 20;
    for (const [px, py] of [[gl.center[0] * rect.width, top[1] * rect.height - h - 2], [gl.center[0] * rect.width, bot[1] * rect.height + 2]]) {
      const x = Math.min(Math.max(px - w / 2, 0), rect.width - w), y = Math.min(Math.max(py, 0), rect.height - h);
      if (!free(x, y, w, h)) continue;
      placed.push({ x, y, w, h });
      const lab = document.createElement('div');
      lab.className = 'gland-label';
      lab.style.borderColor = col; lab.style.left = x + 'px'; lab.style.top = y + 'px';
      lab.innerHTML = `${esc(text)}${gl.points_to ? ` <span class="${gl.points_to === 'SSA' ? 'ssl' : 'hp'}">→ ${LABEL[gl.points_to]}</span>` : ''}`;
      tissue.appendChild(lab);
      break;
    }
  });
}
const glandAt = (x, y) => (result?.glands || []).find(gl => {  // point-in-polygon, x/y in 0..1 of the snip
  let inside = false;
  for (let i = 0, j = gl.outline.length - 1; i < gl.outline.length; j = i++) {
    const [xi, yi] = gl.outline[i], [xj, yj] = gl.outline[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
});

function drawFeatures() {
  const fc = $('feat'), g = fc.getContext('2d');
  g.clearRect(0, 0, fc.width, fc.height);
  document.querySelectorAll('.feat-label, .gland-label').forEach(e => e.remove());
  if (result?.glands && $('showFeat').checked) return drawGlands(fc, g);
  if (!result || !$('showFeat').checked || !keyFeatures.length) return;
  const H = result.heat, cw = fc.width / H.cols, ch = fc.height / H.rows, dpr = window.devicePixelRatio || 1;
  keyFeatures.forEach((f, k) => {
    const m = H.features && H.features[f.feature_id];
    if (!m) return;
    const vals = m.filter((_, i) => H.tissue[i] >= 0.25).sort((x, y) => x - y);
    if (!vals.length) return;
    const thr = Math.max(0.7, vals[Math.floor(vals.length * 0.9)]);  // strongest ~10% of tissue, and clearly above typical
    const on = (r, c) => r >= 0 && c >= 0 && r < H.rows && c < H.cols && m[r * H.cols + c] >= thr && H.tissue[r * H.cols + c] >= 0.25;
    g.strokeStyle = FEAT_COLORS[k % FEAT_COLORS.length]; g.lineWidth = 2 * dpr; g.lineJoin = g.lineCap = 'round';
    g.shadowColor = 'rgba(0,0,0,.6)'; g.shadowBlur = 3 * dpr;
    let best = null;
    g.beginPath();
    for (let r = 0; r < H.rows; r++) for (let c = 0; c < H.cols; c++) {
      if (!on(r, c)) continue;
      const x = c * cw, y = r * ch, v = m[r * H.cols + c];
      if (!best || v > best.v) best = { r, c, v };
      if (!on(r - 1, c)) { g.moveTo(x, y); g.lineTo(x + cw, y); }
      if (!on(r + 1, c)) { g.moveTo(x, y + ch); g.lineTo(x + cw, y + ch); }
      if (!on(r, c - 1)) { g.moveTo(x, y); g.lineTo(x, y + ch); }
      if (!on(r, c + 1)) { g.moveTo(x + cw, y); g.lineTo(x + cw, y + ch); }
    }
    g.stroke(); g.shadowBlur = 0;
    if (best) {
      const lab = document.createElement('div');
      lab.className = 'feat-label';
      lab.style.borderColor = FEAT_COLORS[k % FEAT_COLORS.length];
      lab.style.left = ((best.c + 0.5) * cw / dpr) + 'px';
      lab.style.top = ((best.r + 0.5) * ch / dpr) + 'px';
      lab.innerHTML = `${esc(f.feature)} <span class="${f.points_to === 'SSA' ? 'ssl' : 'hp'}">→ ${LABEL[f.points_to] || ''}</span>${(f.cites || []).map(c => `<sup>${c}</sup>`).join('')}`;
      tissue.appendChild(lab);
    }
  });
}
$('showGrid').addEventListener('change', drawHeat);
$('showFeat').addEventListener('change', drawFeatures);

// hover readout: tile + sub-cell coordinates and local support for the current reference call
const tip = document.createElement('div');
tip.className = 'heat-tip hidden';
document.body.appendChild(tip);
tissue.addEventListener('mousemove', e => {
  if (state !== 'results' || !result) return tip.classList.add('hidden');
  const r = tissue.getBoundingClientRect(), H = result.heat;
  const fc = Math.min(H.cols - 1, Math.floor((e.clientX - r.left) / r.width * H.cols));
  const fr = Math.min(H.rows - 1, Math.floor((e.clientY - r.top) / r.height * H.rows));
  const ref = heatRef === 'ai' ? 'SSA' : (parsed?.call || result.panel.top);  // AI view always reads P(SSL)
  const p = H.p_ssa[fr * H.cols + fc], s = ref === 'SSA' ? p : 1 - p;
  const feats = Object.entries(H.features || {}).sort((x, y) => y[1][fr * H.cols + fc] - x[1][fr * H.cols + fc]).slice(0, 2).map(([k]) => k.replace(/_/g, ' '));
  const gl = glandAt((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  const glandLine = gl ? `<div><b style="color:${GLAND_COLORS[gl.label] || 'var(--text)'}">${gl.label || 'crypt'}</b> · ${gl.crypt_um} µm across` +
    `${gl.roughness != null ? ` · lumen edge roughness ${gl.roughness}` : ''}</div>` : '';
  const tile = `${String.fromCharCode(65 + Math.floor(fr / H.sub))}${Math.floor(fc / H.sub) + 1}`;
  const subc = `${(fr % H.sub) + 1}-${(fc % H.sub) + 1}`;  // row-column inside the field
  tip.innerHTML = H.tissue[fr * H.cols + fc] < 0.25 ? `${tile}·${subc} <span class="muted">no tissue</span>`
    : `${tile}·${subc} ${H.kind === 'evidence'
        ? `<b style="color:rgb(${jet(p)})">${Math.abs(p - 0.5) < 0.06 ? 'little effect on the call' : `pushes toward ${p > 0.5 ? 'SSL' : 'HP'}`}</b>`
        : `<b style="color:rgb(${jet(s)})">P(${LABEL[ref]}) ${Math.round(s * 100)}%</b>`}${result.glands ? glandLine : `<div class="muted">strongest here: ${feats.join(', ')}</div>`}`;
  tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  tip.classList.remove('hidden');
});
tissue.addEventListener('mouseleave', () => tip.classList.add('hidden'));
$('opacity').addEventListener('input', drawHeat);
document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b));
  heatRef = b.dataset.ref; drawHeat();
});

// stream text in word by word (and list items one by one) so LLM output feels like it's being written
function reveal(root, perWord = 14, maxDelay = 1600) {
  let i = 0;
  const walk = node => {
    for (const n of [...node.childNodes]) {
      if (n.nodeType === 3 && n.textContent.trim()) {
        const frag = document.createDocumentFragment();
        n.textContent.split(/(\s+)/).forEach(t => {
          if (!t) return;
          if (/^\s+$/.test(t)) return frag.appendChild(document.createTextNode(t));
          const w = document.createElement('span');
          w.className = 'w'; w.textContent = t; w.style.animationDelay = Math.min(i++ * perWord, maxDelay) + 'ms';
          frag.appendChild(w);
        });
        n.replaceWith(frag);
      } else if (n.nodeType === 1 && n.tagName === 'SUP') {
        n.classList.add('w'); n.style.animationDelay = Math.min(i * perWord, maxDelay) + 'ms';
      } else if (n.nodeType === 1 && n.tagName === 'LI') {
        n.classList.add('rv'); n.style.animationDelay = Math.min(i++ * perWord * 3, maxDelay) + 'ms';
      } else if (n.nodeType === 1) walk(n);
    }
  };
  walk(root);
}

// LLM citations come as [P1], [P1, P4], [P1; P2] or bare P8: render every P# as a superscript
const cite = s => esc(s).replace(/\[\s*P\d+(?:\s*[,;&]\s*P\d+)*\s*\]|\bP\d+\b/g, m => m.match(/P\d+/g).map(r => `<sup>${r}</sup>`).join(''));
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function renderJudge(j) {
  $('verdictCard').classList.remove('pending');
  keyFeatures = (j.key_features || []).filter(f => f.feature_id && result?.criteria?.[f.feature_id]?.usable).slice(0, 3);
  drawFeatures();
  const el = $('litSummary'); el.classList.remove('muted');
  if (j.error) {
    el.innerHTML = j.error === 'not_tissue'
      ? '<span class="warn">Skipped: the selection doesn\'t look like colon polyp tissue.</span>'
      : '<span class="warn">Literature judge unavailable (Lumen).</span>';
    return;
  }
  const agree = !parsed?.call || j.call === parsed.call;
  el.innerHTML = `${pct(LABEL[j.call], j.consistency, agree ? 'good' : 'bad')}
    <div class="alt">- alt: ${LABEL[j.alt]} ${j.alt_consistency}%</div>
    <div class="sub">consistency over ${j.n_samples} runs · model's own estimate ${j.stated_confidence}%${j.agrees_with_panel === false ? ' · <span class="warn">disagrees with the image panel</span>' : j.agrees_with_panel ? ' · agrees with the image panel' : ''}</div>`;
  const feats = (j.key_features || []).map(f =>
    `<li>${LABEL[f.points_to] ? `<b class="${f.points_to === 'SSA' ? 'ssl' : 'hp'}">${LABEL[f.points_to]}</b> ` : '<b class="muted">either</b> '}${esc(f.feature)}${f.strength ? ` <span class="strength">${String(f.feature_id || '').startsWith('crypts:') ? 'measured' : 'score'} · ${esc(f.strength)}</span>` : ''}: ${cite(f.why)} ${(f.cites || []).map(c => `<sup>${c}</sup>`).join('')}</li>`).join('');
  // each cited passage: what it covers, then the primary papers it draws on
  const cites = (j.citations || []).map(c => {
    const src = (c.sources || []).slice(0, 3).map(s =>
      `<div class="src">${esc(s.title)}${s.year ? ` (${s.year})` : ''}${s.doi ? ` · <span class="doi">${esc(s.doi)}</span>` : ''}</div>`).join('');
    const more = (c.sources || []).length > 3 ? `<div class="src muted">+${c.sources.length - 3} more</div>` : '';
    return `<li><sup>${c.ref}</sup> <span class="ptitle">${esc(c.title)}</span>${src}${more}</li>`;
  }).join('');
  $('litBody').innerHTML = `<p class="reason">${cite(j.reasoning)}</p>
    ${feats ? `<ul class="feats">${feats}</ul>` : ''}
    ${(j.check_by_eye || []).length ? `<div class="sub"><b>Check by eye:</b></div><ul class="feats">${j.check_by_eye.map(c => `<li>${cite(c)}</li>`).join('')}</ul>` : ''}${cites ? `<ul class="cites">${cites}</ul>` : '<div class="sub">No papers in the library yet.</div>'}`;
  reveal($('litBody'));
}

function renderFlags(flags) {
  if (!flags.length) return;
  const c = $('flagsCard'); c.classList.remove('hidden');
  c.innerHTML = `<h3>Trust flags</h3><ul>${flags.map(f => `<li>${esc(f)}</li>`).join('')}</ul>`;
  reveal(c.querySelector('ul'));
}

function renderVerdict(v) {
  const el = $('verdict'); el.classList.remove('muted');
  finishPanel();
  if (v.error) { el.innerHTML = '<span class="warn">Verdict unavailable.</span>'; return; }
  const cls = { agrees: 'good', 'partly agrees': 'warn', disagrees: 'bad' }[v.alignment] || '';
  el.innerHTML = `<div class="headline ${cls}">${esc(v.headline)}</div><p>${esc(v.conclusion)}</p>
    ${v.look_again_at ? `<p class="sub"><b>Look again at:</b> ${esc(v.look_again_at)}</p>` : ''}
    <p class="sub">${esc(v.certainty_note)}</p>`;
  el.querySelectorAll('p').forEach(p => reveal(p, 16, 1200));
}

window.addEventListener('resize', () => { if (img) { drawShot(); drawDim(selRect); drawHeat(); } });

// ---------- decision cut-off (gear in the panel header) ----------
const cutHint = s => { $('cutHint').innerHTML = `${s.label} · calls SSL above <b>${Math.round(s.cut * 100)}%</b> ⚙`; };
Cutoff.load().then(cutHint);
Cutoff.onChange(s => { cutHint(s); if (result?.judge_mode === 'panel') { renderPanelSummary(); drawHeat(); } });
const openCut = () => Cutoff.open($('panel').classList.contains('hidden') ? null : $('panel').getBoundingClientRect());
$('gear').onclick = openCut;
$('cutHint').onclick = openCut;
