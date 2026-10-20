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
let screenScale = null;     // the viewer's scale bar, read on the whole screen before the snip (screenshot px)

window.demhist.onInit(({ screenshot, researcher }) => {
  researcherText = researcher;
  $('chip').textContent = `“${researcher}”`;
  img = new Image();
  img.onload = () => { scale = img.width / window.innerWidth; drawShot(); drawDim(); readScreenScale(); };
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
// The backend reads the viewer's scale bar on the whole screenshot as soon as it is taken. Once it's in, one 448 um
// field has a known size on screen: a dashed square of that size follows the mouse, a click puts the field there and a
// drag smaller than a field grows to one around it (a larger drag still lets the field find the tissue inside it).
async function readScreenScale() {
  try {
    const s = await (await fetch(API + '/scale', { signal: AbortSignal.timeout(25000) })).json();
    if (state !== 'select') return;  // already snipped: /region/guess uses the same reading
    if (!s.found) { $('hint').innerHTML = 'No scale bar found on screen · drag to select the slide region · <kbd>Esc</kbd> to cancel'; return; }
    screenScale = s;
    const [x0, y0, x1, y1] = s.bar.map(v => v / scale), pad = 4;
    place($('scaleMark'), { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad });
    $('scaleMarkTag').textContent = `${s.label_um} µm scale bar read`;
    $('scaleMark').classList.remove('hidden');
    const advice = s.field_px > Math.min(img.width, img.height) ? ' · zoom out: one field is bigger than the screen'
      : s.field_px < 224 ? ' · zoom in: one field is under 224 screen pixels' : '';
    $('hint').innerHTML = `One 448 µm field is ${Math.round(s.field_px)} px here${advice} · click or drag where the field should go · <kbd>Esc</kbd> to cancel`;
    if (lastMouse) previewAt(lastMouse.x, lastMouse.y);
  } catch { /* backend not up: plain drag-to-select */ }
}
const fieldSide = () => Math.min(screenScale.field_px / scale, window.innerWidth, window.innerHeight);  // CSS px
function fieldAt(cx, cy) {  // one field-sized square centred on a point, kept on screen
  const side = fieldSide();
  return { x: Math.min(Math.max(0, cx - side / 2), window.innerWidth - side),
           y: Math.min(Math.max(0, cy - side / 2), window.innerHeight - side), w: side, h: side };
}
function previewAt(cx, cy) { place($('fieldPreview'), fieldAt(cx, cy)); $('fieldPreview').classList.remove('hidden'); }
let drag = null, lastMouse = null;
dim.addEventListener('mousedown', e => {
  if (state !== 'select') return;
  drag = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mousemove', e => {
  lastMouse = { x: e.clientX, y: e.clientY };
  if (state === 'select' && screenScale) {
    const r = drag ? norm(drag.x, drag.y, e.clientX, e.clientY) : null;
    r ? previewAt(r.x + r.w / 2, r.y + r.h / 2) : previewAt(e.clientX, e.clientY);
  }
  if (state === 'select' && drag) {
    selRect = norm(drag.x, drag.y, e.clientX, e.clientY);
    drawDim(selRect); place(sel, selRect); sel.classList.remove('hidden');
    const tag = $('sizeTag');
    tag.classList.remove('hidden');
    tag.textContent = `${Math.round(selRect.w * scale)} × ${Math.round(selRect.h * scale)}` +
      (screenScale ? ` · one field ${Math.round(screenScale.field_px)} px` : '');
    tag.style.left = selRect.x + 'px'; tag.style.top = (selRect.y + selRect.h + 8) + 'px';
  }
  if (boxDrag) moveBox(e);
  if (panDrag) pan(e);
});
window.addEventListener('mouseup', () => {
  if (state === 'select' && drag) {
    const d = drag, big = selRect && selRect.w > 40 && selRect.h > 40;
    drag = null;
    if (screenScale) {
      // a click: one field centred there; a drag: at least one field, grown around the drag's centre
      const f = big ? fieldAt(selRect.x + selRect.w / 2, selRect.y + selRect.h / 2) : fieldAt(d.x, d.y);
      selRect = big ? union(selRect, f) : f;
      drawDim(selRect); place(sel, selRect); sel.classList.remove('hidden');
      $('fieldPreview').classList.add('hidden'); $('scaleMark').classList.add('hidden');
      startRegion();
    } else if (big) startRegion();
  }
  boxDrag = null;
  endPan();
});
function union(a, b) {
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
}
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
  placeControls();
}
// map + inspection controls under the field, as wide as the gap beside the panel (wrapping if narrower); above the
// field when there's no room below, so they never sit on the tissue or under the panel
function placeControls() {
  const hc = $('heatControls'), left = selRect.x + tissueBox.x, p = $('panel').getBoundingClientRect();
  hc.style.left = left + 'px';
  hc.style.maxWidth = ((p.width && p.left > left ? p.left : window.innerWidth) - left - 12) + 'px';
  const h = hc.offsetHeight, below = selRect.y + selRect.h + 36, above = selRect.y + tissueBox.y - 34 - h;
  hc.style.top = (below + h <= window.innerHeight - 8 ? below : above >= 8 ? above : window.innerHeight - h - 8) + 'px';
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
    ? `<div class="summary">${pct(LABEL[p.call], p.certainty ?? '–', clsOf(p.call))}<div class="sub">${FULL[p.call]}</div></div>`
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
  placeControls();
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
    `${pct(LABEL[pnl.top], pnl.top_conf, clsOf(pnl.top))}${mine(agree)}
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

// the app's only meaning-carrying colours: HP teal, SSL orange (styles.css --hp / --ssl)
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const rgbOf = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
const CLASS_RGB = () => ({ SSA: rgbOf(cssVar('--ssl')), HP: rgbOf(cssVar('--hp')) });
const classCss = c => (c === 'SSA' ? 'var(--ssl)' : 'var(--hp)');
const clsOf = c => (c === 'SSA' ? 'ssl' : 'hp');
// agreement with the researcher's own call: white text and a symbol, not a colour
const mine = agree => (parsed?.call ? `<div class="mine">${agree ? '\u2713 matches your call' : '\u2717 differs from your call'}</div>` : '');
const smooth = (e0, e1, x) => { const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); };
const refClass = () => heatRef === 'ai' ? 'SSA' : (parsed?.call || result.panel.top);

function drawHeat() {
  if (!result) return;
  const ref = refClass(), H = result.heat, dpr = window.devicePixelRatio || 1;
  const rect = tissue.getBoundingClientRect();
  for (const c of [heat, $('feat')]) { c.width = Math.round(rect.width * dpr); c.height = Math.round(rect.height * dpr); }
  const g = heat.getContext('2d'), a = $('opacity').value / 100;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, heat.width, heat.height);
  const z = zoomTo(g, heat);  // zoomed in: draw only the visible part of the field

  // 1. probability of the reference call per fine cell; low values fade to transparent
  const ai = heatRef === 'ai';
  const rgb = CLASS_RGB();
  const ev = H.kind === 'evidence';  // cover-up map: 0.5 = no effect, above = pushed toward SSL, below = toward HP
  const c0 = result.judge_mode === 'panel' && !ev ? Cutoff.state().cut : 0.5;
  // the soft map only changes with the map, opacity, view and cut-off: rebuilt then, reused while zooming
  const key = [a, heatRef, ref, c0].join('|');
  if (heatBlob.H !== H || heatBlob.key !== key) {
    const small = document.createElement('canvas');
    small.width = H.cols; small.height = H.rows;
    const sg = small.getContext('2d'), im = sg.createImageData(H.cols, H.rows);
    const centre = p => (p <= c0 ? 0.5 * p / c0 : 0.5 + 0.5 * (p - c0) / (1 - c0));  // cut-off -> middle of the scale
    // A whole-snip field gives (nearly) one value everywhere: a uniform tint says nothing about *where* and only
    // washes out the tissue, so a flat map is not painted (the call and colour bar still show the value).
    const tv = H.p_ssa.filter((_, i) => H.tissue[i] >= 0.25);
    const flat = tv.length && Math.max(...tv) - Math.min(...tv) < 0.03;
    for (let i = 0; i < H.p_ssa.length; i++) {
      const p = ai ? centre(H.p_ssa[i]) : H.p_ssa[i];
      // AI view: fixed P(SSL) scale, HP-leaning tissue teal, SSL-leaning orange, undecided (~0.5) fades out.
      // Your-call view: support for your call, in that call's colour, clear when unlikely.
      const v = ai ? p : (ref === 'SSA' ? p : 1 - p);
      // soft fade from 'no effect' (transparent) to strong evidence, so empty cells don't cut hard edges
      const k = ai ? smooth(0.0, 0.4, Math.abs(p - 0.5)) : ev ? smooth(0.0, 0.4, v - 0.5) : smooth(0.22, 0.6, v);
      const alpha = H.tissue[i] < 0.25 || flat ? 0 : (ai ? 0.55 : 0.8) * k;  // never opaque: tissue must stay readable
      im.data.set([...(ai ? (p >= 0.5 ? rgb.SSA : rgb.HP) : rgb[ref]), Math.round(255 * a * alpha)], i * 4);
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
    heatBlob = { H, key, mid };
  }
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(heatBlob.mid, 0, 0, heat.width, heat.height);

  // 2. optional grid: faint sub-cells, stronger lines per judge tile
  const show = $('showGrid').checked;
  if (show) {
    const cw = heat.width / H.cols, ch = heat.height / H.rows;
    const line = (x0, y0, x1, y1) => { g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); };
    g.lineWidth = 1 / z; g.strokeStyle = 'rgba(255,255,255,0.06)';
    // fine lines no closer than ~12 device px, so a 64-cell map doesn't turn into a grey mesh
    const every = Math.max(1, Math.ceil(12 * dpr / (z * Math.min(cw, ch))));
    for (let c = every; c < H.cols; c += every) if (c % H.sub) line(Math.round(c * cw) + .5, 0, Math.round(c * cw) + .5, heat.height);
    for (let r = every; r < H.rows; r += every) if (r % H.sub) line(0, Math.round(r * ch) + .5, heat.width, Math.round(r * ch) + .5);
    g.strokeStyle = 'rgba(255,255,255,0.28)'; g.lineWidth = Math.max(1, dpr) / z;
    for (let c = H.sub; c < H.cols; c += H.sub) line(Math.round(c * cw), 0, Math.round(c * cw), heat.height);
    for (let r = H.sub; r < H.rows; r += H.sub) line(0, Math.round(r * ch), heat.width, Math.round(r * ch));
  }
  g.setTransform(1, 0, 0, 1, 0, 0);
  const tw = rect.width / result.grid.cols, th = rect.height / result.grid.rows;
  const axes = show && z === 1;  // tile letters/numbers sit outside the box: only when the whole field is shown
  $('colLabels').innerHTML = axes ? Array.from({ length: result.grid.cols }, (_, i) => `<span style="width:${tw}px">${i + 1}</span>`).join('') : '';
  $('rowLabels').innerHTML = axes ? Array.from({ length: result.grid.rows }, (_, i) => `<span style="height:${th}px">${String.fromCharCode(65 + (i % 26))}</span>`).join('') : '';

  $('colorbar').classList.remove('hidden');
  $('colorbar').classList.toggle('diverging', ai);
  $('colorbar').style.setProperty('--cb-col', classCss(ref));
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
// literature-judge feature outlines in the colour of the class they point to (white = either)
const featColor = f => (f.points_to === 'SSA' ? cssVar('--ssl') : f.points_to === 'HP' ? cssVar('--hp') : '#ffffff');
let keyFeatures = [];
let heatBlob = {};  // the soft heatmap image, kept while only the zoom changes
// ---------- measured crypts (backend/glands.py): outlines follow real glands, labels come from measurements ----------
// dilated crypts lean SSL, round regular crypts lean HP: drawn in those classes' colours
const GLAND_COLORS = new Proxy({}, { get: (_, k) => ({ 'dilated crypt': cssVar('--ssl'), 'round regular crypt': cssVar('--hp') })[k] });
function drawGlands(fc, g) {
  const W = fc.width, Hh = fc.height, dpr = window.devicePixelRatio || 1, rect = tissue.getBoundingClientRect();
  const placed = [], z = zoomTo(g, fc);
  const free = (x, y, w, h) => !placed.some(b => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
  // strongest evidence first, so the glands that drove the call get their labels before any collision
  [...result.glands].sort((a, b) => !!b.label - !!a.label || Math.abs(b.evidence) - Math.abs(a.evidence)).forEach(gl => {
    const col = GLAND_COLORS[gl.label];
    g.beginPath();
    gl.outline.forEach(([x, y], i) => (i ? g.lineTo(x * W, y * Hh) : g.moveTo(x * W, y * Hh)));
    g.closePath();
    if (!col) {  // a crypt with nothing clear to say: a faint outline only, no label
      g.lineJoin = 'round'; g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 1.2 * dpr / z;
      g.stroke();
      return;
    }
    g.lineJoin = 'round'; g.shadowColor = 'rgba(0,0,0,.55)'; g.shadowBlur = 3 * dpr;
    g.strokeStyle = col; g.lineWidth = 2 * dpr / z;
    g.stroke(); g.shadowBlur = 0;
    if (!inView(toView(...gl.center))) return;  // zoomed in: no label for a crypt whose centre is out of view
    // label just above the gland (or below if that spot is taken); skipped if both collide - hover still shows it
    const top = gl.outline.reduce((m, p) => (p[1] < m[1] ? p : m)), bot = gl.outline.reduce((m, p) => (p[1] > m[1] ? p : m));
    const text = (gl.tags && gl.tags.length > 1) ? gl.tags.join(' · ') : gl.label, w = text.length * 6.2 + 18, h = 20;
    const cx = toView(...gl.center)[0] * rect.width, ty = toView(0, top[1])[1] * rect.height, by = toView(0, bot[1])[1] * rect.height;
    for (const [px, py] of [[cx, ty - h - 2], [cx, by + 2]]) {
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
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, fc.width, fc.height);
  document.querySelectorAll('.feat-label, .gland-label').forEach(e => e.remove());
  if (result?.glands && $('showFeat').checked) return drawGlands(fc, g);
  if (!result || !$('showFeat').checked || !keyFeatures.length) return;
  const H = result.heat, cw = fc.width / H.cols, ch = fc.height / H.rows, dpr = window.devicePixelRatio || 1;
  const z = zoomTo(g, fc);
  keyFeatures.forEach((f, k) => {
    const m = H.features && H.features[f.feature_id];
    if (!m) return;
    const vals = m.filter((_, i) => H.tissue[i] >= 0.25).sort((x, y) => x - y);
    if (!vals.length) return;
    const thr = Math.max(0.7, vals[Math.floor(vals.length * 0.9)]);  // strongest ~10% of tissue, and clearly above typical
    const on = (r, c) => r >= 0 && c >= 0 && r < H.rows && c < H.cols && m[r * H.cols + c] >= thr && H.tissue[r * H.cols + c] >= 0.25;
    g.strokeStyle = featColor(f); g.lineWidth = 2 * dpr / z; g.lineJoin = g.lineCap = 'round';
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
    const at = best && toView((best.c + 0.5) / H.cols, (best.r + 0.5) / H.rows);
    if (at && inView(at)) {
      const lab = document.createElement('div');
      lab.className = 'feat-label';
      lab.style.borderColor = featColor(f);
      lab.style.left = (at[0] * fc.width / dpr) + 'px';
      lab.style.top = (at[1] * fc.height / dpr) + 'px';
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
  if (panDrag) return tip.classList.add('hidden');
  const r = tissue.getBoundingClientRect(), H = result.heat;
  const [fx, fy] = fromView((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);  // 0..1 of the field
  const fc = Math.max(0, Math.min(H.cols - 1, Math.floor(fx * H.cols)));
  const fr = Math.max(0, Math.min(H.rows - 1, Math.floor(fy * H.rows)));
  const ref = heatRef === 'ai' ? 'SSA' : (parsed?.call || result.panel.top);  // AI view always reads P(SSL)
  const p = H.p_ssa[fr * H.cols + fc], s = ref === 'SSA' ? p : 1 - p;
  const feats = Object.entries(H.features || {}).sort((x, y) => y[1][fr * H.cols + fc] - x[1][fr * H.cols + fc]).slice(0, 2).map(([k]) => k.replace(/_/g, ' '));
  const gl = glandAt(fx, fy);
  const glandLine = gl ? `<div><b style="color:${GLAND_COLORS[gl.label] || 'var(--text)'}">${gl.label || 'crypt'}</b>${gl.crypt_um != null ? ` · ${gl.crypt_um} µm across` : ''}` +
    `</div>` : '';
  const tile = `${String.fromCharCode(65 + Math.floor(fr / H.sub))}${Math.floor(fc / H.sub) + 1}`;
  const subc = `${(fr % H.sub) + 1}-${(fc % H.sub) + 1}`;  // row-column inside the field
  tip.innerHTML = H.tissue[fr * H.cols + fc] < 0.25 ? `${tile}·${subc} <span class="muted">no tissue</span>`
    : `${tile}·${subc} ${H.kind === 'evidence'
        ? `<b style="color:${Math.abs(p - 0.5) < 0.06 ? 'var(--text)' : classCss(p > 0.5 ? 'SSA' : 'HP')}">${Math.abs(p - 0.5) < 0.06 ? 'little effect on the call' : `pushes toward ${p > 0.5 ? 'SSL' : 'HP'}`}</b>`
        : `<b style="color:${classCss(ref)}">P(${LABEL[ref]}) ${Math.round(s * 100)}%</b>`}${result.glands ? glandLine : `<div class="muted">strongest here: ${feats.join(', ')}</div>`}`;
  tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  tip.classList.remove('hidden');
});
tissue.addEventListener('mouseleave', () => tip.classList.add('hidden'));
$('opacity').addEventListener('input', drawHeat);
document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b));
  heatRef = b.dataset.ref; drawHeat();
});

// ---------- 4. inspect (once analysis starts): zoom, colour views, hold Space for the bare field ----------
// None of this changes the call: it only changes how the field is shown. The view is drawn from the full-resolution
// screenshot, so zooming shows every pixel the slide viewer drew (no more than that).
const view = $('view'), MAX_ZOOM = 8;
const V = { x: 0, y: 0, w: 1, h: 1 };  // visible part of the field, in 0..1 of the tissue box
const toView = (x, y) => [(x - V.x) / V.w, (y - V.y) / V.h];
const fromView = (u, v) => [V.x + u * V.w, V.y + v * V.h];
const inView = ([u, v]) => u >= 0 && u <= 1 && v >= 0 && v <= 1;
// draw on canvas c as if it showed the whole field: returns the zoom, to keep line widths constant on screen
function zoomTo(g, c) {
  g.setTransform(1 / V.w, 0, 0, 1 / V.h, -V.x * c.width / V.w, -V.y * c.height / V.h);
  return 1 / V.w;
}
const looks = new Set();
const KEY_LOOKS = { e: 'exposure', c: 'contrast', g: 'greyscale', s: 'saturation', d: 'sharpen', n: 'nuclei' };
const cssLooks = () => [looks.has('exposure') && 'brightness(1.3)', looks.has('contrast') && 'contrast(1.55)',
  looks.has('saturation') && 'saturate(1.9)', looks.has('greyscale') && 'grayscale(1)'].filter(Boolean).join(' ') || 'none';

function renderView() {
  if (state !== 'results' || !tissueBox) return;
  const plain = V.w === 1 && !looks.size;
  view.classList.toggle('hidden', plain);  // nothing changed: the screenshot underneath already shows the field
  if (plain) return;
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(view.clientWidth * dpr), H = Math.round(view.clientHeight * dpr);
  if (view.width !== W || view.height !== H) { view.width = W; view.height = H; }
  // the analysed crop (the tissue box, in screenshot px), and the visible part of it
  const bx = (selRect.x + tissueBox.x) * scale, by = (selRect.y + tissueBox.y) * scale, bw = tissueBox.w * scale, bh = tissueBox.h * scale;
  const base = document.createElement('canvas');
  base.width = W; base.height = H;
  const bg = base.getContext('2d');
  bg.imageSmoothingQuality = 'high';
  bg.drawImage(img, bx + V.x * bw, by + V.y * bh, V.w * bw, V.h * bh, 0, 0, W, H);
  if (looks.has('nuclei')) nucleiOnly(bg, W, H);
  const g = view.getContext('2d');
  g.clearRect(0, 0, W, H);
  g.filter = cssLooks();
  g.drawImage(base, 0, 0);
  g.filter = 'none';
  if (looks.has('sharpen')) sharpen(g, W, H);
}

// haematoxylin only (nuclei): colour deconvolution with the standard H&E stain vectors (Ruifrok & Johnston 2001),
// shown back in haematoxylin's own blue-purple on white
const STAINS = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]].map(v => { const n = Math.hypot(...v); return v.map(x => x / n); });
const UNMIX = (m => {  // inverse of the 3x3 stain matrix (rows = stains): optical density -> stain amounts
  const [[a, b, c], [d, e, f], [g, h, i]] = m, A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A, c * h - b * i, b * f - c * e], [B, a * i - c * g, c * d - a * f], [C, b * g - a * h, a * e - b * d]].map(r => r.map(x => x / det));
})(STAINS);
const OD = Float32Array.from({ length: 256 }, (_, v) => -Math.log((v + 1) / 256));  // optical density per 8-bit value
const H_OUT = [0, 1, 2].map(k => Uint8ClampedArray.from({ length: 1024 }, (_, i) => 255 * Math.exp(-(i / 256) * STAINS[0][k])));
function nucleiOnly(g, W, H) {
  const im = g.getImageData(0, 0, W, H), d = im.data, [u0, u1, u2] = [UNMIX[0][0], UNMIX[1][0], UNMIX[2][0]];
  const [r, gr, b] = H_OUT;
  for (let p = 0; p < d.length; p += 4) {
    // haematoxylin amount (0..4 optical density), looked up back to colour in 1/256 steps
    const i = Math.min(1023, Math.max(0, Math.round(256 * (OD[d[p]] * u0 + OD[d[p + 1]] * u1 + OD[d[p + 2]] * u2))));
    d[p] = r[i]; d[p + 1] = gr[i]; d[p + 2] = b[i];
  }
  g.putImageData(im, 0, 0);
}
function sharpen(g, W, H) {  // 3x3 unsharp kernel: centre 5, edge neighbours -1
  const im = g.getImageData(0, 0, W, H), src = new Uint8ClampedArray(im.data), d = im.data, row = W * 4;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const p = y * row + x * 4;
    for (let k = 0; k < 3; k++) d[p + k] = 5 * src[p + k] - src[p + k - 4] - src[p + k + 4] - src[p + k - row] - src[p + k + row];
  }
  g.putImageData(im, 0, 0);
}

let redrawQueued = false;
function redraw() {  // one repaint per frame while scrolling or dragging
  if (redrawQueued) return;
  redrawQueued = true;
  requestAnimationFrame(() => {
    redrawQueued = false;
    renderView(); drawHeat();
    $('zoomTag').textContent = `${(1 / V.w).toFixed(1)}×`;
    tissue.classList.toggle('zoomed', V.w < 1);
  });
}
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
function resetZoom() { Object.assign(V, { x: 0, y: 0, w: 1, h: 1 }); redraw(); }
tissue.addEventListener('wheel', e => {
  if (state !== 'results') return;
  e.preventDefault();
  const r = tissue.getBoundingClientRect(), u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
  const [px, py] = fromView(u, v);  // the point under the mouse stays put
  const w = clamp(V.w * Math.pow(1.0025, e.deltaY), 1 / MAX_ZOOM, 1);  // ~1.28x per wheel notch
  Object.assign(V, { w, h: w, x: clamp(px - u * w, 0, 1 - w), y: clamp(py - v * w, 0, 1 - w) });
  redraw();
}, { passive: false });
let panDrag = null;
tissue.addEventListener('mousedown', e => {
  if (state !== 'results' || V.w === 1) return;
  panDrag = { x: e.clientX, y: e.clientY, vx: V.x, vy: V.y };
  tissue.classList.add('panning');
});
function pan(e) {
  const r = tissue.getBoundingClientRect();
  V.x = clamp(panDrag.vx - (e.clientX - panDrag.x) / r.width * V.w, 0, 1 - V.w);
  V.y = clamp(panDrag.vy - (e.clientY - panDrag.y) / r.height * V.h, 0, 1 - V.h);
  redraw();
}
function endPan() { panDrag = null; tissue.classList.remove('panning'); }
tissue.addEventListener('dblclick', () => state === 'results' && resetZoom());

function toggleLook(name) {
  looks.has(name) ? looks.delete(name) : looks.add(name);
  document.querySelectorAll('#looks button').forEach(b => b.classList.toggle('on', looks.has(b.dataset.look)));
  redraw();
}
document.querySelectorAll('#looks button').forEach(b => b.onclick = () => toggleLook(b.dataset.look));
const setBare = on => document.body.classList.toggle('bare', on);  // hold Space: the field without anything the AI drew
document.addEventListener('keydown', e => {
  if (state !== 'results' || e.ctrlKey || e.altKey || e.metaKey || e.target.matches?.('input[type=text], input[type=number], textarea')) return;
  if (e.key === ' ') { e.preventDefault(); document.activeElement?.blur?.(); setBare(true); return; }
  const look = KEY_LOOKS[e.key.toLowerCase()];
  if (look && !e.repeat) toggleLook(look);
  if (e.key === '0') resetZoom();
});
document.addEventListener('keyup', e => { if (e.key === ' ') { e.preventDefault(); setBare(false); } });
window.addEventListener('blur', () => setBare(false));  // never stuck hidden after Alt+Tab with Space down

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
  el.innerHTML = `${pct(LABEL[j.call], j.consistency, clsOf(j.call))}${mine(agree)}
    <div class="alt">- alt: ${LABEL[j.alt]} ${j.alt_consistency}%</div>
    <div class="sub">consistency over ${j.n_samples} runs · model's own estimate ${j.stated_confidence}%${j.agrees_with_panel === false ? ' · <span class="warn">disagrees with the image panel</span>' : j.agrees_with_panel ? ' · agrees with the image panel' : ''}</div>`;
  const feats = (j.key_features || []).map(f =>
    `<li>${LABEL[f.points_to] ? `<b class="${f.points_to === 'SSA' ? 'ssl' : 'hp'}">${LABEL[f.points_to]}</b> ` : '<b class="muted">either</b> '}${esc(f.feature)}${f.strength ? ` <span class="strength">${String(f.feature_id || '').startsWith('crypts:') ? 'measured' : 'text match'} · ${esc(f.strength)}</span>` : ''}: ${cite(f.why)} ${(f.cites || []).filter(c => !(f.why || '').includes(c)).map(c => `<sup>${c}</sup>`).join('')}</li>`).join('');  // citations already in the sentence are not repeated
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
  const mark = { agrees: '\u2713 ', 'partly agrees': '\u2248 ', disagrees: '\u2717 ' }[v.alignment] || '';
  el.innerHTML = `<div class="headline">${mark}${esc(v.headline)}</div><p>${esc(v.conclusion)}</p>
    ${v.look_again_at ? `<p class="sub"><b>Look again at:</b> ${esc(v.look_again_at)}</p>` : ''}
    <p class="sub">${esc(v.certainty_note)}</p>`;
  el.querySelectorAll('p').forEach(p => reveal(p, 16, 1200));
}

window.addEventListener('resize', () => { if (img) { drawShot(); drawDim(selRect); drawHeat(); renderView(); if (state === 'results') placeControls(); } });

// ---------- decision cut-off (gear in the panel header) ----------
const cutHint = s => { $('cutHint').innerHTML = `${s.label} · calls SSL above <b>${Math.round(s.cut * 100)}%</b> ⚙`; };
Cutoff.load().then(cutHint);
Cutoff.onChange(s => { cutHint(s); if (result?.judge_mode === 'panel') { renderPanelSummary(); drawHeat(); } });
const openCut = () => Cutoff.open($('panel').classList.contains('hidden') ? null : $('panel').getBoundingClientRect());
$('gear').onclick = openCut;
$('cutHint').onclick = openCut;
