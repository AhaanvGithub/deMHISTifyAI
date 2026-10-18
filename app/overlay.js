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
let result = null, parsed = null, heatRef = 'researcher';

window.demhist.onInit(({ screenshot, researcher }) => {
  researcherText = researcher;
  $('chip').textContent = `“${researcher}”`;
  img = new Image();
  img.onload = () => { scale = img.width / window.innerWidth; drawShot(); drawDim(); };
  img.src = screenshot;
});

function fit(c) { c.width = window.innerWidth; c.height = window.innerHeight; }
function drawShot() { fit(shot); shot.getContext('2d').drawImage(img, 0, 0, shot.width, shot.height); }
function drawDim(r) {
  fit(dim);
  const g = dim.getContext('2d');
  g.fillStyle = 'rgba(8,10,16,0.62)';
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
function setTissue(boxPx, label) {
  // box comes in image pixels of the selection crop
  tissueBox = { x: boxPx[0] / scale, y: boxPx[1] / scale, w: (boxPx[2] - boxPx[0]) / scale, h: (boxPx[3] - boxPx[1]) / scale };
  place(tissue, tissueBox); tissue.classList.remove('hidden');
  $('tissueTag').textContent = label;
}
async function startRegion() {
  state = 'region';
  $('hint').textContent = 'Adjust the tissue box if needed, then press Enter';
  $('sizeTag').classList.add('hidden');
  showActions();
  const blob = await selCrop();
  setTissue([0, 0, selRect.w * scale, selRect.h * scale], 'detecting tissue…');
  try {
    const g = await (await postImage('/region/guess', blob)).json();
    if (state === 'region') setTissue(g.box, 'quick guess · refining…');
    const r = await (await postImage('/region/refine', blob)).json();
    if (state === 'region') {
      if (r.box) { tissue.classList.add('anim'); setTissue(r.box, 'refined by vision model · drag to adjust'); setTimeout(() => tissue.classList.remove('anim'), 400); }
      else $('tissueTag').textContent = 'quick guess · drag to adjust';
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
  const dx = e.clientX - boxDrag.x, dy = e.clientY - boxDrag.y, s = boxDrag.start, b = { ...s };
  if (boxDrag.mode === 'move') { b.x = s.x + dx; b.y = s.y + dy; }
  if (boxDrag.mode.includes('w')) { b.x = s.x + dx; b.w = s.w - dx; }
  if (boxDrag.mode.includes('e')) { b.w = s.w + dx; }
  if (boxDrag.mode.includes('n')) { b.y = s.y + dy; b.h = s.h - dy; }
  if (boxDrag.mode.includes('s')) { b.h = s.h + dy; }
  b.x = Math.max(0, Math.min(b.x, selRect.w - 32)); b.y = Math.max(0, Math.min(b.y, selRect.h - 32));
  b.w = Math.max(32, Math.min(b.w, selRect.w - b.x)); b.h = Math.max(32, Math.min(b.h, selRect.h - b.y));
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
  tissue.classList.add('locked'); $('tissueTag').textContent = 'CONCH judges looking…';
  layoutPanel();
  const res = await postImage('/analyze', await tissueCrop(), { researcher: researcherText });
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
  const hc = $('heatControls');
  hc.style.left = (selRect.x + tissueBox.x) + 'px';
  hc.style.top = Math.min(selRect.y + selRect.h + 36, window.innerHeight - 52) + 'px';
}

function handle(ev, d) {
  if (ev === 'parsed') return renderYou(d);
  if (ev === 'conch') return renderConch(d);
  if (ev === 'judge') return renderJudge(d);
  if (ev === 'flags') return renderFlags(d);
  if (ev === 'verdict') return renderVerdict(d);
  if (ev === 'error') {
    $('panelSummary').innerHTML = `<span class="bad">${d.message === 'no_tissue' ? 'No tissue found in the box.' : 'Error: ' + d.message}</span>`;
    $('litSummary').textContent = '—'; $('verdict').textContent = '—';
  }
}

const pct = (label, n, cls = '') => `<span class="call ${cls}">${label} ${n}%</span>`;

function renderYou(p) {
  parsed = p;
  $('youCard').innerHTML = `<h3>You</h3>` + (p.call
    ? `<div class="summary">${pct(LABEL[p.call], p.certainty ?? '–', 'you')}<div class="sub">${FULL[p.call]}</div></div>`
    : `<div class="summary"><span class="warn">“${p.text}”: outside the validated scope</span></div>`);
}

function renderConch(r) {
  result = r;
  $('tissueTag').textContent = `${r.grid.cols} × ${r.grid.rows} tiles`;
  $('heatControls').classList.remove('hidden');
  drawHeat();
  const pnl = r.panel, agree = parsed?.call && pnl.top === parsed.call;
  $('panelSummary').classList.remove('muted');
  $('panelSummary').innerHTML =
    `${pct(LABEL[pnl.top], pnl.top_conf, agree || !parsed?.call ? 'good' : 'bad')}
     <div class="alt">- alt: ${LABEL[pnl.alt]} ${pnl.alt_conf}%</div>
     <div class="votes">${Object.entries(pnl.votes).map(([c, n]) => `${LABEL[c]} ${n}/8`).join(' · ')}</div>`;
  const box = $('judges'); box.innerHTML = '';
  r.judges.forEach((j, i) => setTimeout(() => {
    const el = document.createElement('div');
    el.className = 'judge pop';
    el.title = `${j.prompts.SSA}\nvs\n${j.prompts.HP}`;
    el.innerHTML = `<span class="jid">${j.id}</span><span class="cond">${j.condition === 'raw' ? 'raw' : 'norm'}</span>
                    <b class="${j.top === 'SSA' ? 'ssl' : 'hp'}">${LABEL[j.top]} ${j.top_conf}</b>
                    <span class="jalt">${LABEL[j.alt]} ${j.alt_conf}</span>`;
    box.appendChild(el);
  }, 120 * i));
}

function drawHeat() {
  if (!result) return;
  const ref = heatRef === 'ai' ? result.panel.top : (parsed?.call || result.panel.top);
  const rect = tissue.getBoundingClientRect();
  heat.width = rect.width; heat.height = rect.height;
  const g = heat.getContext('2d'), { rows, cols } = result.grid;
  const tw = heat.width / cols, th = heat.height / rows, a = $('opacity').value / 100;
  g.clearRect(0, 0, heat.width, heat.height);
  for (const t of result.tiles) {
    const x = t.col * tw, y = t.row * th;
    if (!t.is_tissue) { g.fillStyle = `rgba(120,120,130,${0.35 * a})`; g.fillRect(x, y, tw, th); continue; }
    const s = t.probs[ref] ?? 0.5, strength = Math.min(Math.abs(s - 0.5) * 2, 1);
    g.fillStyle = s >= 0.5 ? `rgba(52,211,153,${a * strength})` : `rgba(248,82,96,${a * strength})`;
    g.fillRect(x, y, tw, th);
  }
  g.strokeStyle = 'rgba(255,255,255,0.22)'; g.lineWidth = 1;
  for (let c = 1; c < cols; c++) { g.beginPath(); g.moveTo(c * tw, 0); g.lineTo(c * tw, heat.height); g.stroke(); }
  for (let r = 1; r < rows; r++) { g.beginPath(); g.moveTo(0, r * th); g.lineTo(heat.width, r * th); g.stroke(); }
  $('colLabels').innerHTML = Array.from({ length: cols }, (_, i) => `<span style="width:${tw}px">${i + 1}</span>`).join('');
  $('rowLabels').innerHTML = Array.from({ length: rows }, (_, i) => `<span style="height:${th}px">${String.fromCharCode(65 + (i % 26))}</span>`).join('');
}
$('opacity').addEventListener('input', drawHeat);
document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b));
  heatRef = b.dataset.ref; drawHeat();
});

function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function renderJudge(j) {
  const el = $('litSummary'); el.classList.remove('muted');
  if (j.error) { el.innerHTML = '<span class="warn">Literature judge unavailable (Lumen).</span>'; return; }
  const agree = !parsed?.call || j.call === parsed.call;
  el.innerHTML = `${pct(LABEL[j.call], j.consistency, agree ? 'good' : 'bad')}
    <div class="alt">- alt: ${LABEL[j.alt]} ${j.alt_consistency}%</div>
    <div class="sub">consistency over ${j.n_samples} runs · model's own estimate ${j.stated_confidence}%</div>`;
  const feats = (j.key_features || []).map(f =>
    `<li><b class="${f.points_to === 'SSA' ? 'ssl' : 'hp'}">${LABEL[f.points_to] || ''}</b> ${esc(f.feature)}: ${esc(f.why)} ${f.cites.map(c => `<sup>${c}</sup>`).join('')}</li>`).join('');
  const cites = (j.citations || []).map(c =>
    `<li><sup>${c.ref}</sup> ${esc(c.title)}${c.year ? ` (${c.year})` : ''}${c.doi ? ` · <span class="doi">${esc(c.doi)}</span>` : ''}</li>`).join('');
  $('litBody').innerHTML = `<p class="reason">${esc(j.reasoning).replace(/\[(P\d+)\]/g, '<sup>$1</sup>')}</p>
    ${feats ? `<ul class="feats">${feats}</ul>` : ''}${cites ? `<ul class="cites">${cites}</ul>` : '<div class="sub">No papers in the library yet.</div>'}`;
}

function renderFlags(flags) {
  if (!flags.length) return;
  const c = $('flagsCard'); c.classList.remove('hidden');
  c.innerHTML = `<h3>Trust flags</h3><ul>${flags.map(f => `<li>${esc(f)}</li>`).join('')}</ul>`;
}

function renderVerdict(v) {
  const el = $('verdict'); el.classList.remove('muted');
  if (v.error) { el.innerHTML = '<span class="warn">Verdict unavailable.</span>'; return; }
  const cls = { agrees: 'good', 'partly agrees': 'warn', disagrees: 'bad' }[v.alignment] || '';
  el.innerHTML = `<div class="headline ${cls}">${esc(v.headline)}</div><p>${esc(v.conclusion)}</p>
    ${v.look_again_at ? `<p class="sub"><b>Look again at:</b> ${esc(v.look_again_at)}</p>` : ''}
    <p class="sub">${esc(v.certainty_note)}</p>`;
}

window.addEventListener('resize', () => { if (img) { drawShot(); drawDim(selRect); drawHeat(); } });
