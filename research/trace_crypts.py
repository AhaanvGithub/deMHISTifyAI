#!/usr/bin/env python3
"""Crypt tracing tool: outline every crypt in the ground-truth MHIST fields (research/crypt_gt.py) and tag it.

    .venv/Scripts/python research/trace_crypts.py      ->  open http://127.0.0.1:8777

Local only (binds 127.0.0.1). Every change is saved straight to data/crypt_traces/traces.json; a timestamped
copy of the previous file is kept in data/crypt_traces/backup/. The vote counts are not shown while tracing.
"""

import json
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypt_gt as GT  # noqa: E402

PORT = 8777
FIELDS = GT.fields()
NAMES = list(FIELDS.sample(frac=1, random_state=2)["image"])   # shuffled so vote count can't be guessed from order

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Crypt tracing</title><style>
:root{--bg:#f6f4f1;--fg:#222;--mut:#666;--card:#fff;--line:#ddd}
body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
#wrap{display:flex;gap:18px;padding:14px;align-items:flex-start}
#side{width:330px;flex:none}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:10px}
canvas{background:#fff;border:1px solid var(--line);cursor:crosshair;display:block}
button{font:inherit;padding:4px 10px;margin:2px;border:1px solid #bbb;border-radius:5px;background:#fff;cursor:pointer}
button.on{background:#333;color:#fff}
kbd{border:1px solid #bbb;border-bottom-width:2px;border-radius:3px;padding:0 4px;font-size:12px;background:#fff}
#list div{padding:2px 4px;cursor:pointer;border-radius:4px}#list div.cur{background:#e6e0d8}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
small{color:var(--mut)}h3{margin:4px 0 8px}ul{margin:4px 0;padding-left:18px}
</style></head><body><div id="wrap">
<div><div style="margin-bottom:6px"><b id="title"></b> <small id="pos"></small> <span id="saved" style="color:#2a7"></span></div>
<canvas id="cv"></canvas>
<div style="margin-top:6px"><button onclick="go(-1)">&larr; prev</button><button onclick="go(1)">next &rarr;</button>
<label><input type="checkbox" id="done" onchange="setDone()"> all crypts in this field traced</label>
<label style="margin-left:12px">zoom <select id="zoom" onchange="setZoom()"><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<label style="margin-left:12px"><input type="checkbox" id="hide" onchange="draw()"> hide outlines <kbd>H</kbd></label></div>
<div style="margin-top:6px">note: <input id="note" style="width:520px" onchange="setNote()"></div></div>
<div id="side">
<div class="card"><h3>How to trace</h3><ul>
<li><b>Drag</b> around a crypt to outline it (release to close). Trace the <b>outer edge of the gland</b>: just outside its band of epithelial nuclei, where stroma starts.</li>
<li>Trace <b>every</b> crypt you can recognise, including ones cut by the image edge (run the line along the edge) and longitudinal ones.</li>
<li>Click inside an outline to select it. Tags toggle on the selected crypt: <kbd>1</kbd> dilated <kbd>2</kbd> serrated <kbd>3</kbd> round regular <kbd>0</kbd> clear tags. A crypt can have several tags; no tag = a crypt with nothing clear.</li>
<li><kbd>Del</kbd> delete selected, <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> field, <kbd>D</kbd> mark field done.</li>
<li>Serrated = saw-tooth / star-shaped luminal border you can actually see at this resolution. If you can't tell, don't tag it.</li></ul></div>
<div class="card"><b>Selected:</b> <span id="sel">none</span><br>
<button id="t1" onclick="tog('dilated')">1 dilated</button><button id="t2" onclick="tog('serrated')">2 serrated</button><button id="t3" onclick="tog('round regular')">3 round regular</button><br>
<small><span class="sw" style="background:#d33"></span>dilated <span class="sw" style="background:#e90"></span>serrated <span class="sw" style="background:#27c"></span>round <span class="sw" style="background:#2a2"></span>no tag</small></div>
<div class="card"><b>Fields</b> <small id="prog"></small><div id="list" style="max-height:330px;overflow:auto;margin-top:4px"></div></div>
</div></div>
<script>
let names=[],T={},cur=0,img=new Image(),Z=4,sel=-1,drawing=null,hist=[];
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const COL={'dilated':'#d33','serrated':'#e90','round regular':'#27c','none':'#2a2'};
function F(){const n=names[cur];if(!T[n])T[n]={crypts:[],done:false,note:''};return T[n]}
function main(t){for(const k of ['dilated','serrated','round regular'])if(t.includes(k))return k;return 'none'}
async function init(){const zs=document.getElementById('zoom');zs.value=String(Math.max(2,Math.min(5,Math.floor((innerHeight-110)/224))));names=await (await fetch('/names')).json();T=await (await fetch('/traces')).json();load()}
function load(){sel=-1;hist=[];img.onload=()=>{setZoom()};img.src='/img/'+names[cur];
 document.getElementById('title').textContent=names[cur];document.getElementById('pos').textContent=(cur+1)+' / '+names.length;
 document.getElementById('done').checked=F().done;document.getElementById('note').value=F().note||'';list()}
function setZoom(){Z=+document.getElementById('zoom').value;cv.width=img.width*Z;cv.height=img.height*Z;draw()}
function list(){const L=document.getElementById('list');L.innerHTML='';let d=0;
 names.forEach((n,i)=>{const t=T[n];if(t&&t.done)d++;const e=document.createElement('div');e.className=i==cur?'cur':'';
 e.textContent=(t&&t.done?'✓ ':'   ')+n+'  ('+(t?t.crypts.length:0)+')';e.onclick=()=>{cur=i;load()};L.appendChild(e)});
 document.getElementById('prog').textContent=d+' of '+names.length+' done'}
function draw(){ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(img,0,0,cv.width,cv.height);
 const W=cv.width,H=cv.height;
 if(!document.getElementById('hide').checked)F().crypts.forEach((c,i)=>{ctx.beginPath();c.outline.forEach((p,k)=>k?ctx.lineTo(p[0]*W,p[1]*H):ctx.moveTo(p[0]*W,p[1]*H));ctx.closePath();
  ctx.strokeStyle=COL[main(c.tags)];ctx.lineWidth=i==sel?3.5:2;ctx.setLineDash(c.tags.length>1?[8,4]:[]);ctx.stroke();ctx.setLineDash([]);
  if(i==sel){ctx.fillStyle='rgba(255,255,255,.15)';ctx.fill()}});
 if(drawing&&drawing.length){ctx.beginPath();drawing.forEach((p,k)=>k?ctx.lineTo(p[0]*W,p[1]*H):ctx.moveTo(p[0]*W,p[1]*H));ctx.strokeStyle='#000';ctx.lineWidth=1.5;ctx.stroke()}
 const s=sel>=0?F().crypts[sel]:null;document.getElementById('sel').textContent=s?('#'+(sel+1)+': '+(s.tags.join(', ')||'no tag')):'none';
 [['t1','dilated'],['t2','serrated'],['t3','round regular']].forEach(([id,t])=>document.getElementById(id).className=s&&s.tags.includes(t)?'on':'')}
function pt(e){const r=cv.getBoundingClientRect();return [Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))]}
function inside(p,poly){let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];
 if(((a[1]>p[1])!=(b[1]>p[1]))&&(p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]))c=!c}return c}
let down=null;
cv.onmousedown=e=>{down=pt(e);drawing=[down]};
cv.onmousemove=e=>{if(!drawing)return;const p=pt(e),q=drawing[drawing.length-1];if(Math.hypot((p[0]-q[0])*cv.width,(p[1]-q[1])*cv.height)>=3){drawing.push(p);draw()}};
window.onmouseup=e=>{if(!drawing)return;const d=drawing;drawing=null;
 let len=0;for(let i=1;i<d.length;i++)len+=Math.hypot((d[i][0]-d[i-1][0])*cv.width,(d[i][1]-d[i-1][1])*cv.height);
 if(d.length<6||len<25){const p=down;let hit=-1;F().crypts.forEach((c,i)=>{if(inside(p,c.outline))hit=i});sel=hit;draw();return}
 snap();F().crypts.push({outline:d.map(p=>[+p[0].toFixed(4),+p[1].toFixed(4)]),tags:[]});sel=F().crypts.length-1;save();draw()};
function snap(){hist.push(JSON.stringify(F().crypts))}
function tog(t){if(sel<0)return;snap();const c=F().crypts[sel];c.tags=c.tags.includes(t)?c.tags.filter(x=>x!=t):[...c.tags,t];save();draw()}
function go(k){cur=(cur+k+names.length)%names.length;load()}
function setDone(){F().done=document.getElementById('done').checked;save();list()}
function setNote(){F().note=document.getElementById('note').value;save()}
let timer=null;function save(){clearTimeout(timer);timer=setTimeout(async()=>{const r=await fetch('/traces',{method:'POST',body:JSON.stringify(T)});
 const s=document.getElementById('saved');s.textContent=r.ok?'saved':'SAVE FAILED';s.style.color=r.ok?'#2a7':'#c00';list()},250)}
window.onkeydown=e=>{if(e.target.tagName=='INPUT')return;
 if(e.key=='ArrowRight')go(1);else if(e.key=='ArrowLeft')go(-1);
 else if(e.key=='1')tog('dilated');else if(e.key=='2')tog('serrated');else if(e.key=='3')tog('round regular');
 else if(e.key=='0'&&sel>=0){snap();F().crypts[sel].tags=[];save();draw()}
 else if((e.key=='Delete'||e.key=='Backspace')&&sel>=0){snap();F().crypts.splice(sel,1);sel=-1;save();draw()}
 else if(e.key=='z'&&(e.ctrlKey||e.metaKey)&&hist.length){F().crypts=JSON.parse(hist.pop());sel=-1;save();draw()}
 else if(e.key=='h'||e.key=='H'){const h=document.getElementById('hide');h.checked=!h.checked;draw()}
 else if(e.key=='d'||e.key=='D'){const d=document.getElementById('done');d.checked=!d.checked;setDone()}};
init();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/names":
            return self._send(200, json.dumps(NAMES).encode(), "application/json")
        if self.path == "/traces":
            return self._send(200, json.dumps(GT.load_traces()).encode(), "application/json")
        if self.path.startswith("/img/"):
            name = self.path[5:]
            if name in NAMES:
                return self._send(200, (GT.DATA / "mhist/images" / name).read_bytes(), "image/png")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/traces":
            return self._send(404, b"", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = {k: v for k, v in data.items() if k in NAMES}
        GT.TRACES.parent.mkdir(parents=True, exist_ok=True)
        if GT.TRACES.exists():
            bk = GT.TRACES.parent / "backup"
            bk.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M")      # at most one backup per minute
            shutil.copy(GT.TRACES, bk / f"traces-{stamp}.json")
        tmp = GT.TRACES.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(GT.TRACES)
        self._send(200, b"ok", "text/plain")


if __name__ == "__main__":
    print(f"{len(NAMES)} fields; traces -> {GT.TRACES}\nopen http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
