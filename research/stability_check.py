"""Stability: 25 sequential /analyze requests (varied fields) + 2 overlapping ones; time and GPU memory each."""
import io, json, random, subprocess, threading, time
from pathlib import Path
import requests
from PIL import Image

R = Path("C:/Users/Ahaan/Desktop/deMHISTifyAI")
API = "http://127.0.0.1:8765"
ims = sorted((R / "data/mhist/images").glob("*.png"))
random.seed(0)


def gpu_mb():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0])


def png(path, size):
    b = io.BytesIO(); Image.open(path).convert("RGB").resize((size, size)).save(b, "PNG"); return b.getvalue()


def analyze(data):
    t0 = time.time(); events = []; first_conch = None
    with requests.post(API + "/analyze", files={"image": ("s.png", data, "image/png")},
                       data={"researcher": "hyperplastic polyp, 60%"}, stream=True, timeout=600) as r:
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("event:"):
                ev = line.split(":", 1)[1].strip(); events.append(ev)
                if ev == "conch" and first_conch is None:
                    first_conch = time.time() - t0
    return {"total_s": round(time.time() - t0, 1), "to_result_s": round(first_conch or -1, 1), "events": events}


while True:
    try:
        if requests.get(API + "/health", timeout=2).json().get("conch_ready"): break
    except Exception: pass
    time.sleep(3)
base = gpu_mb()
rows = []
for i in range(25):
    r = analyze(png(random.choice(ims), random.choice([300, 448, 600, 900])))
    r["gpu_mb"] = gpu_mb(); rows.append(r)
    print(i, r["to_result_s"], r["total_s"], r["gpu_mb"], "done" in r["events"], flush=True)
# two overlapping snips (re-snip while the first is still streaming)
res = {}
def go(k): res[k] = analyze(png(random.choice(ims), 600))
t1 = threading.Thread(target=go, args=("a",)); t1.start(); time.sleep(2)
t2 = threading.Thread(target=go, args=("b",)); t2.start(); t1.join(); t2.join()
out = {"gpu_mb_idle_after_load": base, "sequential": rows,
       "overlap": {k: {**v, "complete": "done" in v["events"]} for k, v in res.items()}, "gpu_mb_end": gpu_mb()}
(R / "docs/results/stability_check.json").write_text(json.dumps(out, indent=1))
seq = [r["to_result_s"] for r in rows]
print("result time median", sorted(seq)[len(seq) // 2], "max", max(seq), "| gpu start", base, "max", max(r["gpu_mb"] for r in rows),
      "end", out["gpu_mb_end"], "| overlap complete", [v["complete"] for v in out["overlap"].values()])
