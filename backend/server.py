"""Local backend for the deMHISTifyAI desktop app (http://127.0.0.1:8765).

POST /capture        x, y, w, h -> true-colour screenshot path (and starts reading the scale bar on it)
GET  /scale          -> the scale bar read on that screenshot: one field's size in screen pixels
POST /region/guess   image -> square field on the tissue (tissue vs glass by colour, instant)
POST /analyze        image (already cropped to the confirmed box) + researcher text
                     -> Server-Sent Events: parsed, conch, flags, judge, verdict, done
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: see engine.py (threaded Macenko crash)

import io
import json
from typing import Optional
import threading
from pathlib import Path
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import llm
from region import tissue_square

app = FastAPI(title="deMHISTifyAI backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {"engine": None, "engine_error": None, "scale": None}
FIELD_UM = 448  # one MHIST field: 224 px at ~2.0 um/px
LIBRARY = llm.Library()
POOL = ThreadPoolExecutor(4)
HEAT_POOL = ThreadPoolExecutor(1)  # heatmap passes run one at a time on the GPU, beside the LLM calls
ENGINE_POOL = ThreadPoolExecutor(1)  # every analysis on one long-lived thread (see _warm_up)
GEN_LOCK = threading.Lock()  # guards the snip counter that lets a new snip stop older heatmap passes


def _load_engine():
    try:
        from engine import Engine

        engine = Engine()
        _warm_up(engine)
        STATE["engine"] = engine
    except Exception as e:
        STATE["engine_error"] = f"{type(e).__name__}: {e}"


def _warm_up(engine):
    """One invisible analysis right after loading, on the SAME threads that serve snips: GPU libraries set up per
    thread on Windows, so a snip on a fresh thread paid ~13 s; on a warmed thread it takes ~2 s."""
    try:
        demo = Path(__file__).resolve().parent.parent / "data" / "demo" / "demo_hp_ssl.png"
        img = (Image.open(demo).convert("RGB").crop((70, 120, 1088, 1138)) if demo.exists()
               else Image.new("RGB", (448, 448), (200, 120, 180)))
        ENGINE_POOL.submit(engine.analyze, img).result()
        for sub in (32, 64, 128):
            HEAT_POOL.submit(engine.refine_heat, img, min(img.size), sub).result()
    except Exception:
        pass  # warm-up is only an optimisation


threading.Thread(target=_load_engine, daemon=True).start()


# No torch.cuda.empty_cache() between stages: re-allocating the freed GPU memory on Windows made the next snip
# ~13 s instead of ~2 s. The panel alone peaks at ~4.3 GB of the 8 GB, so keeping PyTorch's cache is safe.


async def _read(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(await file.read())).convert("RGB")


@app.get("/health")
def health():
    return {
        "conch_ready": STATE["engine"] is not None,
        "conch_error": STATE["engine_error"],
        "papers": len(LIBRARY.papers),
        "judge_model": llm.JUDGE_MODEL,
        "verdict_model": llm.VERDICT_MODEL,
    }


@app.post("/capture")
def capture(x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...)):
    """True-colour screen capture for the snip screen (GDI via Pillow; physical pixels). Runs here because
    the app's PowerShell capture helper can be stopped by antivirus, and Electron's own capture
    contrast-stretches the screen (tissue looks over-exposed)."""
    import tempfile
    from PIL import ImageGrab

    path = os.path.join(tempfile.gettempdir(), f"demhist-snip-{os.getpid()}.png")
    shot = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True)
    shot.save(path, compress_level=1)
    # read the viewer's scale bar (if any) in the background; /region/guess uses it to size the field
    STATE["scale"] = POOL.submit(_read_scale, shot)
    return {"path": path}


def _read_scale(shot):
    try:
        import scalebar
        return scalebar.read_scale(shot)
    except Exception:
        return None


def _scale(timeout: float = 3):
    # reading the bar takes ~2-8 s on a full screen (Windows OCR on each candidate bar), started at capture
    fut = STATE.get("scale")
    try:
        return fut.result(timeout=timeout) if fut is not None else None
    except Exception:
        return None


@app.get("/scale")
def scale_bar():
    """The scale bar read on the whole screenshot (before any snip), so the snip screen can size one field up front.
    Bar coordinates and field size are in screenshot pixels."""
    s = _scale(timeout=20)  # asked as soon as the snip screen opens: worth waiting for, the user is still looking
    if not s:
        return {"found": False}
    return {"found": True, "um_per_px": round(s["um_per_px"], 4), "label_um": s["label_um"], "bar": s["bar"],
            "field_px": round(FIELD_UM / s["um_per_px"], 1)}


@app.post("/region/guess")
async def region_guess(image: UploadFile = File(...)):
    img = await _read(image)
    scale = _scale(timeout=10)
    if scale:
        # one MHIST field = 448 um. The snip is in screenshot pixels, the same pixels the scale bar was measured in
        side = FIELD_UM / scale["um_per_px"]
        advice = ("zoom_out" if side > min(img.size) else "zoom_in" if side < 224 else None)
        return {"box": tissue_square(img, side), "size": img.size, "fixed": True, "advice": advice,
                "scale": {"um_per_px": round(scale["um_per_px"], 4), "label_um": scale["label_um"], "field_px": round(side)}}
    return {"box": tissue_square(img), "size": img.size, "fixed": False}


def trust_flags(researcher: dict, conch: dict, judge: dict) -> list:
    flags = []
    if not researcher["in_scope"]:
        flags.append("Outside validated scope (colon serrated polyps): results are not validated.")
    if not conch.get("calibrated"):
        flags.append("Feature scores are relative to this snip only (not yet calibrated on MHIST).")
    mag = conch.get("magnification", {})
    if mag.get("auto") and mag.get("clamped"):
        flags.append(f"Snip zoom is far from the training images (x{mag['zoom_vs_training']}); results may be less reliable.")
    ood = conch.get("ood")
    if ood and ood.get("far"):
        flags.append("This selection doesn't look like colon polyp tissue at all (photo, noise or screen content); "
                     "the image call does not apply.")
    elif ood and ood.get("flag"):
        flags.append("This snip looks unlike the MHIST training images (other tissue, stain, zoom or not tissue); "
                     "treat the image call with caution.")
    if conch.get("judge_mode") == "zero-shot":
        flags.append("Image judges are untrained (prompt-only); expect near-chance accuracy on this task.")
    panel = conch["panel"]
    if panel["agreement"] < 0.75:
        flags.append("Image judges are split (" + ", ".join(f"{'SSL' if c == 'SSA' else c} {n}" for c, n in panel["votes"].items()) + ").")
    by_set = {}
    for j in conch["judges"]:
        by_set.setdefault(json.dumps(j["prompts"], sort_keys=True), []).append(j["top"])
    flips = sum(len(set(v)) > 1 for v in by_set.values())
    if flips:
        flags.append(f"{flips} of {len(by_set)} judges change their answer after colour normalisation (colour-sensitive).")
    if judge.get("call") and judge["call"] != panel["top"]:
        flags.append("Image judges and literature reasoning disagree.")
    if judge.get("consistency") is not None and judge["consistency"] < 60:
        flags.append("Literature reasoning is inconsistent across repeated runs.")
    return flags


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), researcher: str = Form(...), cut: Optional[float] = Form(None)):
    img = await _read(image)

    def stream():
        parsed = llm.parse_researcher(researcher)
        yield _sse("parsed", parsed)

        # the image models take about a minute to load after start-up: wait for them rather than fail
        for _ in range(180):
            if STATE["engine"] is not None or STATE["engine_error"]:
                break
            time.sleep(1)
        engine = STATE["engine"]
        my_gen = None
        if engine is not None:  # this snip supersedes any earlier one still refining its map in the background
            with GEN_LOCK:
                engine.current_gen += 1
                my_gen = engine.current_gen
        if engine is None:
            yield _sse("error", {"stage": "conch", "message": STATE["engine_error"] or "The image models are still loading; try again in a minute"})
            return
        t_an = time.time()
        conch = ENGINE_POOL.submit(engine.analyze, img, None, cut).result()
        if os.environ.get("DEMHIST_PROFILE"):
            print(f"[profile] gen {my_gen}: analyze {time.time() - t_an:.2f} s, threads {threading.active_count()}", flush=True)
        if "error" in conch:
            yield _sse("error", {"stage": "conch", "message": conch["error"], "tiles": conch.get("tiles")})
            return
        yield _sse("conch", conch)

        # literature judge (Lumen) runs while the UI animates the image judges
        query = llm.judge_query(conch)  # from what this field shows (measured crypts) plus the WHO criteria
        papers = LIBRARY.search(query)
        far = bool((conch.get("ood") or {}).get("far"))
        # a photo, noise or something else that isn't colon tissue: no literature reasoning about it (and no LLM calls)
        judge_future = POOL.submit(lambda: {"error": "not_tissue"}) if far else POOL.submit(llm.research_judge, conch, papers)
        # While Lumen thinks, sharpen the heatmap on the GPU in the background: the panel's cover-up map refines
        # 16 -> 32 -> 64 -> 128 cells per field (older judges: one denser pass). Whichever is ready first is sent
        # first, so the literature judge and verdict never wait for the last heatmap pass.
        passes = [] if far else list((32, 64, 128) if conch.get("judge_mode") == "panel" else (8,))
        tile_px = conch["magnification"]["tile_px"]
        def refine(sub):
            t_r = time.time()
            try:
                return engine.refine_heat(img, tile_px, sub, gen=my_gen)
            finally:
                if os.environ.get("DEMHIST_PROFILE"):
                    print(f"[profile] gen {my_gen}: refine {sub} ended after {time.time() - t_r:.2f} s", flush=True)
        heat_future = HEAT_POOL.submit(refine, passes.pop(0)) if passes else None
        judge_sent = False
        while heat_future is not None or not judge_sent:
            waiting = [f for f in (heat_future, None if judge_sent else judge_future) if f is not None]
            done, _ = wait(waiting, return_when=FIRST_COMPLETED)
            if heat_future in done:
                try:
                    yield _sse("heat", heat_future.result())
                    stale = engine.current_gen != my_gen
                    heat_future = HEAT_POOL.submit(refine, passes.pop(0)) if passes and not stale else None
                except Exception as e:
                    if type(e).__name__ != "Superseded":  # superseded: a newer snip took over the GPU; just stop
                        yield _sse("heat_error", {"message": str(e)})
                    heat_future = None
            if not judge_sent and judge_future in done:
                judge_sent = True
                try:
                    judge = judge_future.result()
                except Exception:  # never let the literature side break the stream after the image results
                    judge = {"error": "llm_unavailable"}
                yield _sse("judge", judge)
                flags = trust_flags(parsed, conch, judge)
                yield _sse("flags", flags)
                yield _sse("verdict", llm.not_tissue_verdict() if far else llm.final_verdict(parsed, conch["panel"], judge, flags, conch))
        yield _sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
