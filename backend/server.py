"""Local backend for the deMHISTifyAI desktop app (http://127.0.0.1:8765).

POST /region/guess   image -> instant structure-based box
POST /region/refine  image -> vision-model box (a few seconds)
POST /analyze        image (already cropped to the confirmed box) + researcher text
                     -> Server-Sent Events: parsed, conch, flags, judge, verdict, done
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: see engine.py (threaded Macenko crash)

import io
import json
from typing import Optional
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import llm
from region import VisionLocator, structure_guess

app = FastAPI(title="deMHISTifyAI backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {"engine": None, "engine_error": None}
LOCATOR = VisionLocator()
LIBRARY = llm.Library()
POOL = ThreadPoolExecutor(4)


def _load_engine():
    try:
        from engine import Engine

        STATE["engine"] = Engine()
    except Exception as e:
        STATE["engine_error"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_load_engine, daemon=True).start()


def _free_gpu():
    """hand unused GPU memory back between stages: the image panel and the tissue-box model share 8 GB"""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


async def _read(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(await file.read())).convert("RGB")


@app.get("/health")
def health():
    return {
        "conch_ready": STATE["engine"] is not None,
        "conch_error": STATE["engine_error"],
        "vlm_ready": LOCATOR.ready,
        "vlm_error": LOCATOR.error,
        "vlm_last_answer": LOCATOR.last_answer,
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
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(path, compress_level=1)
    return {"path": path}


@app.post("/region/guess")
async def region_guess(image: UploadFile = File(...)):
    img = await _read(image)
    return {"box": structure_guess(img), "size": img.size}


@app.post("/region/refine")
async def region_refine(image: UploadFile = File(...)):
    img = await _read(image)
    box = POOL.submit(LOCATOR.locate, img).result()
    return {"box": box, "size": img.size, "source": "vlm" if box else "unavailable"}


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
    if ood and ood.get("flag"):
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
        if engine is None:
            yield _sse("error", {"stage": "conch", "message": STATE["engine_error"] or "The image models are still loading; try again in a minute"})
            return
        conch = engine.analyze(img, cut=cut)
        if "error" in conch:
            yield _sse("error", {"stage": "conch", "message": conch["error"], "tiles": conch.get("tiles")})
            return
        yield _sse("conch", conch)
        _free_gpu()

        # literature judge (Lumen) runs while the UI animates the image judges
        top_feats = [v for v in conch["criteria"].values() if v.get("usable")]
        query = ("sessile serrated lesion hyperplastic polyp diagnostic criteria crypt architecture distortion "
                 + " ".join(v["text"] for v in top_feats))
        papers = LIBRARY.search(query)
        judge_future = POOL.submit(llm.research_judge, conch["criteria"], papers, conch["panel"])
        # while Lumen thinks, sharpen the heatmap on the GPU: the panel's cover-up map refines 16 -> 32 -> 64 -> 128 cells
        # per field (older judges: one denser pass)
        for sub in ((32, 64, 128) if conch.get("judge_mode") == "panel" else (8,)):
            try:
                yield _sse("heat", engine.refine_heat(img, conch["magnification"]["tile_px"], sub=sub))
                _free_gpu()
            except Exception as e:
                yield _sse("heat_error", {"message": str(e)})
                break
        judge = judge_future.result()
        yield _sse("judge", judge)

        flags = trust_flags(parsed, conch, judge)
        yield _sse("flags", flags)
        yield _sse("verdict", llm.final_verdict(parsed, conch["panel"], judge, flags))
        yield _sse("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
