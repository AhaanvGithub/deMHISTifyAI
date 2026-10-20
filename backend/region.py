"""Find the slide/tissue region inside a screen snip.

Two stages, matching the app's UX:
1. structure_guess(): instant. Tissue is textured and saturated; toolbars,
   taskbars and page chrome are mostly flat. Returns a box in under ~50 ms.
2. VisionLocator.locate(): a small local vision-language model (Qwen2.5-VL-3B,
   4-bit on the laptop GPU) refines the box a few seconds later.
The user can always drag the box; whatever box they settle on is what CONCH sees.
"""

import json
import re
import threading
from typing import Optional, Tuple

import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 in snip pixels

VLM_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_PROMPT = (
    "This is a screenshot. Find the region that shows the microscope image of tissue "
    "(the histology slide content), excluding toolbars, menus, text, taskbars and blank margins. "
    'Answer only with JSON: {"bbox_2d": [x1, y1, x2, y2]}'
)


def structure_guess(img: Image.Image, work: int = 256) -> Box:
    from scipy import ndimage

    w, h = img.size
    scale = work / max(w, h)
    small = img.convert("RGB").resize((max(int(w * scale), 8), max(int(h * scale), 8)))
    a = np.asarray(small, dtype=np.float32) / 255.0

    gray = a.mean(axis=2)
    gy, gx = np.gradient(gray)
    texture = ndimage.uniform_filter(np.hypot(gx, gy), size=5)
    sat = a.max(axis=2) - a.min(axis=2)  # cheap saturation
    not_white = gray < 0.92
    score = (texture / (texture.max() + 1e-6)) * 0.6 + (sat / (sat.max() + 1e-6)) * 0.4
    mask = (score > np.percentile(score, 55)) & not_white
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_opening(mask, iterations=1)

    labels, n = ndimage.label(mask)
    if n == 0:
        return (0, 0, w, h)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    biggest = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(biggest)
    x0, x1 = xs.min() / scale, (xs.max() + 1) / scale
    y0, y1 = ys.min() / scale, (ys.max() + 1) / scale
    # if the region is tiny relative to the snip, the snip is probably all tissue
    if (x1 - x0) * (y1 - y0) < 0.15 * w * h:
        return (0, 0, w, h)
    return (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h)))


class VisionLocator:
    """Loads the VLM in the background so app start-up isn't blocked."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.error: Optional[str] = None
        self.last_answer: Optional[str] = None  # raw model text, for debugging
        self._ready = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            # eager attention: the default SDPA path produces garbage tokens on RTX 50-series (Blackwell)
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": "eager"}
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                )
            except Exception:
                pass  # no bitsandbytes: fall back to bf16 (~7.5 GB, tight on 8 GB)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_ID, **kwargs).eval()
            # cap resolution: plenty for locating a large region, keeps it fast, and keeps its attention step small
            # enough to share the 8 GB GPU with the image panel
            self.processor = AutoProcessor.from_pretrained(VLM_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.model is not None

    def locate(self, img: Image.Image, timeout: float = 60) -> Optional[Box]:
        import torch

        if not self._ready.wait(timeout) or self.model is None:
            return None
        img = img.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        except torch.OutOfMemoryError:  # GPU shared with the image panel: fall back to the structure-based box
            self.last_answer = "out of GPU memory"
            return None
        finally:
            torch.cuda.empty_cache()
        answer = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        self.last_answer = answer

        m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", answer)
        if not m:
            return None
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        # Qwen2.5-VL answers in the pixel space of its resized input image
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        sx, sy = img.width / (gw * 14), img.height / (gh * 14)
        box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
        x0, y0, x1, y1 = box
        x0, x1 = sorted((max(0, x0), min(img.width, x1)))
        y0, y1 = sorted((max(0, y0), min(img.height, y1)))
        if (x1 - x0) < 32 or (y1 - y0) < 32:
            return None
        return (x0, y0, x1, y1)
