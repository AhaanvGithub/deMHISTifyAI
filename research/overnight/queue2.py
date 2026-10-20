"""Follow-up to queue_runner.py (agreed 2026-10-19). After the main queue finishes:
1. ConvNeXt-Base on the same 5 folds; it replaces ConvNeXt-Tiny if its out-of-fold AUC is higher.
2. MUSK with LoRA on all 24 layers (lr 1e-4); it replaces the 2-layer MUSK if its out-of-fold AUC is higher.
3. Lift the panel hold and run the panel + single test evaluation once, with the final line-up.
Same allowed hours and log as queue_runner.py."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import queue_runner as q  # noqa: E402

BASE = ["--backbone", "convnext_base", "--blocks", "-1", "--lr", "1e-4", "--batch", "8", "--accum", "2"]
LORA = ["--backbone", "musk_lora", "--lr", "1e-4", "--batch", "4", "--accum", "4"]
LORA_SMALL = ["--backbone", "musk_lora", "--lr", "1e-4", "--batch", "2", "--accum", "8"]  # if batch 4 runs out of memory
HOLD = os.path.join(q.REPO, "data", "models", "cv", "HOLD_PANEL")


def oof_auc(tag):
    return json.load(open(os.path.join(q.REPO, "docs", "results", "cv", f"{tag}.json"))).get("oof_auc", -1)


def challenger(name, args, tag, incumbent, fallback_args=None):
    """cross-validate `tag`; keep it only if its out-of-fold AUC beats `incumbent`'s"""
    ok = q.run(f"CV {tag}", lambda: ["research/kfold_cv.py", *args, "--tag", tag])
    if not ok and fallback_args:
        q.log(f"queue2: retrying {tag} with a smaller batch")
        args = fallback_args
        ok = q.run(f"CV {tag} (smaller batch)", lambda: ["research/kfold_cv.py", *args, "--tag", tag])
    if not ok:
        q.log(f"queue2: {name} did not finish; keeping {incumbent}")
        return incumbent
    a, b = oof_auc(incumbent), oof_auc(tag)
    winner = tag if b > a else incumbent
    q.log(f"queue2: out-of-fold AUC {incumbent} {a} vs {tag} {b} -> panel uses {winner}")
    if winner == tag:
        q.run(f"final retrain {tag}", lambda: ["research/kfold_cv.py", *args, "--tag", tag, "--final-epochs", q.median_epoch(tag)])
    return winner


if __name__ == "__main__":
    while "queue finished" not in (open(q.LOG, encoding="utf-8").read() if os.path.exists(q.LOG) else ""):
        time.sleep(30)
    q.log("queue2: main queue finished")
    cnn = challenger("ConvNeXt-Base", BASE, "convnext_base", "convnext_full")
    musk = challenger("MUSK + LoRA", LORA, "musk_lora", "musk_b2", LORA_SMALL)
    members = ["conch_full", cnn, "plip_full", musk]
    if os.path.exists(HOLD):
        os.remove(HOLD)
    q.run("panel + single test evaluation (final line-up)", lambda: ["research/panel.py", "--members", *members])
    q.log("queue2 finished")
