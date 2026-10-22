"""Benchmark queue (agreed 2026-10-19): 5-fold CV for the 4 panel members (CONCH, ConvNeXt, PLIP, MUSK),
final retrain of each on all of train, then panel selection + the single test evaluation. Runs one job
at a time inside the allowed hours (2026-10-19 until 20:00, then 22:00 - 2026-10-20 09:00); a job
running at a cut-off is killed and restarted in the next window (CV resumes from its last finished fold).
Log: %TEMP%/gutfeeling/queue.log (also printed)."""
import datetime as dt
import json
import os
import subprocess
import threading
import time

REPO = r"C:\Users\Ahaan\Desktop\deMHISTifyAI"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
LOG = os.path.join(os.environ["TEMP"], "gutfeeling", "queue.log")
ENV = {**os.environ, "MUSK_SRC": os.path.join(os.environ["TEMP"], "gutfeeling", "musk_src", "MUSK"),
       "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
D = dt.datetime
WINDOWS_FILE = os.path.join(os.environ["TEMP"], "gutfeeling", "windows.json")  # allowed hours, re-read on every check


def windows():
    return [(D.strptime(a, "%Y-%m-%d %H:%M"), D.strptime(b, "%Y-%m-%d %H:%M")) for a, b in json.load(open(WINDOWS_FILE))]

MEMBERS = {  # tag: kfold_cv.py arguments, in order of expected payoff
    "conch_full": ["--backbone", "conch", "--blocks", "-1", "--lr", "1e-5", "--batch", "8", "--accum", "2"],
    "convnext_full": ["--backbone", "convnext", "--blocks", "-1", "--lr", "1e-4"],
    "plip_full": ["--backbone", "plip", "--blocks", "-1", "--lr", "1e-5"],
    "musk_b2": ["--backbone", "musk", "--blocks", "2", "--lr", "1e-5", "--batch", "8"],
}


def median_epoch(tag):
    return str(json.load(open(os.path.join(REPO, "docs", "results", "cv", f"{tag}.json")))["median_best_epoch"])


JOBS = [(f"CV {t}", lambda t=t: ["research/kfold_cv.py", *MEMBERS[t], "--tag", t]) for t in MEMBERS]
JOBS += [(f"final retrain {t}", lambda t=t: ["research/kfold_cv.py", *MEMBERS[t], "--tag", t, "--final-epochs", median_epoch(t)])
         for t in MEMBERS]
JOBS += [("panel + single test evaluation", lambda: ["research/panel.py", "--members", *MEMBERS])]


def log(msg):
    line = f"[{D.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def window_end(now):
    for a, b in windows():
        if a <= now < b:
            return b
    return None


def wait_for_window():
    while True:
        now = D.now()
        if window_end(now):
            return True
        nxt = [a for a, b in windows() if a > now]
        if not nxt:
            return False
        log(f"outside allowed hours; waiting until {nxt[0]:%m-%d %H:%M}")
        while D.now() < nxt[0]:  # short naps against the wall clock: a long sleep stalls if the laptop sleeps
            time.sleep(60)


def run(name, make_args):
    while True:
        if not wait_for_window():
            log(f"no time left for: {name}")
            return False
        end = window_end(D.now())
        try:
            args = make_args()
        except Exception as e:  # e.g. a final retrain whose CV did not finish
            log(f"FAILED {name}: cannot build command ({e})")
            return False
        log(f"START {name} (window ends {end:%H:%M}): {' '.join(args)}")
        p = subprocess.Popen([PY, "-W", "ignore", *args], cwd=REPO, env=ENV, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def pump():
            for line in p.stdout:
                s = line.rstrip()
                if s and "Warning" not in s and "warn(" not in s and "it/s]" not in s:
                    log("   " + s)
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        killed = False
        while p.poll() is None:
            end = window_end(D.now()) or D.now()  # hours can change while a job runs
            if D.now() >= end:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                killed = True
                log(f"PAUSED {name} at cut-off {end:%H:%M}; will resume")
                break
            time.sleep(1)
        t.join(timeout=10)
        if killed:
            continue
        log(f"{'DONE' if p.returncode == 0 else 'FAILED (exit %s)' % p.returncode} {name}")
        return p.returncode == 0


if __name__ == "__main__":
    done = open(LOG, encoding="utf-8").read() if os.path.exists(LOG) else ""
    for name, make_args in JOBS:
        if f"DONE {name}" in done:
            log(f"already done: {name}")
            continue
        run(name, make_args)
    log("queue finished")
