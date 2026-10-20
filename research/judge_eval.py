#!/usr/bin/env python3
"""Literature judge: old vs new, on MHIST TRAINING images (4 clear HP, 4 clear SSL, 4 split 3-4 votes).
Both judges see the same image results; each answer is scored with the same rule checker (backend/llm.py
_violations): invented citations, overclaiming words, evidence that wasn't measured, "seen" claims about features
that weren't measured. Also: call vs the 7-pathologist majority, agreement with the panel, time.
Usage: python research/judge_eval.py OLD_LLM_PY [--reuse-old]   (OLD_LLM_PY: a copy of the previous backend/llm.py;
--reuse-old keeps the old judge's answers from the last docs/results/judge_eval.json instead of asking again)
Writes docs/results/judge_eval.json. Uses ~120 Lumen calls."""

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "research"))
import llm as new  # noqa: E402
from engine import Engine  # noqa: E402

spec = importlib.util.spec_from_file_location("llm_old", sys.argv[1])
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

ann = pd.read_csv(ROOT / "data/mhist/annotations.csv")
tr = ann[ann.Partition == "train"]
v = "Number of Annotators who Selected SSA (Out of 7)"
cases = pd.concat([tr[tr[v] == 0].sample(4, random_state=5), tr[tr[v] == 7].sample(4, random_state=5),
                   tr[tr[v].isin([3, 4])].sample(4, random_state=5)])

e = Engine()
e._dense_heat = lambda snip, tiles, sub=4: {"rows": 1, "cols": 1, "p_ssa": [0.5], "tissue": [1.0], "features": {}}
lib_new = new.Library()
rows = []
prev = {}
if "--reuse-old" in sys.argv and (ROOT / "docs/results/judge_eval.json").exists():
    prev = {r["image"]: r["old"] for r in json.loads((ROOT / "docs/results/judge_eval.json").read_text())["rows"]}
for _, r in cases.iterrows():
    img = Image.open(ROOT / "data/mhist/images" / r["Image Name"]).convert("RGB").resize((560, 560))
    conch = e.analyze(img)
    _, ids, not_measured = new._measured(conch)
    majority = "SSA" if r[v] >= 4 else "HP"
    out = {"image": r["Image Name"], "votes": int(r[v]), "majority": majority, "panel": conch["panel"]["top"]}
    # old judge: its own query and call style
    q_old = ("sessile serrated lesion hyperplastic polyp diagnostic criteria crypt architecture distortion "
             + " ".join(x["text"] for x in conch["criteria"].values() if x.get("usable")))
    for name, fn, papers in (
            ("old", lambda p: old.research_judge(conch["criteria"], p, conch["panel"]), lib_new.search(q_old)),
            ("new", lambda p: new.research_judge(conch, p), lib_new.search(new.judge_query(conch)))):
        if name == "old" and r["Image Name"] in prev:
            out["old"] = prev[r["Image Name"]]; continue
        t0 = time.time(); j = fn(papers); dt = time.time() - t0
        if "error" in j:
            out[name] = {"error": j["error"]}; continue
        valid = {p["ref"] for p in papers}
        shown = {"reasoning": j.get("reasoning", ""), "caveats": j.get("caveats", ""), "check_by_eye": j.get("check_by_eye", []),
                 "key_features": [{**f, "feature_id": (f["feature_id"] if str(f.get("feature_id", "")).startswith("crypts:")
                                                       else "crit:" + str(f.get("feature_id", "")))} for f in j.get("key_features", [])]}
        txt = json.dumps(shown)
        out[name] = {"call": j["call"], "matches_majority": j["call"] == majority, "agrees_with_panel": j["call"] == conch["panel"]["top"],
                     "consistency": j["consistency"], "rule_breaks": new._violations(shown, valid, ids, not_measured),
                     "invented_citations": sum(1 for c in re.findall(r"P\d+", txt) if c not in valid),
                     "overclaims": len(new.OVERCLAIM.findall(txt)), "n_citations": len(set(re.findall(r"P\d+", txt))),
                     "distinct_topics_given": len({p["title"].split(":")[0] for p in papers}), "seconds": round(dt, 1),
                     "reasoning": j.get("reasoning", ""), "check_by_eye": j.get("check_by_eye", []),
                     "key_features": j.get("key_features", [])}
    rows.append(out)
    print(json.dumps({k: (val if not isinstance(val, dict) else {kk: val.get(kk) for kk in ("call", "rule_breaks", "overclaims", "invented_citations", "seconds", "error")})
                      for k, val in out.items()}), flush=True)


def summ(name):
    ok = [x[name] for x in rows if "error" not in x[name]]
    return {"answers": len(ok), "matches_majority": sum(o["matches_majority"] for o in ok), "agrees_with_panel": sum(o["agrees_with_panel"] for o in ok),
            "rule_breaks_total": sum(o["rule_breaks"] for o in ok), "answers_with_any_rule_break": sum(o["rule_breaks"] > 0 for o in ok),
            "invented_citations": sum(o["invented_citations"] for o in ok), "overclaims": sum(o["overclaims"] for o in ok),
            "mean_citations": round(sum(o["n_citations"] for o in ok) / max(1, len(ok)), 1),
            "mean_topics_given": round(sum(o["distinct_topics_given"] for o in ok) / max(1, len(ok)), 1),
            "mean_seconds": round(sum(o["seconds"] for o in ok) / max(1, len(ok)), 1)}


res = {"cases": len(rows), "old": summ("old"), "new": summ("new"), "rows": rows}
(ROOT / "docs/results/judge_eval.json").write_text(json.dumps(res, indent=2))
print(json.dumps({"old": res["old"], "new": res["new"]}, indent=2))
