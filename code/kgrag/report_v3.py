"""Summarise the v3 experiments (gate, cross-encoder, second datasets): means, paired bootstrap, per-type All@5.

    python -m kgrag.report_v3 --dirs HotpotA=../results/eval_v2_seed0 HotpotB=../results/eval_v2_seed1 2Wiki=../results/eval_2wiki MuSiQue=../results/eval_musique --out ../results/v3_report.txt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M

PAIRS = [("hybrid2", "dense"), ("hybrid2_gate", "dense"), ("hybrid2_gate", "hybrid2"), ("ce_dense", "dense"), ("hybrid2_gate", "ce_dense"),
         ("hybrid2", "ce_dense"), ("ce_hybrid2_gate", "ce_dense"), ("ce_hybrid2", "ce_dense"), ("hybrid2_oracle", "hybrid2")]


def rd(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default="v3_report.txt")
    a = ap.parse_args()
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    for item in a.dirs:
        label, d = item.split("=")
        d = Path(d)
        summ = json.loads((d / "summary.json").read_text())
        P(f"\n=== {label}  (n={summ['n_questions']}, passages={summ['n_passages']}) ===")
        P(f"{'system':18s} {'R@5':>6s} {'All@5':>6s} {'gated':>6s} {'sec':>7s}")
        for s, v in summ["systems"].items():
            P(f"{s:18s} {100*v['recall']:6.1f} {100*v['all_support']:6.1f} {100*v.get('gated', 0):6.1f} {v['seconds']:7.1f}")
        recs = {s: rd(d / f"{s}.jsonl") for s in summ["systems"]}
        P("-- paired bootstrap (10,000 resamples), difference in points [95% CI] p")
        for a_, b_ in PAIRS:
            if recs.get(a_) is None or recs.get(b_) is None:
                continue
            ids = sorted(set(recs[a_]) & set(recs[b_]))
            for key in ("all_support", "recall"):
                dd, lo, hi, p = M.paired_bootstrap([recs[a_][i][key] for i in ids], [recs[b_][i][key] for i in ids])
                P(f"{a_:16s} vs {b_:12s} {key:12s} {100*dd:+5.1f} [{100*lo:+5.1f},{100*hi:+5.1f}] p={p:.4f}")
        qt = {}
        for s, r in recs.items():
            if r:
                for i, x in r.items():
                    if x.get("qtype"):
                        qt[i] = x["qtype"]
        if qt:
            P("-- All@5 by question type")
            types = sorted(set(qt.values()))
            P(f"{'system':18s} " + " ".join(f"{t[:14]:>14s}" for t in types))
            P(f"{'(n)':18s} " + " ".join(f"{sum(1 for v in qt.values() if v == t):14d}" for t in types))
            for s in ("dense", "ce_dense", "hybrid2", "hybrid2_gate", "hybrid2_oracle", "ce_hybrid2", "ce_hybrid2_gate"):
                if recs.get(s):
                    by = defaultdict(list)
                    for i, x in recs[s].items():
                        if i in qt:
                            by[qt[i]].append(x["all_support"])
                    P(f"{s:18s} " + " ".join(f"{100*np.mean(by[t]):14.1f}" for t in types))
    Path(a.out).write_text("\n".join(lines))
    print("REPORT_WRITTEN")


if __name__ == "__main__":
    main()
