"""Recall@2/@5, All@2/@5 (chunk level and, for MultiHop-RAG, document level) and paired bootstrap against a reference system.

  python -m kgrag.rank_report --dataset hotpot --n 1000 --seed 0 --dir ../results_jis/hotpotA --ref dense \
      --systems dense bm25 hybrid2_gate titlelink hipporag2 mdr --latex ../results_jis/hotpotA/table.tex
R@2/R@5 are the metrics HippoRAG and LinearRAG report, so published numbers can be put next to ours (with the caveat that corpora differ).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M


def per_q(recs, exs, passages):
    out = {k: [] for k in ("R@2", "R@5", "All@2", "All@5", "DocAll@5")}
    for r, ex in zip(recs, exs):
        rk = r["ranked"]
        out["R@2"].append(M.recall_at_k(rk[:2], ex["sup_pids"]))
        out["R@5"].append(M.recall_at_k(rk[:5], ex["sup_pids"]))
        out["All@2"].append(M.all_support_at_k(rk[:2], ex["sup_pids"]))
        out["All@5"].append(M.all_support_at_k(rk[:5], ex["sup_pids"]))
        if "sup_docs" in ex:
            out["DocAll@5"].append(float(ex["sup_docs"] <= {passages[p]["doc"] for p in rk[:5]}))
    return {k: np.array(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--ref", default="dense")
    ap.add_argument("--latex", default=None)
    a = ap.parse_args()
    exs, ps = D.load(a.dataset, a.path, a.n, a.seed)
    res = {}
    for s in a.systems:
        recs = [json.loads(l) for l in (Path(a.dir) / f"{s}.jsonl").read_text().splitlines()]
        assert [r["qid"] for r in recs] == [e["qid"] for e in exs], f"{s}: question order differs from the dataset load"
        res[s] = per_q(recs, exs, ps)
    metrics = list(res[a.ref])
    rows = []
    for s in a.systems:
        cells = []
        for m in metrics:
            v = 100 * res[s][m].mean()
            if s == a.ref or s not in res or m != "All@5":
                cells.append(f"{v:.1f}")
            else:
                d, lo, hi, p = M.paired_bootstrap(res[s][m], res[a.ref][m])
                cells.append(f"{v:.1f} ({100*d:+.1f} [{100*lo:+.1f},{100*hi:+.1f}])")
        rows.append((s, cells))
    print("system".ljust(18), *[m.ljust(30 if m == "All@5" else 9) for m in metrics])
    for s, c in rows:
        print(s.ljust(18), *[x.ljust(30 if m == "All@5" else 9) for x, m in zip(c, metrics)])
    if a.latex:
        L = ["\\begin{tabular}{l" + "r" * len(metrics) + "}", "\\toprule", "System & " + " & ".join(metrics) + " \\\\", "\\midrule"]
        L += [s.replace("_", "\\_") + " & " + " & ".join(c) + " \\\\" for s, c in rows] + ["\\bottomrule", "\\end{tabular}"]
        Path(a.latex).write_text("\n".join(L) + "\n")
        print("wrote", a.latex)


if __name__ == "__main__":
    main()
