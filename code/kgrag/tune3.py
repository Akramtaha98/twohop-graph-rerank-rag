"""Select the question-type gate of the two-hop re-ranker on a HotpotQA *train* sample only (other settings frozen from tune2).

    python -m kgrag.tune3 --tuned2 results/tuned2.json --n 500 --dense bge --out results/tuned3.json
The oracle gate (gold question type) is reported as an upper bound and is never selected.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .retrievers import Dense, Hybrid2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned2", default="results/tuned2.json")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", default="tfidf", choices=["tfidf", "bge"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tuned3.json")
    a = ap.parse_args()
    b = json.loads(Path(a.tuned2).read_text())["best"]
    exs, passages = D.load_hotpot(None, a.n, a.seed, split="train")
    dense = Dense(a.dense, cache=True).fit(passages)
    rows = []
    for gate in ["none", "titles", "rule", "oracle"]:
        h = Hybrid2(dense, m=b["m"], gamma=b["gamma"], tb=b["tb"], ner=b["ner"], gate=gate).fit(passages)
        res = [h.search(e["question"], a.k, e.get("qtype"))[0] for e in exs]
        sc = [M.all_support_at_k(p, e["sup_pids"]) for p, e in zip(res, exs)]
        gated = float(np.mean([bool(h._skip(e["question"], dense.scores(e["question"]), a.k, e.get("qtype"), set(h.g.query_entities(e["question"])))) for e in exs])) if gate != "none" else 0.0
        rows.append({"gate": gate, "all_support": float(np.mean(sc)), "gated_share": gated})
        print(rows[-1], flush=True)
    best = max((r for r in rows if r["gate"] != "oracle"), key=lambda r: r["all_support"])
    Path(a.out).write_text(json.dumps({"best": best, "grid": rows, "n": a.n}, indent=2))
    print("BEST_GATE", best["gate"])


if __name__ == "__main__":
    main()
