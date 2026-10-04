"""Select Hybrid2 hyper-parameters and the NER mode on a HotpotQA *train* sample only.

    python -m kgrag.tune2 --n 500 --dense bge --out results/tuned2.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .retrievers import Dense, Hybrid2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", default="tfidf", choices=["tfidf", "bge"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ners", nargs="+", default=["regex", "spacy", "both"])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--ms", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--tbs", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--out", default="results/tuned2.json")
    a = ap.parse_args()

    exs, passages = D.load_hotpot(None, a.n, a.seed, split="train")
    dense = Dense(a.dense, cache=True).fit(passages)
    base = float(np.mean([M.all_support_at_k(dense.search(e["question"], a.k)[0], e["sup_pids"]) for e in exs]))
    print("dense baseline all@k on train:", round(base, 4), flush=True)
    rows = []
    for ner in a.ners:
        for m, gamma, tb in itertools.product(a.ms, a.gammas, a.tbs):
            h = Hybrid2(dense, m=m, gamma=gamma, tb=tb, ner=ner).fit(passages)
            sc = [M.all_support_at_k(h.search(e["question"], a.k)[0], e["sup_pids"]) for e in exs]
            rows.append({"ner": ner, "m": m, "gamma": gamma, "tb": tb, "all_support": float(np.mean(sc))})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["all_support"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"best": best, "grid": rows, "n": a.n, "dense": a.dense, "dense_baseline": base}, indent=2))
    print("BEST", best)


if __name__ == "__main__":
    main()
