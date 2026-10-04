"""Select hybrid hyper-parameters on a HotpotQA *train* sample (never on the evaluation questions).

    python -m kgrag.tune --n 500 --dense tfidf --out results/tuned.json
Stores the best (alpha, lam, beta); pass the file to `kgrag.run --tuned`.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .retrievers import Dense, Hybrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", default="tfidf", choices=["tfidf", "bge"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tuned.json")
    a = ap.parse_args()

    exs, passages = D.load_hotpot(None, a.n, a.seed, split="train")
    dense = Dense(a.dense).fit(passages)
    grid = list(itertools.product([0.15, 0.3], [0.25, 0.5, 0.75], [0.3, 0.5, 0.7]))
    rows = []
    for alpha, lam, beta in grid:
        h = Hybrid(dense, alpha=alpha, lam=lam, beta=beta).fit(passages)
        sc = [M.all_support_at_k(h.search(e["question"], a.k)[0], e["sup_pids"]) for e in exs]
        rows.append({"alpha": alpha, "lam": lam, "beta": beta, "all_support": float(np.mean(sc))})
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["all_support"])
    spec = f"hybrid:hybrid:alpha={best['alpha']},lam={best['lam']},beta={best['beta']}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"best": best, "spec": spec, "grid": rows, "n": a.n, "dense": a.dense}, indent=2))
    print("BEST", spec)


if __name__ == "__main__":
    main()
