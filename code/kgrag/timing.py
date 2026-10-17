"""Latency, indexing time and memory of the retrievers, with warm-up and repeated runs.

    python -m kgrag.timing --dataset hotpot --n 200 --repeats 5 --warmup 10 --ce --out results/timing_cpu.json

Per question the full retrieval call is timed (encoding the question, exhaustive similarity over all passages, expanded-query scorings, normalisation,
graph look-ups and, for the cross-encoder, scoring of 20 pairs). The first `--warmup` questions are run and discarded, the remaining questions are run
`--repeats` times, and the mean per-question latency of each repeat is reported as mean +- standard deviation over repeats.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from . import data as D
from .controls import cached_dense
from .retrievers import CrossRerank, Hybrid2


def sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def bench(fn, qs, warmup, repeats):
    for q in qs[:warmup]:
        fn(q)
    sync()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in qs[warmup:]:
            fn(q)
        sync()
        runs.append(1000 * (time.perf_counter() - t0) / max(1, len(qs) - warmup))
    return {"ms_per_question_mean": round(float(np.mean(runs)), 2), "ms_per_question_sd": round(float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0, 2), "runs_ms": [round(r, 2) for r in runs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pool_questions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--ce", action="store_true")
    ap.add_argument("--cache", default=None, help="embedding cache dir shared with kgrag.controls (index time is then not measured here)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exs, passages = D.load_hotpot(None, a.pool_questions or a.n, a.seed)
    exs = exs[: a.n]
    qs = [e["question"] for e in exs]
    res = {"n_passages": len(passages), "n_timed_questions": len(qs) - a.warmup, "repeats": a.repeats, "warmup": a.warmup}
    t0 = time.perf_counter()
    dense = cached_dense(passages, a.cache)
    sync()
    res["index_seconds_dense_or_cache_load"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    h = Hybrid2(dense, m=3, gamma=0.3, tb=0.5, ner="both", gate="none").fit(passages)
    res["index_seconds_graph"] = round(time.perf_counter() - t0, 1)
    hg = Hybrid2(dense, m=3, gamma=0.3, tb=0.5, ner="both", gate="titles")
    hg.__dict__.update({k: v for k, v in h.__dict__.items() if k not in {"gate"}})
    ht = Hybrid2(dense, m=3, gamma=1.0, tb=0.5, ner="both", gate="none", mode="title_only")
    ht.__dict__.update({k: v for k, v in h.__dict__.items() if k not in {"gate", "mode", "gamma"}})
    res["embedding_mb"] = round(dense.E.nbytes / 1e6, 1)
    res["graph_nnz"] = int(h.g.A.nnz)
    res["graph_entities"] = int(h.g.m)
    res["graph_matrix_mb"] = round((h.g.A.data.nbytes + h.g.A.indices.nbytes + h.g.A.indptr.nbytes + h.g.Ac.data.nbytes + h.g.Ac.indices.nbytes + h.g.Ac.indptr.nbytes) / 1e6, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if __import__("sys").platform != "darwin" else 1024 * 1024), 0)
    res["dense"] = bench(lambda q: dense.search(q, 5), qs, a.warmup, a.repeats)
    res["two_hop"] = bench(lambda q: h.search(q, 5), qs, a.warmup, a.repeats)
    res["two_hop_gate"] = bench(lambda q: hg.search(q, 5), qs, a.warmup, a.repeats)
    res["title_only"] = bench(lambda q: ht.search(q, 5), qs, a.warmup, a.repeats)
    if a.ce:
        ce = CrossRerank(dense, top=20).fit(passages)
        res["ce_dense_top20"] = bench(lambda q: ce.search(q, 5), qs, a.warmup, max(2, a.repeats // 2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
