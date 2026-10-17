"""Controls, candidate coverage and corpus-scale checks for the two-hop re-ranker (retrieval only).

For every question the first-hop passages and the expanded-query scores are computed ONCE and shared by all variants, so
the variants differ only in how the boost is formed:

  two_hop      graph-restricted boost  w_p * (s2_p(c) + tb * 1[title(c) in B(p)])   for c in the graph neighbours of p   (the proposed method)
  expand_all   same expanded queries, same first-hop passages, same encoder, but NO graph: every passage c != p gets w_p * s2_p(c)
  title_only   no expanded query: graph neighbours whose title is a bridge entity get w_p * tb (the graph + title match only)
  dense        the dense ranking

Each variant is reported ungated and with the title gate. gamma is either frozen (0.3, the value tuned for two_hop on HotpotQA train)
or tuned for that variant on the HotpotQA TRAIN sample (see --tune); the evaluation labels are never used for selection.

    python -m kgrag.controls --dataset hotpot --n 1000 --seed 0 --out results/controls/hotpot_A --gammas 0.3 --cache results/emb
    python -m kgrag.controls --dataset hotpot --split train --n 500 --tune --out results/controls/train
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .retrievers import Dense, Hybrid2, _ptext

VARIANTS = ["dense", "two_hop", "expand_all", "title_only"]


def cached_dense(passages, cache_dir):
    """BGE dense index with the passage embeddings cached on disk (keyed by the corpus content)."""
    from sentence_transformers import SentenceTransformer

    d = Dense("bge")
    d.st = SentenceTransformer(d.model_name)
    if cache_dir:
        h = hashlib.md5("\x00".join(_ptext(p) for p in passages).encode()).hexdigest()[:16]
        f = Path(cache_dir) / f"E_{h}.npy"
        if f.exists():
            d.E = np.load(f)
            return d
    d.E = d.st.encode([_ptext(p) for p in passages], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(f, d.E)
    return d


def _norm(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def components(h, q):
    """Everything the variants need: dense scores, first-hop set and weights, and per first-hop passage the graph candidates,
    the title-link flags and the min-max normalised expanded-query scores over the WHOLE corpus."""
    g = h.g
    d = h.dense.scores(q)
    dn = _norm(d)
    qset = set(g.query_entities(q))
    first = np.argsort(-d)[: h.m]
    w = np.exp((dn[first] - dn[first].max()) / h.tau)
    w = w / w.max()
    per = []
    for p, wp in zip(first, w):
        cols = g.A.indices[g.A.indptr[p] : g.A.indptr[p + 1]]
        bridge = [j for j in cols if j not in qset]
        if not bridge:
            continue
        bset = set(bridge)
        cand = np.unique(np.concatenate([g.Ac.indices[g.Ac.indptr[j] : g.Ac.indptr[j + 1]] for j in bridge]))
        cand = cand[cand != p]
        sc = h.dense.scores(h._expand(q, p))
        s2full = _norm(sc)
        tl = np.array([h.title_col[c] in bset for c in cand], dtype=float) if len(cand) else np.zeros(0)
        per.append((int(p), float(wp), cand, tl, s2full))
    return d, dn, qset, per


def variant_scores(h, dn, per, gamma, tb):
    n = len(dn)
    two, exp, tit = np.zeros(n), np.zeros(n), np.zeros(n)
    for p, wp, cand, tl, s2 in per:
        if len(cand):
            two[cand] = np.maximum(two[cand], wp * (s2[cand] + tb * tl))
            tit[cand] = np.maximum(tit[cand], wp * tb * tl)
        allc = np.ones(n, dtype=bool)
        allc[p] = False
        exp[allc] = np.maximum(exp[allc], wp * s2[allc])
    return {"two_hop": dn + gamma * two, "expand_all": dn + gamma * exp, "title_only": dn + gamma * tit}


def evaluate(h, exs, gammas, tb, ks=(5, 20), ce=None):
    """Return {variant@gamma[+gate]: [per-question record]}. Optionally a cross-encoder `ce` (callable(q, pids)->scores) for the
    common-pool comparison."""
    out = {}

    def add(name, ranking, e, gated=False):
        r20 = [int(x) for x in ranking[:20]]
        out.setdefault(name, []).append({
            "qid": e["qid"], "qtype": e.get("qtype", ""), "gated": gated,
            "recall": M.recall_at_k(r20[:5], e["sup_pids"]), "all_support": M.all_support_at_k(r20[:5], e["sup_pids"]),
            "recall20": M.recall_at_k(r20, e["sup_pids"]), "all20": M.all_support_at_k(r20, e["sup_pids"]),
            "recall10": M.recall_at_k(r20[:10], e["sup_pids"]), "all10": M.all_support_at_k(r20[:10], e["sup_pids"]),
        })

    for e in exs:
        q = e["question"]
        d, dn, qset, per = components(h, q)
        dense_rank = list(np.argsort(-d)[:20])
        gate = h._skip(q, d, 5, e.get("qtype"), qset) if h.gate != "none" else False
        add("dense", dense_rank, e)
        for gam in gammas:
            sc = variant_scores(h, dn, per, gam, tb)
            for v, s in sc.items():
                rank = list(np.argsort(-s)[:20])
                add(f"{v}@{gam}", rank, e)
                add(f"{v}@{gam}+gate", dense_rank if gate else rank, e, gated=bool(gate))
                if ce is not None and v == "two_hop" and gam == gammas[0]:
                    # common-pool comparison: the same candidate pool ordered by dense score, by the two-hop score, and by the cross-encoder
                    for pool_name, pool in (("poolD", dense_rank), ("poolH", rank)):
                        cs = np.asarray(ce(q, [int(x) for x in pool]))
                        add(f"{pool_name}:ce", [pool[i] for i in np.argsort(-cs)], e)
                        add(f"{pool_name}:dense", sorted(pool, key=lambda c: -d[c]), e)
                        add(f"{pool_name}:two_hop", sorted(pool, key=lambda c: -s[c]), e)
    return out


def summarise(out):
    rows = {}
    for name, recs in out.items():
        rows[name] = {k: round(100 * float(np.mean([r[k] for r in recs])), 2) for k in ("all_support", "recall", "all10", "all20", "recall20")}
        rows[name]["n"] = len(recs)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot", choices=["hotpot", "2wiki", "musique"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_questions", type=int, default=None, help="build the corpus from this many questions (>= n) but evaluate the first --n")
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.3])
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--tb", type=float, default=0.5)
    ap.add_argument("--ner", default="both")
    ap.add_argument("--gate", default="titles")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ce", action="store_true", help="also run the cross-encoder common-pool comparison (slow on CPU)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    npool = a.pool_questions or a.n
    if a.dataset == "hotpot":
        exs, passages = D.load_hotpot(None, npool, a.seed, split=a.split)
    elif a.dataset == "2wiki":
        exs, passages = D.load_2wiki(None, npool, a.seed)
    else:
        exs, passages = D.load_musique(None, npool, a.seed)
    exs = exs[: a.n]
    print(f"{a.dataset}/{a.split}: {len(exs)} questions, {len(passages)} passages", flush=True)
    t0 = time.time()
    dense = cached_dense(passages, a.cache)
    t_dense = time.time() - t0
    t0 = time.time()
    h = Hybrid2(dense, m=a.m, gamma=a.gammas[0], tb=a.tb, ner=a.ner, gate=a.gate).fit(passages)
    t_graph = time.time() - t0
    ce = None
    if a.ce:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        ce = lambda q, pids: model.predict([(q, _ptext(passages[p])) for p in pids], batch_size=20, show_progress_bar=False)
    t0 = time.time()
    out = evaluate(h, exs, a.gammas, a.tb, ce=ce)
    t_eval = time.time() - t0
    res = summarise(out)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": a.dataset, "n_questions": len(exs), "n_passages": len(passages), "pool_questions": npool, "seed": a.seed,
            "seconds_dense_index": round(t_dense, 1), "seconds_graph_build": round(t_graph, 1), "seconds_eval": round(t_eval, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
            "embedding_mb": round(dense.E.nbytes / 1e6, 1), "graph_nnz": int(h.g.A.nnz), "graph_entities": int(h.g.m),
            "gammas": a.gammas, "tb": a.tb, "m": a.m, "gate": a.gate}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "rows": res}, indent=1))
    for name, recs in out.items():
        with open(out_dir / (name.replace(":", "_").replace("+", "_") + ".jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(json.dumps(meta))
    for k, v in res.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
