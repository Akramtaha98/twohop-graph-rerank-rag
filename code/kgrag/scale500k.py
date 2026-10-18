"""Large-pool and blind-title validation on HotpotQA.

Pool: every dev question's paragraphs (66,635 passages, as in the corpus-scale experiment) plus the paragraphs of randomly drawn *train* questions until
the pool holds --pool_size passages. Evaluation: the untouched sample C (seed 2, 1,000 dev questions). Train questions never enter the evaluation set.

--blind_titles replaces every passage title by a synthetic id (doc_<n>) in the corpus, for the dense index and the graph alike, so no title match
can be used. --blind_mode head uses instead the leading phrase of the first sentence as the title (a head-entity proxy that needs no Wikipedia title).

Per pool and mode the script writes <out>/<pool>_<mode>/{summary.json, <system>.jsonl}: R@5, All@5 per question, indexing time, memory and latency.

    python -m kgrag.scale500k --pool_size 500000 --also_dev_pool --also_blind --cache ../cache --out ../results_h200/scale500k
"""
import argparse
import json
import os
import random
import re
import resource
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import text as T
from .controls import cached_dense, evaluate
from .retrievers import Hybrid2, _ptext
from .timing import bench, sync

SYSTEMS = {"dense": "dense", "titlelink": "title_only@1.0", "two_hop": "two_hop@0.3", "two_hop_gate": "two_hop@0.3+gate"}


def _records(split):
    from datasets import load_dataset

    return load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)


def _add_record(corp, r):
    ctx = r["context"]
    if isinstance(ctx, dict):
        ctx = list(zip(ctx["title"], ctx["sentences"]))
        sf = set(r["supporting_facts"]["title"])
    else:
        sf = {t for t, _ in r["supporting_facts"]}
    cand, sup = [], set()
    for title, sents in ctx:
        pid = corp.add(title, " ".join(s.strip() for s in sents))
        cand.append(pid)
        if title in sf:
            sup.add(pid)
    return dict(qid=str(r.get("id", r.get("_id"))), question=r["question"], answer=r["answer"], sup_pids=sup, cand_pids=cand, qtype=r.get("type", ""))


def build_pool(pool_size, eval_seed=2, n_eval=1000, train_seed=0):
    """Return (eval examples, passages). pool_size=0 -> dev paragraphs only."""
    dev = list(_records("validation"))
    corp = D.Corpus()
    all_dev = [_add_record(corp, r) for r in dev]
    rng = random.Random(eval_seed)
    order = list(dev)
    rng.shuffle(order)
    ev_ids = {str(r.get("id", r.get("_id"))) for r in order[:n_eval]}
    exs = [e for e in all_dev if e["qid"] in ev_ids]
    n_dev = len(corp.passages)
    n_train = 0
    if pool_size and pool_size > n_dev:
        tr = list(_records("train"))
        random.Random(train_seed).shuffle(tr)
        for r in tr:
            if len(corp.passages) >= pool_size:
                break
            _add_record(corp, r)
            n_train += 1
    print(f"pool: {len(corp.passages)} passages ({n_dev} from dev, {n_train} train questions added); {len(exs)} evaluation questions", flush=True)
    return exs, corp.passages, n_dev


_HEAD = re.compile(r"^(.{2,80}?)(?:\s*\(|\s+(?:is|was|are|were|has|had)\b|,)")


def blind(passages, mode="doc"):
    out = []
    for i, p in enumerate(passages):
        if mode == "head":
            m = _HEAD.match(p["text"])
            t = m.group(1) if m else " ".join(p["text"].split()[:4])
        else:
            t = f"doc_{i}"
        out.append({"title": t, "text": p["text"]})
    return out


def prewarm_spacy(passages, n_process):
    """Fill the spaCy memo with batch processing so that the graph build does not call spaCy one passage at a time."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    todo = [p["text"] for p in passages if p["text"] not in T._SPACY_MEMO]
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(todo, batch_size=256, n_process=n_process)):
        T._SPACY_MEMO[todo[i]] = [T._clean_entity(e.text) for e in doc.ents if len(e.text) > 1]
        if i and i % 50000 == 0:
            print(f"  spaCy {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"spaCy done: {len(todo)} passages in {time.time() - t0:.0f}s", flush=True)


def run_one(name, exs, passages, args, out_root):
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name}: {len(passages)} passages", flush=True)
    t0 = time.time()
    dense = cached_dense(passages, args.cache)
    sync()
    t_dense = time.time() - t0
    t0 = time.time()
    h = Hybrid2(dense, m=3, gamma=0.3, tb=0.5, ner="both", gate="titles").fit(passages)
    t_graph = time.time() - t0
    t0 = time.time()
    res = evaluate(h, exs, [0.3, 1.0], 0.5)
    t_eval = time.time() - t0
    rows = {}
    for label, key in SYSTEMS.items():
        recs = res[key]
        rows[label] = {"all_support": float(np.mean([r["all_support"] for r in recs])), "recall": float(np.mean([r["recall"] for r in recs])),
                       "all20": float(np.mean([r["all20"] for r in recs])), "gated": float(np.mean([r["gated"] for r in recs])), "n": len(recs)}
        with open(out / f"{label}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    # latency (same protocol as kgrag.timing): first --lat_warmup questions discarded, then --lat_repeats passes
    qs = [e["question"] for e in exs][: args.lat_n + args.lat_warmup]
    lat = {}
    h.gate, h.mode, h.gamma = "none", "two_hop", 0.3
    lat["dense"] = bench(lambda q: dense.search(q, 5), qs, args.lat_warmup, args.lat_repeats)
    lat["two_hop"] = bench(lambda q: h.search(q, 5), qs, args.lat_warmup, args.lat_repeats)
    h.gate = "titles"
    lat["two_hop_gate"] = bench(lambda q: h.search(q, 5), qs, args.lat_warmup, args.lat_repeats)
    h.gate, h.mode, h.gamma = "none", "title_only", 1.0
    lat["titlelink"] = bench(lambda q: h.search(q, 5), qs, args.lat_warmup, args.lat_repeats)
    meta = {"name": name, "n_passages": len(passages), "n_questions": len(exs), "seconds_dense_index": round(t_dense, 1), "seconds_graph_build": round(t_graph, 1),
            "seconds_eval": round(t_eval, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0), "embedding_mb": round(dense.E.nbytes / 1e6, 1),
            "graph_nnz": int(h.g.A.nnz), "graph_entities": int(h.g.m), "latency_questions": len(qs) - args.lat_warmup, "latency_repeats": args.lat_repeats}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows, "latency": lat}, indent=1))
    print(json.dumps(meta), flush=True)
    for k, v in rows.items():
        print(f"  {k:14s} R@5={100 * v['recall']:.1f} All@5={100 * v['all_support']:.1f} gated={100 * v['gated']:.1f}% | {lat[k]['ms_per_question_mean']} ms", flush=True)
    del h, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_size", type=int, default=500000, help="target number of passages (dev + train questions); 0 = dev only")
    ap.add_argument("--also_dev_pool", action="store_true", help="also run the dev-only pool (66,635 passages) for the 66k-vs-500k comparison")
    ap.add_argument("--blind_titles", action="store_true", help="replace every passage title by a synthetic id (run only this mode)")
    ap.add_argument("--blind_mode", default="doc", choices=["doc", "head"])
    ap.add_argument("--also_blind", action="store_true", help="run the standard and the blind-title mode in one process (shares spaCy and the pool)")
    ap.add_argument("--only", default=None, help="comma list of configurations to run, e.g. 500k_blind or 66k_std,66k_blind (default: all)")
    ap.add_argument("--eval_seed", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--lat_n", type=int, default=100)
    ap.add_argument("--lat_warmup", type=int, default=10)
    ap.add_argument("--lat_repeats", type=int, default=3)
    ap.add_argument("--spacy_procs", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--check_ids", default=None, help="results/eval_v2_seed2/dense.jsonl: assert the evaluation questions equal sample C")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    only = set(a.only.split(",")) if a.only else None
    if only and not any(x.startswith("500k") for x in only):
        a.pool_size = 0  # only the dev-only pool is needed: skip loading train and the 500k spaCy pass
    exs, passages, n_dev = build_pool(a.pool_size, a.eval_seed, a.n)
    if a.check_ids:
        ref = {json.loads(l)["qid"] for l in Path(a.check_ids).read_text().splitlines()}
        assert {e["qid"] for e in exs} == ref, "evaluation questions differ from sample C"
        print("evaluation questions identical to sample C", flush=True)
    prewarm_spacy(passages, a.spacy_procs)
    modes = ["blind"] if a.blind_titles else (["std", "blind"] if a.also_blind else ["std"])
    pools = [(a.pool_size, passages)]
    if a.also_dev_pool and a.pool_size:
        pools.append((0, passages[:n_dev]))  # dev paragraphs are added first, so passage ids and evaluation examples are identical
    for size, pas in pools:
        tag = f"{len(pas) // 1000}k"
        for mode in modes:
            ps = blind(pas, a.blind_mode) if mode == "blind" else pas
            # pool ids are the same, so the evaluation examples carry over; the dev pool is a prefix of the large pool
            name = f"{tag}_{'blind' + ('_head' if a.blind_mode == 'head' else '') if mode == 'blind' else 'std'}"
            if (only and name not in only) or (Path(a.out) / name / "summary.json").exists():
                print(f"skip {name}", flush=True)
                continue
            run_one(name, exs, ps, a, a.out)
    print("SCALE_DONE")


if __name__ == "__main__":
    main()
