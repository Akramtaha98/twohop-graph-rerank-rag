"""Run retrieval + generation for a set of systems and write per-question and summary results.

Example (offline smoke test):
    python -m kgrag.run --dataset toy --n 200 --generator extractive --out results/toy
Example (real):
    python -m kgrag.run --dataset hotpot --n 1000 --dense bge --generator hf --out results/hotpot
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M
from .generate import Extractive, HFGenerator, build_context
from .retrievers import BM25, Dense, GraphPPR, Hybrid, Hybrid2

def default_systems(best=None, best2=None):
    """Baselines, the proposed hybrid and its ablations. `best` overrides hyper-parameters (from tune.py)."""
    base = dict(alpha=0.15, lam=0.5, beta=0.5)
    base.update(best or {})

    def spec(label, **change):
        kw = {**base, **change}
        return f"{label}:hybrid:" + ",".join(f"{k}={v}" for k, v in kw.items())

    h2 = None
    if best2:
        b2 = dict(best2)
        def spec2(label, **change):
            kw = {**b2, **change}
            return f"{label}:hybrid2:" + ",".join(f"{k}={v}" for k, v in kw.items())
        h2 = [spec2("hybrid2"), spec2("abl2_no_title", tb=0), spec2("abl2_one_seed", m=1), spec2("abl2_regex_ner", ner="regex")]
    return ["none", "bm25", "dense", "graph", spec("hybrid"), *(h2 or []), spec("abl_no_entity_seed", lam=0), spec("abl_no_dense_seed", lam=1),
            spec("abl_no_fusion", beta=0), spec("abl_no_idf", idf=0), spec("abl_alpha05", alpha=0.5), "oracle"]


def parse_spec(spec):
    """'label:kind:key=val,key=val' or 'kind'. Returns (label, kind, kwargs)."""
    parts = spec.split(":")
    if len(parts) == 1:
        return parts[0], parts[0], {}
    label, kind = parts[0], parts[1]
    kw = {}
    for kv in parts[2:]:
        for item in filter(None, kv.split(",")):
            k, v = item.split("=")
            kw[k] = v if k == "ner" else float(v) if k in {"alpha", "lam", "beta", "temp", "max_df_frac", "gamma", "tb"} else int(v)
    if "idf" in kw:
        kw["idf"] = bool(kw["idf"])
    return label, kind, kw


def build(kind, kw, dense, passages, args):
    if kind == "bm25":
        return BM25().fit(passages)
    if kind == "dense":
        return dense
    if kind == "graph":
        return GraphPPR(ner=args.ner, **kw).fit(passages)
    if kind == "hybrid":
        return Hybrid(dense, ner=args.ner, **kw).fit(passages)
    if kind == "hybrid2":
        return Hybrid2(dense, **{"ner": args.ner, **kw}).fit(passages)
    return None  # none / oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy", choices=["toy", "hotpot", "2wiki", "musique"])
    ap.add_argument("--path", default=None, help="local data file (required for 2wiki/musique)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dense", default="tfidf", choices=["tfidf", "bge"])
    ap.add_argument("--ner", default="regex", choices=["regex", "spacy", "both"])
    ap.add_argument("--generator", default="extractive", choices=["extractive", "hf", "none"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--gen_systems", nargs="*", default=None, help="run the LLM only for these system labels (others: retrieval metrics only)")
    ap.add_argument("--append", action="store_true", help="merge into an existing summary.json in --out")
    ap.add_argument("--tuned", default=None, help="results/tuned.json from kgrag.tune")
    ap.add_argument("--tuned2", default=None, help="results/tuned2.json from kgrag.tune2 (adds the v2 method + its ablations)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exs, passages = D.load(args.dataset, args.path, args.n, args.seed)
    print(f"{args.dataset}: {len(exs)} questions, {len(passages)} pooled passages")

    dense = Dense(args.dense).fit(passages)
    gen = None
    if args.generator == "extractive":
        gen = Extractive()
    elif args.generator == "hf":
        gen = HFGenerator(args.model, args.adapter, load_4bit=args.load_4bit)

    best = json.loads(Path(args.tuned).read_text())["best"] if args.tuned else None
    if best:
        best = {k: best[k] for k in ("alpha", "lam", "beta")}
    best2 = json.loads(Path(args.tuned2).read_text())["best"] if args.tuned2 else None
    systems = args.systems or default_systems(best, best2)
    gen_for = set(args.gen_systems) if args.gen_systems else None
    summary = {"config": vars(args), "systems_run": systems, "n_questions": len(exs), "n_passages": len(passages), "systems": {}}
    for spec in systems:
        label, kind, kw = parse_spec(spec)
        r = build(kind, kw, dense, passages, args)
        recs, t0 = [], time.time()
        for ex in exs:
            bridges = {}
            if kind == "none":
                pids = []
            elif kind == "oracle":
                pids = sorted(ex["sup_pids"])
            else:
                pids, extra = r.search(ex["question"], args.k)
                bridges = extra.get("bridges", {})
            rec = {
                "qid": ex["qid"],
                "recall": M.recall_at_k(pids, ex["sup_pids"]),
                "all_support": M.all_support_at_k(pids, ex["sup_pids"]),
            }
            if gen is not None and (gen_for is None or label in gen_for):
                ctx = build_context(passages, pids, bridges, chain=(kind in {"graph", "hybrid"}))
                pred = gen(ex["question"], ctx)
                rec.update(pred=pred, em=M.exact_match(pred, ex["answer"]), f1=M.f1(pred, ex["answer"]),
                           support=M.support_rate(pred, ctx), unsupported=M.unsupported(pred, ctx))
            recs.append(rec)
        with open(out / f"{label}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        agg = {k: float(np.mean([x[k] for x in recs])) for k in recs[0] if k not in {"qid", "pred"}}
        agg["seconds"] = round(time.time() - t0, 1)
        summary["systems"][label] = agg
        print(label, {k: round(v, 3) for k, v in agg.items()})
    if args.append and (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text())
        old["systems"].update(summary["systems"])
        summary = old
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
