"""Official HippoRAG 2 (pip install hipporag) on a pooled corpus, written in the same per-question format as kgrag.run.

1) serve an LLM for OpenIE/indexing:  vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 8192 --port 8000
2) python -m kgrag.hipporag_baseline --dataset hotpot --n 1000 --seed 0 --out ../results_jis/hotpotA --llm_model Qwen/Qwen2.5-7B-Instruct

API per the HippoRAG README: HippoRAG(save_dir, llm_model_name, llm_base_url, embedding_model_name); index(docs=[str]).
retrieve(queries=[...], num_to_retrieve=k) is NOT documented in the README; it exists in the source as in the original repo and returns a list of
QuerySolution (.docs, .doc_scores) -- if your installed version differs, edit the block marked API below and nothing else.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpot", "2wiki", "musique", "multihoprag"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="hipporag2")
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    ap.add_argument("--embed_model", default="nvidia/NV-Embed-v2")
    ap.add_argument("--save_dir", default=None)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exs, passages = D.load(a.dataset, a.path, a.n, a.seed)
    docs = [f"{p['title']}\n{p['text']}" for p in passages]  # HippoRAG's own corpus convention: title + newline + text
    doc2pid = {d: i for i, d in enumerate(docs)}
    print(f"{a.dataset}: {len(exs)} questions, {len(docs)} passages")

    from hipporag import HippoRAG  # API: import

    save_dir = a.save_dir or str(out / "hipporag_index")
    hr = HippoRAG(save_dir=save_dir, llm_model_name=a.llm_model, llm_base_url=a.llm_base_url, embedding_model_name=a.embed_model)  # API: constructor
    t0 = time.time()
    hr.index(docs=docs)  # API: indexing (LLM OpenIE over every passage; this is the slow part)
    t_index = time.time() - t0
    t0 = time.time()
    sols = hr.retrieve(queries=[e["question"] for e in exs], num_to_retrieve=a.k)  # API: retrieval
    if isinstance(sols, tuple):  # some versions return (solutions, metrics)
        sols = sols[0]
    t_ret = time.time() - t0

    recs, lost = [], 0
    for ex, s in zip(exs, sols):
        pids = []
        for d in s.docs[: a.k]:  # API: result field
            if d in doc2pid:
                pids.append(doc2pid[d])
            else:
                lost += 1
        recs.append({"qid": ex["qid"], "recall": M.recall_at_k(pids, ex["sup_pids"]), "all_support": M.all_support_at_k(pids, ex["sup_pids"]),
                     "qtype": ex.get("qtype", ""), "gated": False, "ranked": pids})
    if lost:
        print(f"WARNING: {lost} returned docs could not be mapped back to passages (string normalisation?)")
    with open(out / f"{a.label}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    agg = {k: float(np.mean([r[k] for r in recs])) for k in ("recall", "all_support")}
    agg.update(seconds=round(t_ret, 1), index_seconds=round(t_index, 1), llm=a.llm_model, embed=a.embed_model)
    sp = out / "summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {"systems": {}, "n_questions": len(exs), "n_passages": len(passages)}
    summ["systems"][a.label] = agg
    sp.write_text(json.dumps(summ, indent=2))
    print(a.label, agg)


if __name__ == "__main__":
    main()
