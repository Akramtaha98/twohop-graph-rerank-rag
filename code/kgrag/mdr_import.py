"""Convert the output of MDR's eval_mhop_retrieval.py (--save-path) into kgrag per-question records.

  python -m kgrag.mdr_import --dataset hotpot --n 1000 --seed 0 --mdr_out mdr_out/hotpotA.json --out ../results_jis/hotpotA [--id2doc path]

MDR returns, per question, ranked passage CHAINS (first hop, second hop). We flatten chains in rank order, drop duplicates and keep the first k
passages, which is the passage list a downstream reader would see. The output schema of MDR is not documented in its README: this script looks for
a list of chains under the keys below and for chain items given as {"title","text"}, [title,text] or ids (needs --id2doc). If nothing matches it
prints the keys it found; adapt CHAIN_KEYS / item handling then.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M

CHAIN_KEYS = ("candidate_chains", "chains", "retrieved", "top_chains")


def _items(path):
    t = Path(path).read_text()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, list) else list(obj.values())
    except json.JSONDecodeError:
        return [json.loads(l) for l in t.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpot", "2wiki", "musique", "multihoprag"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mdr_out", required=True)
    ap.add_argument("--id2doc", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="mdr")
    a = ap.parse_args()
    exs, ps = D.load(a.dataset, a.path, a.n, a.seed)
    key_tt = {(p["title"], p["text"]): i for i, p in enumerate(ps)}
    by_title = {}
    for i, p in enumerate(ps):
        by_title.setdefault(p["title"], []).append(i)
    id2doc = json.loads(Path(a.id2doc).read_text()) if a.id2doc and Path(a.id2doc).exists() else None
    by_id = {}
    for it in _items(a.mdr_out):
        by_id[str(it.get("_id", it.get("id", "")))] = it

    def to_pid(x):
        if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
            x = id2doc[str(x)]
        if isinstance(x, dict):
            t, tx = x.get("title"), x.get("text")
        else:
            t, tx = x[0], (x[1] if len(x) > 1 else None)
        if (t, tx) in key_tt:
            return key_tt[(t, tx)]
        c = by_title.get(t, [])
        return c[0] if len(c) == 1 else None

    recs, bad = [], 0
    for ex in exs:
        it = by_id.get(str(ex["qid"]))
        chains = next((it[k] for k in CHAIN_KEYS if it and k in it), None)
        if chains is None:
            raise SystemExit(f"no chain list for qid {ex['qid']}; keys found: {sorted(it) if it else 'record missing'}")
        pids = []
        for ch in chains:
            for x in ch:
                p = to_pid(x)
                if p is None:
                    bad += 1
                elif p not in pids:
                    pids.append(p)
            if len(pids) >= a.k:
                break
        pids = pids[: a.k]
        recs.append({"qid": ex["qid"], "recall": M.recall_at_k(pids, ex["sup_pids"]), "all_support": M.all_support_at_k(pids, ex["sup_pids"]),
                     "qtype": ex.get("qtype", ""), "gated": False, "ranked": pids})
    if bad:
        print(f"WARNING: {bad} chain items could not be mapped to passages")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.label}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    sp = out / "summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {"systems": {}}
    summ["systems"][a.label] = {k: float(np.mean([r[k] for r in recs])) for k in ("recall", "all_support")}
    sp.write_text(json.dumps(summ, indent=2))
    print(a.label, summ["systems"][a.label])


if __name__ == "__main__":
    main()
