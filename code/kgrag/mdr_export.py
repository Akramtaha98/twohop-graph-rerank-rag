"""Export a pooled corpus + questions for the official MDR code (facebookresearch/multihop_dense_retrieval).

  python -m kgrag.mdr_export --dataset hotpot --n 1000 --seed 0 --out ../mdr_in/hotpotA
writes  <out>/corpus.jsonl     one {"title","text"} per line, in passage-id order (format required by scripts/encode_corpus.py)
        <out>/questions.jsonl  {"_id","question","answer","type","sp":[{"title","text"}]}  (best-guess schema for eval_mhop_retrieval.py;
                               MDR's README does not document it -- compare with data/hotpot/hotpot_qas_val.json after download.sh and adapt if needed)
"""
import argparse
import json
from pathlib import Path

from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpot", "2wiki", "musique", "multihoprag"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exs, ps = D.load(a.dataset, a.path, a.n, a.seed)
    with open(out / "corpus.jsonl", "w") as f:
        for p in ps:
            f.write(json.dumps({"title": p["title"], "text": p["text"]}) + "\n")
    with open(out / "questions.jsonl", "w") as f:
        for e in exs:
            f.write(json.dumps({"_id": e["qid"], "question": e["question"], "answer": e["answer"], "type": e.get("qtype", ""),
                                "sp": [{"title": ps[i]["title"], "text": ps[i]["text"]} for i in sorted(e["sup_pids"])]}) + "\n")
    print(f"wrote {len(ps)} passages and {len(exs)} questions to {out}")


if __name__ == "__main__":
    main()
