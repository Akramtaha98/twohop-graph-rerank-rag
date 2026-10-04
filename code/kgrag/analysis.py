"""Per-question-type and win/loss analysis of the two-hop re-ranker versus dense retrieval.

    python -m kgrag.analysis --a results/hotpot --b results/eval_v2_seed1 --out paper
Writes paper/tables/errors.tex and paper/tables/numbers_analysis.tex. Needs the HotpotQA validation split (for question type).
"""
import argparse
import collections
import json
from pathlib import Path


def load_meta():
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return {r["id"]: r["type"] for r in ds}


def read(d, name):
    return {json.loads(l)["qid"]: json.loads(l) for l in Path(d, f"{name}.jsonl").read_text().splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    meta = load_meta()
    mac, rows = [], []
    for lab, d in (("A", a.a), ("B", a.b)):
        dn, h2 = read(d, "dense"), read(d, "hybrid2")
        qs = list(dn)
        cnt = collections.Counter()
        by = collections.defaultdict(lambda: [0, 0, 0])
        for q in qs:
            x, y = dn[q]["all_support"] == 1, h2[q]["all_support"] == 1
            cnt["both" if x and y else "dense_only" if x else "htwo_only" if y else "neither"] += 1
            by[meta[q]][0] += 1
            by[meta[q]][1] += dn[q]["all_support"]
            by[meta[q]][2] += h2[q]["all_support"]
        for k in ("both", "dense_only", "htwo_only", "neither"):
            mac.append(f"\\newcommand{{\\W{lab}{''.join(w.title() for w in k.split('_'))}}}{{{cnt[k]}}}")
        for t, (n, sd, sh) in sorted(by.items()):
            T = t.title()
            mac += [f"\\newcommand{{\\T{lab}{T}N}}{{{n}}}", f"\\newcommand{{\\T{lab}{T}Dense}}{{{100*sd/n:.1f}}}", f"\\newcommand{{\\T{lab}{T}Two}}{{{100*sh/n:.1f}}}"]
            rows.append((lab, t, n, 100 * sd / n, 100 * sh / n))
        if lab == "A":
            for k, cond in (("HOnly", lambda q: not dn[q]["all_support"] and h2[q]["all_support"]), ("DOnly", lambda q: dn[q]["all_support"] and not h2[q]["all_support"]),
                            ("Both", lambda q: dn[q]["all_support"] and h2[q]["all_support"]), ("Neither", lambda q: not dn[q]["all_support"] and not h2[q]["all_support"])):
                qq = [q for q in qs if cond(q) and "em" in dn[q] and "em" in h2[q]]
                if qq:
                    mac += [f"\\newcommand{{\\EM{k}Dense}}{{{100*sum(dn[q]['em'] for q in qq)/len(qq):.1f}}}",
                            f"\\newcommand{{\\EM{k}Two}}{{{100*sum(h2[q]['em'] for q in qq)/len(qq):.1f}}}"]
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{All@5 (\\%) by HotpotQA question type, dense retrieval versus two-hop re-ranking, on samples A and B.}",
             "\\label{tab:bytype}", "\\begin{tabular}{llrrr}", "\\toprule", "Sample & Type & $n$ & Dense & Two-hop \\\\", "\\midrule"]
    for lab, t, n, sd, sh in rows:
        lines.append(f"{lab} & {t} & {n} & {sd:.1f} & {sh:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = Path(a.out, "tables")
    out.mkdir(parents=True, exist_ok=True)
    (out / "errors.tex").write_text("\n".join(lines))
    (out / "numbers_analysis.tex").write_text("\n".join(mac))
    print("ANALYSIS_WRITTEN")


if __name__ == "__main__":
    main()
