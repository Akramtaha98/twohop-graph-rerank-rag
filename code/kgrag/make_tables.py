"""Turn results/*/summary.json (+ per-question jsonl) into LaTeX tables, macros and a figure.

    python -m kgrag.make_tables --runs "HotpotQA=results/hotpot" "MuSiQue=results/musique" --out paper
Every number in the paper's tables, figure and results macros comes from this script.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import metrics as M

MAIN = ["none", "bm25", "dense", "graph", "hybrid", "hybrid2", "hybrid_lora", "oracle"]
PRETTY = {"none": "No retrieval", "bm25": "BM25", "dense": "Dense", "graph": "Graph-PPR (entity seeds only)",
          "hybrid": "PPR hybrid (v1)", "hybrid_lora": "KG-guided hybrid + LoRA", "hybrid2": "Two-hop graph re-ranking (ours)",
          "abl2_no_title": "w/o title-link bonus", "abl2_one_seed": "first hop from top-1 only", "abl2_regex_ner": "regex entities only", "oracle": "Gold passages (upper bound)",
          "abl_no_entity_seed": "w/o entity seeds", "abl_no_dense_seed": "w/o dense seeds",
          "abl_no_fusion": "w/o dense-score fusion", "abl_no_idf": "w/o IDF edge weights", "abl_alpha05": "restart $\\alpha=0.5$"}
COLS = [("recall", "R@5", True), ("all_support", "All@5", True), ("em", "EM", True), ("f1", "F1", True), ("unsupported", "Unsup.$\\downarrow$", False)]
MAC = {"recall": "Recall", "all_support": "AllSupp", "em": "EM", "f1": "FOne", "unsupported": "Unsupp", "support": "Support"}
LET = lambda s: re.sub(r"[^A-Za-z]", "", s.replace("2", "Two").replace("0", "Zero").replace("1", "One"))  # noqa: E731


def fmt(x):
    return f"{100 * x:.1f}"


def load(runs):
    out = {}
    for item in runs:
        label, path = item.split("=")
        out[label] = (Path(path), json.loads((Path(path) / "summary.json").read_text()))
    return out


def table(runs, systems, caption, label):
    first = next(iter(runs.values()))[1]["systems"]
    cols = [c for c in COLS if any(c[0] in s for s in first.values())]
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Dataset & System & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for di, (dname, (_, summ)) in enumerate(runs.items()):
        sysd = {s: summ["systems"][s] for s in systems if s in summ["systems"]}
        best = {}
        for key, _, hi in cols:
            cand = {s: v[key] for s, v in sysd.items() if key in v and s not in {"oracle"}}
            if cand:
                best[key] = (max if hi else min)(cand.values())
        for si, (s, v) in enumerate(sysd.items()):
            cells = []
            for key, _, hi in cols:
                if key not in v:
                    cells.append("--")
                    continue
                txt = fmt(v[key])
                cells.append(f"\\textbf{{{txt}}}" if key in best and abs(v[key] - best[key]) < 1e-12 and s != "oracle" else txt)
            head = f"\\multirow{{{len(sysd)}}}{{*}}{{{dname}}}" if si == 0 else ""
            lines.append(f"{head} & {PRETTY.get(s, s)} & " + " & ".join(cells) + " \\\\")
        if di < len(runs) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def significance(runs, a="hybrid2", b="dense"):
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{Paired bootstrap (10\\,000 resamples) of {PRETTY[a]} versus {PRETTY[b]}: mean difference in percentage points, 95\\% CI and two-sided $p$.}}",
             "\\label{tab:signif}", "\\begin{tabular}{llrrr}", "\\toprule", "Dataset & Metric & $\\Delta$ & 95\\% CI & $p$ \\\\", "\\midrule"]
    for dname, (path, summ) in runs.items():
        for key, lab, hi in COLS:
            fa, fb = path / f"{a}.jsonl", path / f"{b}.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            ra = [json.loads(l) for l in fa.read_text().splitlines()]
            rb = [json.loads(l) for l in fb.read_text().splitlines()]
            if key not in ra[0]:
                continue
            d, lo, hi_, p = M.paired_bootstrap([r[key] for r in ra], [r[key] for r in rb])
            ps = "<0.001" if p < 0.001 else f"{p:.3f}"
            lines.append(f"{dname} & {lab} & {100 * d:+.1f} & [{100 * lo:+.1f}, {100 * hi_:+.1f}] & {ps} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def retr_table(runs):
    """Retrieval-only table: systems x (R@5, All@5) for each independent sample."""
    systems = ["bm25", "dense", "hybrid", "hybrid2", "abl2_no_title", "abl2_one_seed", "abl2_regex_ner", "oracle"]
    names = list(runs)
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Retrieval on two independent samples of 1\\,000 HotpotQA questions (\\%). Sample~A was also used to evaluate the earlier PPR variant; sample~B was used only once, for the final configuration. Best non-oracle value in bold.}",
             "\\label{tab:retr2}", "\\begin{tabular}{l" + "rr" * len(names) + "}", "\\toprule",
             "System & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in names) + " \\\\",
             "& " + " & ".join("R@5 & All@5" for _ in names) + " \\\\", "\\midrule"]
    best = {}
    for n in names:
        for key in ("recall", "all_support"):
            best[(n, key)] = max(runs[n][1]["systems"][s][key] for s in systems if s != "oracle" and s in runs[n][1]["systems"])
    for sname in systems:
        cells = []
        for n in names:
            v = runs[n][1]["systems"].get(sname)
            for key in ("recall", "all_support"):
                if v is None:
                    cells.append("--")
                else:
                    t = fmt(v[key])
                    cells.append(f"\\textbf{{{t}}}" if sname != "oracle" and abs(v[key] - best[(n, key)]) < 1e-12 else t)
        lines.append(f"{PRETTY.get(sname, sname)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def macros(runs):
    out = []
    for dname, (_, summ) in runs.items():
        D = LET(dname)
        out.append(f"\\newcommand{{\\nQ{D}}}{{{summ['n_questions']}}}")
        out.append(f"\\newcommand{{\\nP{D}}}{{{summ['n_passages']}}}")
        for s, v in summ["systems"].items():
            S = LET(s.replace("_", " ").title().replace(" ", ""))
            for k, val in v.items():
                if k in MAC:
                    out.append(f"\\newcommand{{\\R{D}{S}{MAC[k]}}}{{{fmt(val)}}}")
    return "\n".join(out)


def figure(runs, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "f1" if any("f1" in v for _, s in runs.values() for v in s["systems"].values()) else "all_support"
    # only plot systems that have the metric in every run (e.g. graph-only has no F1 when the LLM is run on key systems only)
    systems = [s for s in ["bm25", "dense", "hybrid", "hybrid2"]
               if all(metric in summ["systems"].get(s, {}) for _, (_, summ) in runs.items())]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        vals = [100 * summ["systems"][s][metric] for _, (_, summ) in runs.items() if s in summ["systems"]]
        ax.bar(np.arange(len(vals)) + i * w, vals, w, label=PRETTY[s].split(" (")[0])
    ax.set_xticks(np.arange(len(runs)) + 0.4 - w / 2)
    ax.set_xticklabels(list(runs.keys()))
    ax.set_ylabel("F1 (%)" if metric == "f1" else "All supporting passages in top-5 (%)")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval", nargs="*", default=None, help="label=dir of retrieval-only samples (adds retr2 table and macros)")
    ap.add_argument("--signif", nargs="*", default=None, help="label=dir pairs for the paired bootstrap table (default: --runs)")
    ap.add_argument("--tuned2", default=None, help="results/tuned2.json (adds v2 hyper-parameter macros)")
    ap.add_argument("--tuned", default=None, help="results/tuned.json (adds hyper-parameter macros)")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    runs = load(a.runs)
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "tables/main_results.tex").write_text(table(runs, MAIN, "Main results (\\%). R@5: recall of supporting passages in the top-5; All@5: all supporting passages retrieved; EM/F1: answer quality; Unsup.: share of answers not lexically supported by the retrieved evidence (lower is better). Best non-oracle value in bold.", "tab:main"))
    (out / "tables/ablation.tex").write_text(table(runs, ["hybrid"] + [s for s in PRETTY if s.startswith("abl_")], "Ablation of the earlier PPR hybrid (\\%); the PPR variant was not run with the generator except in its full form.", "tab:ablation"))
    sig_runs = load(a.signif) if a.signif else runs
    (out / "tables/signif.tex").write_text(significance(sig_runs))
    ev = load(a.eval) if a.eval else None
    if ev:
        (out / "tables/retr2.tex").write_text(retr_table(ev))
    mac = macros(runs)
    if a.tuned:
        t = json.loads(Path(a.tuned).read_text())
        b = t["best"]
        mac += (f"\n\\newcommand{{\\tAlpha}}{{{b['alpha']}}}\\newcommand{{\\tLam}}{{{b['lam']}}}\\newcommand{{\\tBeta}}{{{b['beta']}}}"
                f"\\newcommand{{\\tTuneN}}{{{t['n']}}}\\newcommand{{\\tTuneAll}}{{{fmt(b['all_support'])}}}")
    if ev:
        mac += "\n" + macros(ev)
    if a.tuned2:
        t2 = json.loads(Path(a.tuned2).read_text())
        b2 = t2["best"]
        mac += (f"\n\\newcommand{{\\tTwoM}}{{{b2['m']}}}\\newcommand{{\\tTwoGamma}}{{{b2['gamma']}}}\\newcommand{{\\tTwoTb}}{{{b2['tb']}}}"
                f"\\newcommand{{\\tTwoNer}}{{{b2['ner']}}}\\newcommand{{\\tTwoAll}}{{{fmt(b2['all_support'])}}}"
                f"\\newcommand{{\\tTwoDense}}{{{fmt(t2['dense_baseline'])}}}")
    (out / "tables/numbers.tex").write_text(mac)
    figure(runs, out / "figs/main.pdf")
    print("TABLES_WRITTEN")


if __name__ == "__main__":
    main()
