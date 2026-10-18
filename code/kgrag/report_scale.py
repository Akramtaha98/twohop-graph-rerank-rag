"""Tables for the 500k-pool and blind-title runs (LaTeX + Markdown + macros).
Usage: python -m kgrag.report_scale --res ../results/scale500k --out ../results/scale500k/tables"""
import argparse
import json
from pathlib import Path

import numpy as np

SYS = [("dense", "Dense (BGE-small)"), ("titlelink", "TitleLink only"), ("two_hop", "Two-hop"), ("two_hop_gate", "Two-hop + gate")]
CODE = {"dense": "Dense", "titlelink": "Tl", "two_hop": "Two", "two_hop_gate": "Gate"}


def rd(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = max(float((np.abs(m - m.mean()) >= abs(d.mean())).mean()), 1 / n)
    return 100 * d.mean(), 100 * lo, 100 * hi, p


def f1(x):
    return f"{100 * x:.1f}"


def load(R, name):
    d = R / name
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    s["recs"] = {k: rd(d / f"{k}.jsonl") for k, _ in SYS}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p.name for p in R.iterdir() if p.is_dir() and (p / "summary.json").exists() and p.name != "tables")
    S = {d: load(R, d) for d in dirs}
    stds = sorted((d for d in dirs if d.endswith("_std")), key=lambda d: S[d]["meta"]["n_passages"])
    small, big = (stds[0], stds[-1]) if len(stds) > 1 else (None, stds[0] if stds else None)
    mac, md = [], []

    # ---- Table 1: 66k vs 500k
    cols = [(d, S[d]["meta"]["n_passages"]) for d in (small, big) if d]
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Corpus scale on the untouched sample C (1\\,000 HotpotQA dev questions, BGE-small): the dev-only pool against a pool enlarged with HotpotQA train paragraphs. "
         "Settings are the frozen ones. $\\Delta$ is the change from the small to the large pool; latency is mean ms per question on one H200.}",
         "\\label{tab:scale500}", "\\begin{tabular}{l" + "rrr" * len(cols) + "}", "\\toprule",
         "System & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n:,} passages}}".replace(",", "{,}") for _, n in cols) + " \\\\",
         " & " + " & ".join("All@5 & R@5 & ms" for _ in cols) + " \\\\", "\\midrule"]
    md += ["| System | " + " | ".join(f"{n:,}: All@5 | R@5 | ms" for _, n in cols) + " | ΔAll@5 [95% CI] |", "|---|" + "---|" * (3 * len(cols) + 1)]
    for k, lab in SYS:
        cells, mcells = [], []
        for d, _ in cols:
            r, t = S[d]["rows"][k], S[d]["latency"][k]["ms_per_question_mean"]
            cells += [f1(r["all_support"]), f1(r["recall"]), f"{t:.0f}"]
            mcells += [f1(r["all_support"]), f1(r["recall"]), f"{t:.0f}"]
            tag = "S" if d == small else "L"
            mac += [f"\\newcommand{{\\Sc{tag}{CODE[k]}All}}{{{f1(r['all_support'])}}}", f"\\newcommand{{\\Sc{tag}{CODE[k]}Rec}}{{{f1(r['recall'])}}}", f"\\newcommand{{\\Sc{tag}{CODE[k]}Ms}}{{{t:.0f}}}"]
        dl = ""
        if len(cols) == 2:
            m_, lo, hi, p = boot([r["all_support"] for r in S[cols[1][0]]["recs"][k]], [r["all_support"] for r in S[cols[0][0]]["recs"][k]])
            dl = f"{m_:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            mac.append(f"\\newcommand{{\\ScD{CODE[k]}}}{{{m_:+.1f}}}")
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        md.append(f"| {lab} | " + " | ".join(mcells) + f" | {dl} |")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "scale500_main.tex").write_text("\n".join(L) + "\n")

    # ---- gains over dense at each scale
    md += ["", "Paired bootstrap, All@5 gain over Dense (10,000 resamples):", "", "| Pool | System | Δ [95% CI] | p |", "|---|---|---|---|"]
    for d, n in cols:
        for k, lab in SYS[1:]:
            m_, lo, hi, p = boot([r["all_support"] for r in S[d]["recs"][k]], [r["all_support"] for r in S[d]["recs"]["dense"]])
            md.append(f"| {n:,} | {lab} | {m_:+.1f} [{lo:+.1f}, {hi:+.1f}] | {p:.4f} |")
            mac.append(f"\\newcommand{{\\ScG{'S' if d == small else 'L'}{CODE[k]}}}{{{m_:+.1f}}}")

    # ---- Table 2: blind titles (all pools present)
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Blind-title control on sample C. Every passage title is replaced by a synthetic identifier (\\texttt{doc\\_$i$}) in the dense index and in the graph, so no title or title--question match can be used. "
         "All@5 (\\%) with the real titles and with blind titles; $\\Delta$ is blind minus real.}",
         "\\label{tab:blind}", "\\begin{tabular}{llrrr}", "\\toprule", "Pool & System & Real titles & Blind titles & $\\Delta$ [95\\% CI] \\\\", "\\midrule"]
    md += ["", "Blind-title control (All@5):", "", "| Pool | System | Real | Blind | Δ [95% CI] |", "|---|---|---|---|---|"]
    for d, n in cols:
        bd = next((x for x in dirs if x.startswith(d.split("_")[0] + "_blind")), None)
        if not bd:
            continue
        for k, lab in SYS:
            re_, bl = S[d]["rows"][k]["all_support"], S[bd]["rows"][k]["all_support"]
            m_, lo, hi, p = boot([r["all_support"] for r in S[bd]["recs"][k]], [r["all_support"] for r in S[d]["recs"][k]])
            L.append(f"{n:,} & {lab} & {f1(re_)} & {f1(bl)} & {m_:+.1f} [{lo:+.1f}, {hi:+.1f}] \\\\".replace(f"{n:,} &", f"{n:,}".replace(",", "{,}") + " &", 1))
            md.append(f"| {n:,} | {lab} | {f1(re_)} | {f1(bl)} | {m_:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
            mac.append(f"\\newcommand{{\\Sb{'S' if d == small else 'L'}{CODE[k]}}}{{{f1(bl)}}}")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "scale500_blind.tex").write_text("\n".join(L) + "\n")

    # ---- cost
    if big:
        m = S[big]["meta"]
        for key, name in (("seconds_dense_index", "Idx"), ("seconds_graph_build", "Graph"), ("peak_rss_mb", "Rss"), ("graph_nnz", "Nnz"), ("graph_entities", "Ents"), ("embedding_mb", "Emb"), ("n_passages", "N")):
            mac.append(f"\\newcommand{{\\ScL{name}}}{{{m[key]:,.0f}}}".replace(",", "{,}") if key in ("graph_nnz", "graph_entities", "n_passages") else f"\\newcommand{{\\ScL{name}}}{{{m[key]:.0f}}}")
        md += ["", f"Large pool cost: dense index {m['seconds_dense_index']} s, graph build {m['seconds_graph_build']} s, peak RSS {m['peak_rss_mb']:.0f} MB, embeddings {m['embedding_mb']} MB, graph nnz {m['graph_nnz']:,}."]
    (out / "scale500_tables.md").write_text("\n".join(md) + "\n")
    (out / "numbers_scale500.tex").write_text("\n".join(mac) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
