"""Publication figures from the stored per-question files (vector PDF + 300 dpi PNG).
    python -m kgrag.make_figures --res ../mac/results --out ../paper/figs
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M

C = {"dense": "#7f7f7f", "spriglike": "#a6cee3", "linearlike": "#1f78b4", "hybrid2_gate": "#d95f02", "titlelink": "#1b9e77", "ce_dense": "#7570b3", "ce_hybrid2_gate": "#e7298a"}
L = {"dense": "Dense", "spriglike": "SPRIG-style", "linearlike": "LinearRAG-style", "hybrid2_gate": "Two-hop + gate", "titlelink": "TitleLink", "ce_dense": "Cross-encoder (dense top-20)", "ce_hybrid2_gate": "Cross-encoder (two-hop top-20)"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "pdf.fonttype": 42})


def rd(p):
    p = Path(p)
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else None


def ci(vals, B=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return 100 * v.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = [("HotpotQA A", R / "eval_v2_seed0"), ("HotpotQA B", R / "eval_v2_seed1"), ("2WikiMultiHopQA", R / "eval_2wiki"), ("MuSiQue", R / "eval_musique")]
    sysl = ["dense", "spriglike", "linearlike", "hybrid2_gate", "titlelink", "ce_dense", "ce_hybrid2_gate"]

    # Fig 1: All@5 by system and set, bootstrap 95% intervals
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    w = 0.115
    for i, s in enumerate(sysl):
        for j, (nm, d) in enumerate(sets):
            r = rd(d / f"{s}.jsonl")
            m, lo, hi = ci([x["all_support"] for x in r.values()])
            ax.bar(j + (i - 3) * w, m, w * 0.92, color=C[s], label=L[s] if j == 0 else None, yerr=[[m - lo], [hi - m]], error_kw={"lw": 0.7, "capsize": 1.2})
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([n for n, _ in sets])
    ax.set_ylabel("All@5 (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
    ax.yaxis.grid(True, lw=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "fig_all5")

    # Fig 2: paired differences to dense retrieval (forest plot)
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    cmp = ["spriglike", "linearlike", "hybrid2_gate", "titlelink", "ce_dense", "ce_hybrid2_gate"]
    for ax, (nm, d) in zip(axs, sets):
        base = rd(d / "dense.jsonl")
        for k, s in enumerate(cmp):
            r = rd(d / f"{s}.jsonl")
            q = sorted(set(r) & set(base))
            dd, lo, hi, _ = M.paired_bootstrap([r[i]["all_support"] for i in q], [base[i]["all_support"] for i in q])
            ax.errorbar(100 * dd, k, xerr=[[100 * (dd - lo)], [100 * (hi - dd)]], fmt="o", color=C[s], ms=4, lw=1.2, capsize=2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(nm, fontsize=8)
        ax.set_yticks(range(len(cmp)))
        ax.set_yticklabels([L[s] for s in cmp], fontsize=7)
        ax.invert_yaxis()
        ax.xaxis.grid(True, lw=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("All@5 minus dense (points)", fontsize=7)
    save(fig, out, "fig_paired")

    # Fig 3: candidate discovery against ordering
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (nm, dd) in zip(axs, [("HotpotQA A", R / "controls_ce" / "hotpot_0"), ("HotpotQA B", R / "controls_ce" / "hotpot_1")]):
        groups = []
        for pl, pn in (("D", "Dense top-20 pool"), ("H", "Two-hop top-20 pool")):
            r = {o: rd(dd / f"pool{pl}_{o}.jsonl") for o in ("dense", "two_hop", "ce")}
            groups.append((pn, [100 * np.mean([x["all_support"] for x in r[o].values()]) for o in ("dense", "two_hop", "ce")], 100 * np.mean([x["all20"] for x in r["dense"].values()])))
        for g, (pn, v, a20) in enumerate(groups):
            for o, (val, col, lab) in enumerate(zip(v, ["#7f7f7f", "#d95f02", "#7570b3"], ["Dense order", "Two-hop order", "Cross-encoder order"])):
                ax.bar(g + (o - 1) * 0.25, val, 0.23, color=col, label=lab if g == 0 else None)
                ax.text(g + (o - 1) * 0.25, val + 0.8, f"{val:.1f}", ha="center", fontsize=6.5)
            ax.plot([g - 0.4, g + 0.4], [a20, a20], color="k", lw=1.0, ls="--", label="All@20 of the pool" if g == 0 else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
        ax.set_title(nm, fontsize=8)
        ax.set_ylim(60, 100)
        ax.yaxis.grid(True, lw=0.3, alpha=0.6)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("All@5 (%)")
    axs[0].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
    save(fig, out, "fig_pool")

    # Fig 4: answers (EM, F1) on both samples
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (nm, d, ss) in zip(axs, [("HotpotQA A", R / "answersA_gpu", ["dense", "spriglike", "linearlike", "hybrid2_gate", "ce_dense", "titlelink"]), ("HotpotQA B", R / "eval_v2_seed1", ["dense", "hybrid2_gate", "ce_dense", "titlelink"])]):
        for k, s in enumerate(ss):
            r = rd(d / f"{s}.jsonl")
            for o, m in enumerate(("em", "f1")):
                v, lo, hi = ci([x[m] for x in r.values()])
                ax.bar(k + (o - 0.5) * 0.38, v, 0.36, color=C[s], alpha=1.0 if m == "em" else 0.55, yerr=[[v - lo], [hi - v]], error_kw={"lw": 0.7, "capsize": 1.2})
        ax.set_xticks(range(len(ss)))
        ax.set_xticklabels([L[s].replace(" (dense top-20)", "").replace("Cross-encoder", "Cross-enc.") for s in ss], rotation=30, ha="right", fontsize=7)
        ax.set_title(nm, fontsize=8)
        ax.set_ylim(30, 62)
        ax.yaxis.grid(True, lw=0.3, alpha=0.6)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("EM (solid) and F1 (light), %")
    save(fig, out, "fig_answers")
    print("FIGS_OK")


if __name__ == "__main__":
    main()
