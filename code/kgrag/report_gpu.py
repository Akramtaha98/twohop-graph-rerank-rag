"""Tables and macros for the GPU run (TitleLink retrieval and answers, sample-B answers, common-pool cross-encoder comparison).

    python -m kgrag.report_gpu --res ../mac/results --out ../paper
Writes tables/gpu_titlelink.tex, tables/gpu_answers.tex, tables/gpu_commonpool.tex and tables/numbers_gpu.tex. Numbers are recomputed from the
per-question files; paired bootstrap with 10 000 resamples.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as M


def rd(p):
    p = Path(p)
    return {json.loads(l)["qid"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else None


def sg(v):
    return f"{v:+.1f}".replace("-", "$-$")


def pc(x):
    return f"{100 * x:.1f}"


def paired(a, b, key):
    q = sorted(set(a) & set(b))
    d, lo, hi, p = M.paired_bootstrap([a[i][key] for i in q], [b[i][key] for i in q])
    return 100 * d, 100 * lo, 100 * hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="../paper")
    a = ap.parse_args()
    R, out = Path(a.res), Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    mac = [f"\\newcommand{{\\GpuAnsGpuA}}{{{'1' if (R / 'answersA_gpu').exists() else '0'}}}"]

    # ---- TitleLink retrieval on the four sets, and a cross-encoder after it
    sets = [("HA", "HotpotQA A", R / "eval_v2_seed0"), ("HB", "HotpotQA B", R / "eval_v2_seed1"), ("TW", "2Wiki", R / "eval_2wiki"), ("MQ", "MuSiQue", R / "eval_musique")]
    rows = [("dense", "Dense", "Dense"), ("hybrid2", "Two-hop", "Two"), ("hybrid2_gate", "Two-hop + gate", "Gate"), ("titlelink", "TitleLink", "TL"),
            ("spriglike", "SPRIG-style (re-implemented)", "Sprig"), ("linearlike", "LinearRAG-style (re-implemented)", "Lin"), ("ce_dense", "Cross-encoder on dense top-20", "CE"), ("ce_hybrid2_gate", "Cross-encoder on gated two-hop top-20", "CEGate"), ("ce_titlelink", "Cross-encoder on TitleLink top-20", "CETL")]
    T = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
         "\\caption{TitleLink (graph and title link only, no expanded query, $\\gamma=\\CtlGammaTit$ chosen on training questions) next to the other systems: R@5 and All@5 (\\%).}",
         "\\label{tab:titlelink}", "\\begin{tabular}{l" + "rr" * len(sets) + "}", "\\toprule",
         "System & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n, _ in sets) + " \\\\", "& " + " & ".join("R@5 & All@5" for _ in sets) + " \\\\", "\\midrule"]
    data = {}
    for k, _, d in sets:
        for s, _, _ in rows:
            r = rd(d / f"{s}.jsonl")
            if r:
                data[(k, s)] = r
    for s, lab, code in rows:
        cells = []
        for k, _, _ in sets:
            r = data.get((k, s))
            if r is None:
                cells += ["--", "--"]
                continue
            rec, al = np.mean([x["recall"] for x in r.values()]), np.mean([x["all_support"] for x in r.values()])
            cells += [pc(rec), pc(al)]
            mac += [f"\\newcommand{{\\Gpu{k}{code}Rec}}{{{pc(rec)}}}", f"\\newcommand{{\\Gpu{k}{code}All}}{{{pc(al)}}}"]
        T.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables" / "gpu_titlelink.tex").write_text("\n".join(T) + "\n")
    for k, _, _ in sets:  # paired: TitleLink vs two-hop (ungated) and vs gated two-hop
        for ref, code in (("hybrid2", "Two"), ("hybrid2_gate", "Gate"), ("ce_dense", "CE"), ("spriglike", "Sprig"), ("linearlike", "Lin")):
            if (k, "titlelink") in data and (k, ref) in data:
                d, lo, hi, _ = paired(data[(k, "titlelink")], data[(k, ref)], "all_support")
                mac += [f"\\newcommand{{\\GpuPair{k}TLv{code}D}}{{{sg(d)}}}", f"\\newcommand{{\\GpuPair{k}TLv{code}Lo}}{{{sg(lo)}}}", f"\\newcommand{{\\GpuPair{k}TLv{code}Hi}}{{{sg(hi)}}}"]

    # ---- answers: sample A (Mac for the original systems, GPU for TitleLink) and sample B (all on the GPU)
    gpuA = (R / "answersA_gpu").exists()
    adir = R / "answersA_gpu" if gpuA else R / "hotpot"
    A = {s: rd(adir / f"{s}.jsonl") for s in ("dense", "hybrid2", "hybrid2_gate", "ce_dense", "titlelink", "spriglike", "linearlike")}
    B = {s: rd(R / "eval_v2_seed1" / f"{s}.jsonl") for s in ("dense", "hybrid2_gate", "ce_dense", "titlelink")}
    lab = {"dense": "Dense", "hybrid2": "Two-hop", "hybrid2_gate": "Two-hop + gate", "ce_dense": "Cross-encoder (dense top-20)", "titlelink": "TitleLink", "spriglike": "SPRIG-style", "linearlike": "LinearRAG-style"}
    code = {"dense": "Dense", "hybrid2": "Two", "hybrid2_gate": "Gate", "ce_dense": "CE", "titlelink": "TL", "spriglike": "Sprig", "linearlike": "Lin"}
    U = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
         "\\caption{Answers of Qwen2.5-1.5B-Instruct (\\%), \\emph{rerun of all systems on the H200 GPU server} (the original laptop run of sample A is in Table~\\ref{tab:answers}; values differ by 0.1 point or less in exact match). Unsup.: lexically unsupported answers (lower is better). " + ("All systems on both samples were run on the same GPU server." if gpuA else "On sample A the TitleLink answers were generated on a GPU server and the other systems on the laptop; on sample B all systems were run on the GPU server, so the comparisons within B are like for like.") + "}",
         "\\label{tab:gpuanswers}", "\\begin{tabular}{llrrrr}", "\\toprule", "Sample & System & All@5 & EM & F1 & Unsup.$\\downarrow$ \\\\", "\\midrule"]
    for key, D, sysl in (("A", A, ("dense", "hybrid2", "hybrid2_gate", "ce_dense", "titlelink", "spriglike", "linearlike")), ("B", B, ("dense", "hybrid2_gate", "ce_dense", "titlelink"))):
        for i, s in enumerate(sysl):
            r = D[s]
            if r is None:
                continue
            vals = {m: np.mean([x[m] for x in r.values()]) for m in ("all_support", "em", "f1", "unsupported")}
            U.append(f"{'Sample ' + key if i == 0 else ''} & {lab[s]} & {pc(vals['all_support'])} & {pc(vals['em'])} & {pc(vals['f1'])} & {pc(vals['unsupported'])} \\\\")
            for m, mc in (("all_support", "All"), ("em", "EM"), ("f1", "FOne"), ("unsupported", "Unsup")):
                mac.append(f"\\newcommand{{\\GpuAns{key}{code[s]}{mc}}}{{{pc(vals[m])}}}")
        U.append("\\midrule" if key == "A" else "")
    U = [u for u in U if u] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tables" / "gpu_answers.tex").write_text("\n".join(U) + "\n")
    pairs = [("A", "hybrid2_gate", "ce_dense"), ("A", "hybrid2_gate", "dense"), ("A", "hybrid2_gate", "spriglike"), ("A", "hybrid2_gate", "linearlike"), ("A", "titlelink", "dense"), ("A", "titlelink", "hybrid2_gate"), ("A", "titlelink", "ce_dense"), ("B", "titlelink", "dense"), ("B", "titlelink", "hybrid2_gate"), ("B", "titlelink", "ce_dense"),
             ("B", "hybrid2_gate", "dense"), ("B", "hybrid2_gate", "ce_dense"), ("B", "ce_dense", "dense")]
    for key, x, y in pairs:
        D = A if key == "A" else B
        if D[x] is None or D[y] is None:
            continue
        for m, mc in (("em", "EM"), ("f1", "FOne")):
            d, lo, hi, _ = paired(D[x], D[y], m)
            mac += [f"\\newcommand{{\\GpuPa{key}{code[x]}v{code[y]}{mc}D}}{{{sg(d)}}}", f"\\newcommand{{\\GpuPa{key}{code[x]}v{code[y]}{mc}Lo}}{{{sg(lo)}}}", f"\\newcommand{{\\GpuPa{key}{code[x]}v{code[y]}{mc}Hi}}{{{sg(hi)}}}"]

    # ---- common-pool comparison: the same top-20 candidate pool ordered by dense, two-hop and the cross-encoder
    P = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
         "\\caption{Common-pool comparison, All@5 (\\%): the same 20 candidates are ordered by the dense score, the two-hop score or the cross-encoder. Pool D is the dense top-20, pool H the (ungated) two-hop top-20. The last column is the All@20 of the pool.}",
         "\\label{tab:commonpool}", "\\begin{tabular}{llrrrr}", "\\toprule", "Set & Pool & Dense order & Two-hop order & Cross-encoder order & Pool All@20 \\\\", "\\midrule"]
    for k, nm, d in (("HA", "HotpotQA A", R / "controls_ce" / "hotpot_0"), ("HB", "HotpotQA B", R / "controls_ce" / "hotpot_1")):
        for pl, pn in (("D", "D (dense top-20)"), ("H", "H (two-hop top-20)")):
            r = {o: rd(d / f"pool{pl}_{o}.jsonl") for o in ("dense", "two_hop", "ce")}
            if any(v is None for v in r.values()):
                continue
            v = {o: np.mean([x["all_support"] for x in r[o].values()]) for o in r}
            a20 = np.mean([x["all20"] for x in r["dense"].values()])
            P.append(f"{nm if pl == 'D' else ''} & {pn} & {pc(v['dense'])} & {pc(v['two_hop'])} & {pc(v['ce'])} & {pc(a20)} \\\\")
            for o, oc in (("dense", "Dense"), ("two_hop", "Two"), ("ce", "CE")):
                mac.append(f"\\newcommand{{\\Pool{k}{pl}{oc}}}{{{pc(v[o])}}}")
            mac.append(f"\\newcommand{{\\Pool{k}{pl}AllTwenty}}{{{pc(a20)}}}")
            d1, lo, hi, _ = paired(r["ce"], r["two_hop"], "all_support")
            mac += [f"\\newcommand{{\\Pool{k}{pl}CEvTwoD}}{{{sg(d1)}}}", f"\\newcommand{{\\Pool{k}{pl}CEvTwoLo}}{{{sg(lo)}}}", f"\\newcommand{{\\Pool{k}{pl}CEvTwoHi}}{{{sg(hi)}}}"]
    P += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tables" / "gpu_commonpool.tex").write_text("\n".join(P) + "\n")
    (out / "tables" / "numbers_gpu.tex").write_text("\n".join(mac) + "\n")
    print("GPU_TABLES_WRITTEN", len(mac))


if __name__ == "__main__":
    main()
