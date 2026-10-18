"""Tables for the r6 revision run: extra baselines, other encoders, untouched sample C.
Usage: python -m kgrag.report_r6 --res ../results --out ../paper"""
import argparse, json
from pathlib import Path
import numpy as np
from . import metrics as M

LAB = {"dense": "Dense", "bm25": "BM25", "fusion_rrf": "BM25+dense (RRF)", "fusion_linear": "BM25+dense (linear)", "qtitle": "Question-title boost",
       "hybrid2": "Two-hop", "hybrid2_gate": "Two-hop + gate", "titlelink": "TitleLink", "ce_dense": "Cross-encoder (dense top-20)"}
CODE = {"dense": "Dense", "bm25": "Bm", "fusion_rrf": "Rrf", "fusion_linear": "Lin", "qtitle": "Qt", "hybrid2": "Two", "hybrid2_gate": "Gate", "titlelink": "Tl", "ce_dense": "Ce"}
DS = [("A", "HotpotQA A"), ("B", "HotpotQA B"), ("W", "2Wiki"), ("Q", "MuSiQue")]


def rd(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def pc(x):
    return f"{100 * x:.1f}"


def sg(x):
    return f"{x:+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="../paper")
    a = ap.parse_args()
    R, out = Path(a.res), Path(a.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    mac = []
    summ = lambda d: json.loads((R / d / "summary.json").read_text())["systems"]
    dirs = {"small": {"A": "eval_v2_seed0", "B": "eval_v2_seed1", "W": "eval_2wiki", "Q": "eval_musique"},
            "base": {"A": "enc_base_hotpotA", "B": "enc_base_hotpotB", "W": "enc_base_2wiki", "Q": "enc_base_musique"},
            "large": {"A": "enc_large_hotpotA", "B": "enc_large_hotpotB", "W": "enc_large_2wiki", "Q": "enc_large_musique"}}
    # ---- baselines table (BGE-small)
    systems = ["bm25", "dense", "fusion_rrf", "fusion_linear", "qtitle", "titlelink", "hybrid2_gate"]
    S = {k: summ(d) for k, d in dirs["small"].items()}
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Added baselines (All@5, \\%, BGE-small, 1\\,000 questions per set). BM25+dense fusion uses untuned reciprocal-rank fusion ($k=60$) or an equal-weight min--max sum; the question-title boost adds the title-link bonus ($0.5$) to dense scores of passages whose title is an entity named in the question, with no first hop. Best value per column in bold.}",
         "\\label{tab:r6base}", "\\begin{tabular}{lrrrr}", "\\toprule", "System & HotpotQA A & HotpotQA B & 2Wiki & MuSiQue \\\\", "\\midrule"]
    best = {k: max(S[k][s]["all_support"] for s in systems) for k in S}
    for s in systems:
        cells = []
        for k, _ in DS:
            v = S[k][s]["all_support"]
            cells.append(("\\textbf{%s}" if abs(v - best[k]) < 1e-9 else "%s") % pc(v))
            mac.append(f"\\newcommand{{\\Rb{k}{CODE[s]}}}{{{pc(v)}}}")
        L.append(f"{LAB[s]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "r6_baselines.tex").write_text("\n".join(L) + "\n")
    # ---- encoders
    enc = [("small", "BGE-small (33M)"), ("base", "BGE-base (110M)"), ("large", "BGE-large (335M)")]
    es = ["dense", "qtitle", "titlelink", "hybrid2_gate"]
    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
         "\\caption{Other BGE encoders (All@5, \\%, 1\\,000 questions per set). The method settings are those frozen with BGE-small and were not re-tuned for the larger encoders.}",
         "\\label{tab:r6enc}", "\\begin{tabular}{llrrrr}", "\\toprule", "Encoder & System & HotpotQA A & HotpotQA B & 2Wiki & MuSiQue \\\\", "\\midrule"]
    for e, en in enc:
        SS = {k: summ(d) for k, d in dirs[e].items()}
        for i, s in enumerate(es):
            row = []
            for k, _ in DS:
                v = SS[k][s]["all_support"]; row.append(pc(v))
                mac.append(f"\\newcommand{{\\Re{e.capitalize()}{k}{CODE[s]}}}{{{pc(v)}}}")
                if s == "titlelink":
                    mac.append(f"\\newcommand{{\\Re{e.capitalize()}{k}TlGap}}{{{pc(v - SS[k]['dense']['all_support'])}}}")
            L.append((f"\\multirow{{{len(es)}}}{{*}}{{{en}}}" if i == 0 else "") + f" & {LAB[s]} & " + " & ".join(row) + " \\\\")
        L.append("\\midrule" if e != "large" else "")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "r6_encoders.tex").write_text("\n".join(l for l in L if l != "") + "\n")
    # ---- sample C and the part disjoint from A and B
    ids = lambda d: {r["qid"] for r in rd(R / d / "dense.jsonl")}
    A, B = ids("eval_v2_seed0"), ids("eval_v2_seed1")
    C = {s: rd(R / "eval_v2_seed2" / f"{s}.jsonl") for s in ("dense", "bm25", "fusion_rrf", "fusion_linear", "qtitle", "hybrid2", "hybrid2_gate", "titlelink", "ce_dense")}
    cid = [r["qid"] for r in C["dense"]]
    keep = [i for i, q in enumerate(cid) if q not in A and q not in B]
    mac += [f"\\newcommand{{\\RcN}}{{{len(cid)}}}", f"\\newcommand{{\\RcDisj}}{{{len(keep)}}}", f"\\newcommand{{\\RcOverlap}}{{{len(cid) - len(keep)}}}"]
    cs = ["dense", "fusion_rrf", "qtitle", "ce_dense", "hybrid2", "hybrid2_gate", "titlelink"]
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Untouched HotpotQA sample C (seed~2), evaluated once with the frozen methods (All@5, \\%). Left: all 1\\,000 questions; right: the \\RcDisj{} questions that are in neither sample A nor sample B. $\\Delta$: paired difference to dense retrieval with a 95\\% bootstrap interval (10\\,000 resamples; exploratory).}",
         "\\label{tab:r6c}", "\\begin{tabular}{lrrrr}", "\\toprule", "System & All@5 (all) & $\\Delta$ (all) & All@5 (disjoint) & $\\Delta$ (disjoint) \\\\", "\\midrule"]
    for s in cs:
        x = np.array([r["all_support"] for r in C[s]]); d = np.array([r["all_support"] for r in C["dense"]])
        row = [pc(x.mean())]; mac.append(f"\\newcommand{{\\Rc{CODE[s]}All}}{{{pc(x.mean())}}}")
        for idx, nm in ((None, "All"), (keep, "Dis")):
            xs, ds = (x, d) if idx is None else (x[idx], d[idx])
            if s == "dense":
                dd = "--"; v = pc(xs.mean())
            else:
                o, lo, hi, p = M.paired_bootstrap(xs, ds); dd = f"{sg(100 * o)} [{sg(100 * lo)}, {sg(100 * hi)}]"; v = pc(xs.mean())
                mac += [f"\\newcommand{{\\Rc{CODE[s]}{nm}D}}{{{sg(100 * o)}}}", f"\\newcommand{{\\Rc{CODE[s]}{nm}Lo}}{{{sg(100 * lo)}}}", f"\\newcommand{{\\Rc{CODE[s]}{nm}Hi}}{{{sg(100 * hi)}}}"]
            if nm == "Dis":
                mac.append(f"\\newcommand{{\\Rc{CODE[s]}DisAll}}{{{v}}}")
            row += [dd] if nm == "All" else [v, dd]
        L.append(f"{LAB[s]} & " + " & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "r6_samplec.tex").write_text("\n".join(L) + "\n")
    # paired: gate vs titlelink, TitleLink vs CE on disjoint
    for x, y, nm in (("titlelink", "ce_dense", "TlCe"), ("hybrid2_gate", "titlelink", "GateTl")):
        xa = np.array([r["all_support"] for r in C[x]])[keep]; ya = np.array([r["all_support"] for r in C[y]])[keep]
        o, lo, hi, p = M.paired_bootstrap(xa, ya)
        mac += [f"\\newcommand{{\\Rp{nm}D}}{{{sg(100 * o)}}}", f"\\newcommand{{\\Rp{nm}Lo}}{{{sg(100 * lo)}}}", f"\\newcommand{{\\Rp{nm}Hi}}{{{sg(100 * hi)}}}"]
    (out / "numbers_r6.tex").write_text("% generated by kgrag.report_r6\n" + "\n".join(dict.fromkeys(mac)) + "\n")
    print("R6_WRITTEN", len(mac), "overlap", len(cid) - len(keep))


if __name__ == "__main__":
    main()
