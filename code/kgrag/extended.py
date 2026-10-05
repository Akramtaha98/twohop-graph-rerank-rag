"""Tables and macros for the extended evaluation (gate, cross-encoder, 2WikiMultiHopQA, MuSiQue, answers).

    python -m kgrag.extended --hotpot ../results/hotpot --a ../results/eval_v2_seed0 --b ../results/eval_v2_seed1 \
        --wiki ../results/eval_2wiki --musique ../results/eval_musique --out ../paper
Every number in the extended-evaluation section is written by this script.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M

DS = [("HA", "HotpotQA A"), ("HB", "HotpotQA B"), ("TW", "2Wiki"), ("MQ", "MuSiQue")]
SYS = [("bm25", "BM25", "BM"), ("dense", "Dense", "Dense"), ("ce_dense", "Cross-encoder on dense top-20", "CE"),
       ("hybrid2", "Two-hop re-ranking", "Two"), ("hybrid2_gate", "Two-hop + question-type gate", "Gate"),
       ("ce_hybrid2", "Cross-encoder on two-hop top-20", "CETwo"), ("ce_hybrid2_gate", "Cross-encoder on gated two-hop top-20", "CEGate"),
       ("hybrid2_oracle", "Two-hop + oracle gate (upper bound)", "Orac")]
CODE = {s: c for s, _, c in SYS}
SHORT = {"bm25": "BM25", "dense": "Dense", "ce_dense": "CE (dense top-20)", "hybrid2": "Two-hop", "hybrid2_gate": "Two-hop + gate",
         "ce_hybrid2": "CE (two-hop top-20)", "ce_hybrid2_gate": "CE (gated two-hop)", "hybrid2_oracle": "Two-hop + oracle gate"}
PAIRS = [("hybrid2", "dense"), ("ce_dense", "dense"), ("hybrid2", "ce_dense"), ("ce_hybrid2", "ce_dense"), ("hybrid2_gate", "hybrid2"), ("hybrid2_gate", "dense"), ("hybrid2_gate", "ce_dense"), ("ce_hybrid2_gate", "ce_dense"), ("ce_hybrid2", "hybrid2")]


def rd(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in Path(p).read_text().splitlines()} if Path(p).exists() else None


def sg(v):
    return f"{v:+.1f}".replace("-", "$-$")


def f1(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--musique", required=True)
    ap.add_argument("--out", default="../paper")
    a = ap.parse_args()
    out = Path(a.out)
    dirs = {"HA": a.a, "HB": a.b, "TW": a.wiki, "MQ": a.musique}
    summ = {k: json.loads((Path(d) / "summary.json").read_text()) for k, d in dirs.items()}
    recs = {k: {s: rd(Path(d) / f"{s}.jsonl") for s, _, _ in SYS} for k, d in dirs.items()}
    mac = []

    # ---- main retrieval table (R@5 / All@5 / seconds)
    L = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
         "\\caption{Retrieval with two-hop re-ranking, a question-type gate and a cross-encoder on four evaluation sets (\\%; 1\\,000 questions each). Settings were frozen from HotpotQA training data; 2WikiMultiHopQA and MuSiQue are zero-shot transfer. Sec.: retrieval time for 1\\,000 questions on the M4 laptop. Best non-oracle value per column in bold.}",
         "\\label{tab:ext}", "\\begin{tabular}{l" + "rrr" * len(DS) + "}", "\\toprule",
         "System & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for _, n in DS) + " \\\\",
         "& " + " & ".join("R@5 & All@5 & Sec." for _ in DS) + " \\\\", "\\midrule"]
    best = {}
    for k, _ in DS:
        for m in ("recall", "all_support"):
            best[(k, m)] = max(summ[k]["systems"][s][m] for s, _, _ in SYS if s in summ[k]["systems"] and s != "hybrid2_oracle")
    for s, name, c in SYS:
        cells = []
        for k, _ in DS:
            v = summ[k]["systems"].get(s)
            if v is None:
                cells += ["--", "--", "--"]
                continue
            for m in ("recall", "all_support"):
                t = f1(v[m])
                cells.append(f"\\textbf{{{t}}}" if s != "hybrid2_oracle" and abs(v[m] - best[(k, m)]) < 1e-12 else t)
            cells.append(f"{v['seconds']:.0f}")
            mac += [f"\\newcommand{{\\Ext{k}{c}All}}{{{f1(v['all_support'])}}}", f"\\newcommand{{\\Ext{k}{c}Rec}}{{{f1(v['recall'])}}}",
                    f"\\newcommand{{\\Ext{k}{c}Sec}}{{{v['seconds']:.1f}}}"]
            if "gated" in v:
                mac.append(f"\\newcommand{{\\Ext{k}{c}Gated}}{{{f1(v['gated'])}}}")
        L.append(f"{SHORT[s]} & " + " & ".join(cells) + " \\\\")
        if s in ("dense", "hybrid2_gate"):
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables/ext_retrieval.tex").write_text("\n".join(L))

    # ---- paired bootstrap on All@5 and R@5
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Paired bootstrap (10\\,000 resamples) for All@5: mean difference in points with 95\\% interval. An interval that excludes zero corresponds to $p<0.05$.}",
         "\\label{tab:extsig}", "\\begin{tabular}{l" + "r" * len(DS) + "}", "\\toprule", "Comparison & " + " & ".join(n for _, n in DS) + " \\\\", "\\midrule"]
    names = SHORT
    for x, y in PAIRS:
        cells = []
        for k, _ in DS:
            rx, ry = recs[k][x], recs[k][y]
            ids = sorted(set(rx) & set(ry))
            for key, tag in (("all_support", "All"), ("recall", "Rec")):
                d, lo, hi, p = M.paired_bootstrap([rx[i][key] for i in ids], [ry[i][key] for i in ids])
                nm = f"Ext{k}{CODE[x]}V{CODE[y]}{tag}"
                mac += [f"\\newcommand{{\\{nm}D}}{{{sg(100*d)}}}", f"\\newcommand{{\\{nm}Lo}}{{{sg(100*lo)}}}", f"\\newcommand{{\\{nm}Hi}}{{{sg(100*hi)}}}",
                        f"\\newcommand{{\\{nm}P}}{{{('$<$0.001' if p < 0.001 else f'{p:.3f}')}}}"]
                if tag == "All":
                    cells.append(f"{sg(100*d)} [{sg(100*lo)}, {sg(100*hi)}]")
        L.append(f"{names[x]} vs {names[y]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables/ext_signif.tex").write_text("\n".join(L))

    # ---- per question type
    def grp(k, t):
        if k == "MQ":
            return "4hop" if t.startswith("4hop") else "3hop" if t.startswith("3hop") else t
        return t
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
         "\\caption{All@5 (\\%) by question type. CE: cross-encoder. The gate restores comparison questions to about the dense level. Two-hop re-ranking gains on bridge, compositional, inference and 2-hop questions, but not on 3-hop or 4-hop MuSiQue questions (it loses on 3-hop).}",
         "\\label{tab:bytype2}"]
    cols = [("HA", "bridge"), ("HA", "comparison"), ("HB", "bridge"), ("HB", "comparison"), ("TW", "compositional"), ("TW", "inference"),
            ("TW", "comparison"), ("TW", "bridge_comparison"), ("MQ", "2hop"), ("MQ", "3hop"), ("MQ", "4hop")]
    qt = {}
    for k, _ in DS:
        g = recs[k]["hybrid2_gate"]
        qt[k] = {i: grp(k, x["qtype"]) for i, x in g.items()}
    TS = {"bridge": "bridge", "comparison": "compar.", "compositional": "compos.", "inference": "infer.", "bridge_comparison": "br.-cmp.", "2hop": "2-hop", "3hop": "3-hop", "4hop": "4-hop"}
    L += ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
          " & \\multicolumn{2}{c}{HotpotQA A} & \\multicolumn{2}{c}{HotpotQA B} & \\multicolumn{4}{c}{2WikiMultiHopQA} & \\multicolumn{3}{c}{MuSiQue} \\\\",
          "System & " + " & ".join(TS[t] for _, t in cols) + " \\\\",
          "\\midrule", "$n$ & " + " & ".join(str(sum(1 for v in qt[k].values() if v == t)) for k, t in cols) + " \\\\", "\\midrule"]
    for s, name, c in SYS:
        if s == "bm25":
            continue
        row = []
        for k, t in cols:
            r = recs[k][s]
            ids = [i for i, v in qt[k].items() if v == t and i in r]
            val = float(np.mean([r[i]["all_support"] for i in ids])) if ids else float("nan")
            row.append(f1(val))
            mac.append(f"\\newcommand{{\\Typ{k}{CODE[s]}{t.replace('_', '').replace('1','').replace('2','Two').replace('3','Three').replace('4','Four')}}}{{{f1(val)}}}")
        L.append(f"{SHORT[s]} & " + " & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables/ext_bytype.tex").write_text("\n".join(L))
    # counts as macros
    for k, t in cols:
        mac.append(f"\\newcommand{{\\TypN{k}{t.replace('_', '').replace('1','').replace('2','Two').replace('3','Three').replace('4','Four')}}}{{{sum(1 for v in qt[k].values() if v == t)}}}")

    # ---- answer quality on sample A (HotpotQA run with the generator)
    hs = json.loads((Path(a.hotpot) / "summary.json").read_text())["systems"]
    hr = {s: rd(Path(a.hotpot) / f"{s}.jsonl") for s in ("dense", "ce_dense", "hybrid2", "hybrid2_gate", "oracle")}
    rows = [("dense", "Dense", "Dense"), ("ce_dense", "Cross-encoder on dense top-20", "CE"), ("hybrid2", "Two-hop re-ranking", "Two"),
            ("hybrid2_gate", "Two-hop + gate", "Gate"), ("oracle", "Gold passages (upper bound)", "Orac")]
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\caption{Answers of Qwen2.5-1.5B-Instruct on sample A (\\%). Unsup.: lexically unsupported answers (lower is better). Sec.: end-to-end time for 1\\,000 questions.}",
         "\\label{tab:answers}", "\\begin{tabular}{lrrrrr}", "\\toprule", "System & All@5 & EM & F1 & Unsup.$\\downarrow$ & Sec. \\\\", "\\midrule"]
    for s, name, c in rows:
        v = hs[s]
        L.append(f"{name} & {f1(v['all_support'])} & {f1(v['em'])} & {f1(v['f1'])} & {f1(v['unsupported'])} & {v['seconds']:.0f} \\\\")
        mac += [f"\\newcommand{{\\Ans{c}EM}}{{{f1(v['em'])}}}", f"\\newcommand{{\\Ans{c}FOne}}{{{f1(v['f1'])}}}", f"\\newcommand{{\\Ans{c}Unsup}}{{{f1(v['unsupported'])}}}",
                f"\\newcommand{{\\Ans{c}Sec}}{{{v['seconds']:.0f}}}"]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tables/ext_answers.tex").write_text("\n".join(L))
    for x, y in [("hybrid2_gate", "dense"), ("hybrid2", "ce_dense"), ("hybrid2_gate", "ce_dense"), ("ce_dense", "dense"), ("hybrid2_gate", "hybrid2")]:
        ids = sorted(set(hr[x]) & set(hr[y]))
        for key, tag in (("em", "EM"), ("f1", "FOne"), ("unsupported", "Unsup")):
            d, lo, hi, p = M.paired_bootstrap([hr[x][i][key] for i in ids], [hr[y][i][key] for i in ids])
            nm = f"Ans{CODE.get(x, 'Orac')}V{CODE.get(y, 'Orac')}{tag}"
            mac += [f"\\newcommand{{\\{nm}D}}{{{sg(100*d)}}}", f"\\newcommand{{\\{nm}Lo}}{{{sg(100*lo)}}}", f"\\newcommand{{\\{nm}Hi}}{{{sg(100*hi)}}}",
                    f"\\newcommand{{\\{nm}P}}{{{('$<$0.001' if p < 0.001 else f'{p:.3f}')}}}"]
    # speed ratio
    mac.append(f"\\newcommand{{\\ExtSpeedRatio}}{{{summ['HA']['systems']['ce_dense']['seconds'] / summ['HA']['systems']['hybrid2_gate']['seconds']:.0f}}}")
    t3 = json.loads((Path(a.hotpot).parent / "tuned3.json").read_text())
    for g in t3["grid"]:
        mac.append(f"\\newcommand{{\\Gate{g['gate'].capitalize()}Train}}{{{f1(g['all_support'])}}}\\newcommand{{\\Gate{g['gate'].capitalize()}Share}}{{{f1(g['gated_share'])}}}")
    mac.append(f"\\newcommand{{\\GateChosen}}{{{t3['best']['gate']}}}")
    (out / "tables/numbers_ext.tex").write_text("\n".join(mac))
    print("EXT_WRITTEN", len(mac))


if __name__ == "__main__":
    main()
