"""Tables and macros for the controls, candidate-coverage, 2Wiki audit and corpus-scale experiments.

    python -m kgrag.report_controls --dir ../ctl --out ../paper [--timing ../ctl/timing_cpu.json ...]

gamma of each variant is selected on the HotpotQA TRAIN sample only (`train/summary.json`); gamma = 0.3 is the frozen value of the proposed method.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as M

DS = [("HA", "hotpot_A", "HotpotQA A"), ("HB", "hotpot_B", "HotpotQA B"), ("HX", "hotpot_B", "B$\\setminus$A"), ("TW", "2wiki", "2Wiki"), ("MQ", "musique", "MuSiQue")]
SAMPLE_A_IDS = set(Path(__file__).resolve().parent.parent.parent.joinpath("results_ids", "sample_ids_hotpot_A.txt").read_text().split()) if Path(__file__).resolve().parent.parent.parent.joinpath("results_ids", "sample_ids_hotpot_A.txt").exists() else set()
VAR = [("dense", "Dense", "Dense"), ("two_hop", "Two-hop (graph + expansion + title)", "Two"),
       ("expand_all", "Expansion, no graph", "Exp"), ("title_only", "Graph + title link, no expansion", "Tit")]


def sg(v):
    return f"{v:+.1f}".replace("-", "$-$")


def rd(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).exists() else None


def fn(v, gate=False, g=None):
    return f"{v}@{g}" + ("+gate" if gate else "") if v != "dense" else "dense"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="../paper")
    ap.add_argument("--timing", nargs="*", default=[], help="label=path.json")
    a = ap.parse_args()
    root, out = Path(a.dir), Path(a.out)
    mac = []
    ids_dir = Path(__file__).resolve().parent.parent.parent / "results_ids"
    if (ids_dir / "sample_ids_hotpot_B.txt").exists():
        ia, ib = set((ids_dir / "sample_ids_hotpot_A.txt").read_text().split()), set((ids_dir / "sample_ids_hotpot_B.txt").read_text().split())
        mac.append(f"\\newcommand{{\\nOverlapAB}}{{{len(ia & ib)}}}")
        mac.append(f"\\newcommand{{\\nDisjointB}}{{{len(ib - ia)}}}")
    tr = json.loads((root / "train" / "summary.json").read_text())["rows"]
    gam = {}
    for v in ("two_hop", "expand_all", "title_only"):
        c = {float(k.split("@")[1]): r["all_support"] for k, r in tr.items() if k.startswith(v + "@") and "gate" not in k}
        gam[v] = max(sorted(c), key=lambda g: (c[g], -abs(g - 0.3)))
        mac.append(f"\\newcommand{{\\CtlGamma{dict(two_hop='Two', expand_all='Exp', title_only='Tit')[v]}}}{{{gam[v]}}}")
        mac.append(f"\\newcommand{{\\CtlTrain{dict(two_hop='Two', expand_all='Exp', title_only='Tit')[v]}All}}{{{100 * 0 + c[gam[v]]:.1f}}}")
    trs = json.loads((root / "train" / "summary.json").read_text())
    mac += [f"\\newcommand{{\\ClTrainN}}{{{trs['meta']['n_questions']}}}", f"\\newcommand{{\\ClTrainP}}{{{trs['meta']['n_passages']:,}}}".replace(",", "\\,"), f"\\newcommand{{\\ClDenseTrain}}{{{trs['rows']['dense']['all_support']:.1f}}}"]
    code = {"dense": "Dense", "two_hop": "Two", "expand_all": "Exp", "title_only": "Tit"}

    def get(ds, v, g, gate, key=None):
        name = "dense" if v == "dense" else f"{v}@{g}" + ("+gate" if gate else "")
        recs = rd(root / ds / (name.replace(":", "_").replace("+", "_") + ".jsonl"))
        if recs is not None and key == "HX":
            recs = [r for r in recs if r["qid"] not in SAMPLE_A_IDS]
        return recs

    # ---- controls table (All@5), frozen gamma 0.3 and train-selected gamma
    rows = [("dense", None, "Dense"), ("two_hop", 0.3, "Two-hop (graph + expansion + title)"),
            ("expand_all", 0.3, "Expansion only, no graph, $\\gamma=0.3$"), ("expand_all", "T", "Expansion only, no graph, $\\gamma$ tuned on train"),
            ("title_only", 0.3, "Graph + title link only, $\\gamma=0.3$"), ("title_only", "T", "Graph + title link only, $\\gamma$ tuned on train")]
    cap = {False: "Controls for the two-hop re-ranker, without the question-type gate: All@5 (\\%) in four settings. Dense is the baseline. The full method uses the graph, the title bonus and the expanded queries. \\emph{Expansion only} scores the expanded queries against the whole corpus without the graph or the title bonus. \\emph{Graph + title link only} keeps the graph and the title bonus and drops the expanded queries. All variants share the encoder and the first-hop set ($m=3$); the first two variants use the same expanded queries. $\\gamma$ was either frozen at 0.3 or chosen on HotpotQA training questions only ($\\gamma_{\\mathrm{exp}}=\\CtlGammaExp$, $\\gamma_{\\mathrm{tit}}=\\CtlGammaTit$).",
           True: "The same controls as in Table~\\ref{tab:controls} with the title gate applied to every variant (the gate returns the dense ranking for comparison-style questions)."}
    L = []
    allv = {}
    for gate in (False, True):
        L += ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\caption{" + cap[gate] + "}",
              "\\label{tab:controls}" if not gate else "\\label{tab:controlsgate}", "\\begin{tabular}{l" + "r" * len(DS) + "}", "\\toprule",
              "Variant & " + " & ".join(n for _, _, n in DS) + " \\\\", "\\midrule"]
        for v, g, label in rows:
            if g == "T" and gam[v] == 0.3:
                continue
            gg = gam[v] if g == "T" else g
            cells = []
            for k, d, _ in DS:
                recs = get(d, v, gg, gate if v != "dense" else False, k) if not (v == "dense" and gate) else get(d, "dense", None, False, k)
                if recs is None:
                    cells.append("--")
                    continue
                m = 100 * float(np.mean([r["all_support"] for r in recs]))
                cells.append(f"{m:.1f}")
                tag = {None: "", 0.3: "F", "T": "T"}[g]
                allv[(k, v, tag, gate)] = recs
                mac.append(f"\\newcommand{{\\Ctl{k}{code[v]}{tag}All{'Gate' if gate else ''}}}{{{m:.1f}}}")
            L.append(f"{label} & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables" / "ctl_controls.tex").write_text("\n".join(L) + "\n")

    # ---- paired comparisons of the proposed method against the controls (ungated All@5)
    S = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Paired differences in All@5 (points, 95\\% bootstrap interval, 10\\,000 resamples) between the proposed two-hop re-ranker ($\\gamma=0.3$) and each control (ungated, $\\gamma$ tuned on train for the control).}",
         "\\label{tab:ctlsig}", "\\begin{tabular}{lll}", "\\toprule", "Set & vs expansion only & vs graph + title only \\\\", "\\midrule"]
    for k, d, n in DS:
        cells = []
        for v in ("expand_all", "title_only"):
            A, B = allv.get((k, "two_hop", "F", False)), allv.get((k, v, "T", False)) or allv.get((k, v, "F", False))
            if A is None or B is None:
                cells.append("--")
                continue
            ra, rb = {r["qid"]: r["all_support"] for r in A}, {r["qid"]: r["all_support"] for r in B}
            q = sorted(set(ra) & set(rb))
            dd, lo, hi, p = M.paired_bootstrap([ra[i] for i in q], [rb[i] for i in q])
            cells.append(f"{sg(100 * dd)} ({sg(100 * lo)} to {sg(100 * hi)})")
            c = code[v]
            mac += [f"\\newcommand{{\\CtlPair{k}{c}D}}{{{sg(100 * dd)}}}", f"\\newcommand{{\\CtlPair{k}{c}Lo}}{{{sg(100 * lo)}}}", f"\\newcommand{{\\CtlPair{k}{c}Hi}}{{{sg(100 * hi)}}}",
                    f"\\newcommand{{\\CtlPair{k}{c}P}}{{{'$<$0.001' if p < 0.001 else f'{p:.3f}'}}}"]
        S.append(f"{n} & " + " & ".join(cells) + " \\\\")
    S += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tables" / "ctl_sig.tex").write_text("\n".join(S) + "\n")

    # ---- paired differences against dense (used in the sample-overlap robustness paragraph)
    for k in ("HA", "HB", "HX"):
        base = allv.get((k, "dense", "", False))
        for nm, key in (("Two", (k, "two", "", False)), ("Gate", None)):
            A = allv.get((k, "two_hop", "F", False)) if nm == "Two" else allv.get((k, "two_hop", "F", True))
            if base is None or A is None:
                continue
            ra, rb = {r["qid"]: r["all_support"] for r in A}, {r["qid"]: r["all_support"] for r in base}
            q = sorted(set(ra) & set(rb))
            dd, lo, hi, pv = M.paired_bootstrap([ra[i] for i in q], [rb[i] for i in q])
            mac += [f"\\newcommand{{\\CtlVD{k}{nm}D}}{{{sg(100 * dd)}}}", f"\\newcommand{{\\CtlVD{k}{nm}Lo}}{{{sg(100 * lo)}}}", f"\\newcommand{{\\CtlVD{k}{nm}Hi}}{{{sg(100 * hi)}}}", f"\\newcommand{{\\CtlN{k}}}{{{len(q)}}}"]

    # ---- candidate coverage (R@k and All@k before any cross-encoder)
    C = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
         "\\caption{Candidate coverage (\\%): recall and All@$k$ of the first-stage rankings. A cross-encoder over the top-20 of a ranking cannot exceed the R@20 and All@20 of that ranking.}",
         "\\label{tab:coverage}", "\\begin{tabular}{ll" + "r" * 5 + "}", "\\toprule", "Set & Ranking & R@5 & All@5 & All@10 & R@20 & All@20 \\\\", "\\midrule"]
    for k, d, n in DS:
        for lab, v, g, gate, c in (("Dense", "dense", None, False, "Dense"), ("Two-hop", "two_hop", 0.3, False, "Two"), ("Two-hop + gate", "two_hop", 0.3, True, "Gate"), ("Title-link only", "title_only", gam["title_only"], False, "Tit")):
            recs = get(d, v, g, gate, k)
            if recs is None:
                continue
            vals = {key: 100 * float(np.mean([r[key] for r in recs])) for key in ("recall", "all_support", "all10", "recall20", "all20")}
            C.append(f"{n if lab == 'Dense' else ''} & {lab} & " + " & ".join(f"{vals[key]:.1f}" for key in ("recall", "all_support", "all10", "recall20", "all20")) + " \\\\")
            for key, nm in (("recall", "Rec"), ("all_support", "All"), ("all10", "AllTen"), ("recall20", "RecTwenty"), ("all20", "AllTwenty")):
                mac.append(f"\\newcommand{{\\Cov{k}{c}{nm}}}{{{vals[key]:.1f}}}")
        C.append("\\midrule" if k != "MQ" else "")
    C += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "tables" / "ctl_coverage.tex").write_text("\n".join(C) + "\n")

    # ---- 2WikiMultiHopQA audit by question type
    exs, _ = D.load_2wiki(None, 1000, 0)
    by = defaultdict(list)
    for e in exs:
        by[e["qtype"]].append(e)
    dense, two, tit = get("2wiki", "dense", None, False, "TW"), get("2wiki", "two_hop", 0.3, False, "TW"), get("2wiki", "title_only", gam["title_only"], False, "TW")
    if dense and two and tit:
        Q = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Audit of 2WikiMultiHopQA by question type (1\\,000 questions): size of the supporting set, share of questions whose supporting set fits in five passages, and retrieval (\\%) of dense retrieval (D), the two-hop re-ranker (T) and the title-link-only variant (L).}",
             "\\label{tab:audit}", "\\begin{tabular}{lrcrrrrrrr}", "\\toprule",
             "Type & $n$ & $|S|$ & fits in 5 & D R@5 & D All@5 & D All@20 & T R@5 & T All@5 & L All@5 \\\\", "\\midrule"]
        rd_, rt_, rl_ = {r["qid"]: r for r in dense}, {r["qid"]: r for r in two}, {r["qid"]: r for r in tit}
        for t in sorted(by):
            ids = [e["qid"] for e in by[t]]
            sz = sorted({len(e["sup_pids"]) for e in by[t]})
            fit = 100 * float(np.mean([len(e["sup_pids"]) <= 5 for e in by[t]]))
            f = lambda R, key: 100 * float(np.mean([R[i][key] for i in ids]))
            Q.append(f"{t.replace('_', ' ')} & {len(ids)} & {'/'.join(map(str, sz))} & {fit:.0f} & {f(rd_, 'recall'):.1f} & {f(rd_, 'all_support'):.1f} & {f(rd_, 'all20'):.1f} & {f(rt_, 'recall'):.1f} & {f(rt_, 'all_support'):.1f} & {f(rl_, 'all_support'):.1f} \\\\")
            tc = t.replace("_", "").title().replace(" ", "")
            for lab, R in (("Dense", rd_), ("Two", rt_), ("Tit", rl_)):
                for key, nm in (("recall", "Rec"), ("all_support", "All"), ("all10", "AllTen"), ("all20", "AllTwenty")):
                    mac.append(f"\\newcommand{{\\Aud{lab}{tc}{nm}}}{{{f(R, key):.1f}}}")
            mac.append(f"\\newcommand{{\\AudSize{tc}}}{{{'/'.join(map(str, sz))}}}")
            mac.append(f"\\newcommand{{\\AudN{tc}}}{{{len(ids)}}}")
        Q += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out / "tables" / "ctl_audit.tex").write_text("\n".join(Q) + "\n")

    # ---- corpus scale
    sc = []
    for nm, d, code_ in (("1\\,000", "hotpot_A", "One"), ("3\\,000", "scale_3000", "Three"), ("7\\,405", "scale_7405", "Full")):
        p = root / d / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        r, m = s["rows"], s["meta"]
        g = lambda name, key="all_support": r[name][key] if name in r else float("nan")
        idx = m["seconds_dense_index"]
        if d == "hotpot_A" and (root / "hotpot_A_grid1" / "summary.json").exists():  # the final run loaded cached embeddings; the first run measured the encoding
            idx = json.loads((root / "hotpot_A_grid1" / "summary.json").read_text())["meta"]["seconds_dense_index"]
        m = dict(m, index_seconds=idx + m["seconds_graph_build"])
        mac.append(f"\\newcommand{{\\Scale{code_}GAll}}{{{g('two_hop@0.3+gate'):.1f}}}")
        sc.append((nm, code_, m, g("dense"), g("two_hop@0.3"), g("title_only@1.0"), g("dense", "all20"), g("two_hop@0.3", "all20"), g("title_only@1.0", "all20")))
    if sc:
        T = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Corpus scale: the same 1\\,000 HotpotQA questions (sample A) retrieved from corpora pooled from more development questions (D: dense; T: two-hop; L: title-link only). Index: dense encoding plus graph construction on the cloud CPU container (2 cores), not the laptop; RSS: peak resident memory of the whole process.}",
             "\\label{tab:scale}", "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "Pool (questions) & Passages & D All@5 & T All@5 & L All@5 & D All@20 & T All@20 & L All@20 & Index (min) & RSS (GB) \\\\", "\\midrule"]
        for (nm, code_, m, dA, tA, lA, d20, t20, l20) in sc:
            T.append(f"{nm} & {m['n_passages']:,} & {dA:.1f} & {tA:.1f} & {lA:.1f} & {d20:.1f} & {t20:.1f} & {l20:.1f} & {m['index_seconds'] / 60:.0f} & {m['peak_rss_mb'] / 1024:.1f} \\\\".replace(",", "\\,").replace("\\\\,", "\\,"))
            for key, v in (("Pass", m["n_passages"]), ("DAll", dA), ("TAll", tA), ("LAll", lA), ("DTwenty", d20), ("TTwenty", t20), ("LTwenty", l20), ("IndexMin", m["index_seconds"] / 60), ("RSSgb", m["peak_rss_mb"] / 1024), ("EmbMB", m["embedding_mb"]), ("GraphNNZ", m["graph_nnz"])):
                mac.append(f"\\newcommand{{\\Scale{code_}{key}}}{{{(f'{v:,.0f}' if key in ('Pass', 'GraphNNZ') else f'{v:,.1f}')}}}".replace(",", "\\,"))
        T += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out / "tables" / "ctl_scale.tex").write_text("\n".join(T) + "\n")

    # ---- timing
    tm = {}
    for item in a.timing:
        lab, path = item.split("=")
        tm[lab] = json.loads(Path(path).read_text())
    if tm:
        U = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Retrieval latency per question (ms; mean $\\pm$ standard deviation over repeated passes, after warm-up). The cross-encoder rescored the dense top-20.}",
             "\\label{tab:timing}", "\\begin{tabular}{lrrrr}", "\\toprule", "Setting & Dense & Two-hop & Two-hop + gate & CE (dense top-20) \\\\", "\\midrule"]
        for lab, t in tm.items():
            def c(key):
                return f"{t[key]['ms_per_question_mean']:.0f} $\\pm$ {t[key]['ms_per_question_sd']:.0f}" if key in t else "--"
            U.append(f"Cloud CPU: {t["n_passages"]:,} passages & {c('dense')} & {c('two_hop')} & {c('two_hop_gate')} & {c('ce_dense_top20')} \\\\".replace(",", "\\,"))
            code_ = "".join(w.capitalize() for w in lab.replace("-", " ").split())
            for key, nm in (("dense", "Dense"), ("two_hop", "Two"), ("two_hop_gate", "Gate"), ("ce_dense_top20", "CE")):
                if key in t:
                    mac.append(f"\\newcommand{{\\Tm{code_}{nm}}}{{{t[key]['ms_per_question_mean']:.0f}}}")
                    mac.append(f"\\newcommand{{\\Tm{code_}{nm}Sd}}{{{t[key]['ms_per_question_sd']:.0f}}}")
            if "ce_dense_top20" in t:
                mac.append(f"\\newcommand{{\\Tm{code_}Ratio}}{{{t['ce_dense_top20']['ms_per_question_mean'] / t['two_hop']['ms_per_question_mean']:.1f}}}")
            mac.append(f"\\newcommand{{\\Tm{code_}Slow}}{{{t['two_hop']['ms_per_question_mean'] / t['dense']['ms_per_question_mean']:.1f}}}")
        U += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (out / "tables" / "ctl_timing.tex").write_text("\n".join(U) + "\n")

    (out / "tables" / "numbers_ctl.tex").write_text("% generated by kgrag.report_controls\n" + "\n".join(dict.fromkeys(mac)) + "\n")
    print("CTL_TABLES_WRITTEN", len(mac))


if __name__ == "__main__":
    main()
