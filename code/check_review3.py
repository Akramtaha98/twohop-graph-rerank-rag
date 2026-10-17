"""Gate checks for the third-review revision. Each check prints a success-only token when (and only when) it holds.

    python code/check_review3.py <name>
The numbers are re-computed from the per-question result files and compared with the macros the manuscript uses, so a table that was
not regenerated, or a macro that was copied by hand, fails.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAPER, CTL = ROOT / "paper", ROOT / "ctl"
TEX = (PAPER / "main.tex").read_text(errors="ignore")


def macros(path):
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", Path(path).read_text(errors="ignore"), re.M):
        out[m.group(1)] = m.group(2)
    return out


def recs(d, name):
    f = CTL / d / (name.replace(":", "_").replace("+", "_") + ".jsonl")
    return [json.loads(l) for l in f.read_text().splitlines()]


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def need(cond, msg):
    if not cond:
        fail(msg)


def num(s):
    return float(str(s).replace("$-$", "-").replace("+", "").replace("$<$", ""))


def related():
    bib = (PAPER / "refs.bib").read_text()
    for key in ("zhuang2025linearrag", "wang2026sprig", "melo2025griever"):
        need(f"@" in bib and f"{key}," in bib, f"bib key {key}")
        need(len(re.findall(key, TEX)) >= 3, f"{key} cited fewer than 3 times (text, table, intro)")
    need("\\label{sec:lightgraph}" in TEX, "related-work subsection missing")
    for name in ("LinearRAG", "SPRIG", "GRiever"):
        need(TEX.count(name) >= 4, f"{name} mentioned too rarely")
    need(re.search(r"LinearRAG \\citep\{zhuang2025linearrag\} &", TEX), "positioning table row for LinearRAG")
    print("RELATED_OK")


def controls():
    M = macros(PAPER / "tables" / "numbers_ctl.tex")
    for t in ("ctl_controls.tex", "ctl_sig.tex"):
        need((PAPER / "tables" / t).exists(), t + " missing")
    for t in ("tab:controls", "tab:ctlsig"):
        need(f"\\ref{{{t}}}" in TEX, f"text does not reference {t}")
    # recompute All@5 for the controls and compare with the macros
    for key, d in (("HA", "hotpot_A"), ("HB", "hotpot_B"), ("TW", "2wiki"), ("MQ", "musique")):
        for code, name in (("ExpF", "expand_all@0.3"), ("TitT", f"title_only@{M['CtlGammaTit']}"), ("TwoF", "two_hop@0.3")):
            v = 100 * np.mean([r["all_support"] for r in recs(d, name)])
            mv = num(M[f"Ctl{key}{code}All"])
            need(abs(v - mv) < 0.06, f"{key} {code}: recomputed {v:.2f} vs macro {mv}")
    need(not re.search(r"\\Ctl\w+\}\{\}", TEX), "empty control macro")
    print("CONTROLS_OK")


def audit():
    need((PAPER / "tables" / "ctl_audit.tex").exists(), "audit table missing")
    need("\\ref{tab:audit}" in TEX, "audit table not referenced")
    need("we did not analyse why" not in TEX, "old unexplained sentence still present")
    need("which we have not verified" not in TEX, "old speculation still present")
    M = macros(PAPER / "tables" / "numbers_ctl.tex")
    need(M.get("AudSizeBridgecomparison") == "4", "bridge-comparison support size is not 4 in the macros")
    sys.path.insert(0, str(ROOT / "code"))
    from kgrag import data as D

    exs, _ = D.load_2wiki(None, 1000, 0)
    sizes = {len(e["sup_pids"]) for e in exs if e["qtype"] == "bridge_comparison"}
    need(sizes == {4}, f"recomputed support sizes {sizes}")
    print("AUDIT_OK")


def coverage():
    M = macros(PAPER / "tables" / "numbers_ctl.tex")
    need((PAPER / "tables" / "ctl_coverage.tex").exists(), "coverage table missing")
    need("\\ref{tab:coverage}" in TEX, "coverage table not referenced")
    for key, d in (("HA", "hotpot_A"), ("HB", "hotpot_B"), ("TW", "2wiki"), ("MQ", "musique")):
        for code, name in (("Dense", "dense"), ("Two", "two_hop@0.3")):
            for mk, field in (("RecTwenty", "recall20"), ("AllTwenty", "all20")):
                v = 100 * np.mean([r[field] for r in recs(d, name)])
                mv = num(M[f"Cov{key}{code}{mk}"])
                need(abs(v - mv) < 0.06, f"{key} {code} {mk}: {v:.2f} vs {mv}")
    print("COVERAGE_OK")


def timing():
    need((PAPER / "tables" / "ctl_timing.tex").exists(), "timing table missing")
    t = json.loads((CTL / "timing_cpu_1k.json").read_text())
    need(t["warmup"] >= 5 and t["repeats"] >= 3, "warm-up/repeats too small")
    for k in ("dense", "two_hop", "two_hop_gate", "title_only"):
        need(t[k]["ms_per_question_sd"] >= 0 and len(t[k]["runs_ms"]) >= 3, f"{k} lacks repeated runs")
    need("warm-up" in TEX, "text does not mention warm-up")
    need("index" in TEX.lower() and "memory" in TEX.lower(), "text lacks indexing/memory")
    need("\\ExtSpeedRatio" not in TEX, "old single-run speed ratio still used")
    need("one eleventh" not in TEX and "eleven times" not in TEX, "old 11x headline remains")
    print("TIMING_OK")


def scale():
    s = json.loads((CTL / "scale_7405" / "summary.json").read_text())
    need(s["meta"]["n_passages"] >= 60000, "full pool not run")
    need((PAPER / "tables" / "ctl_scale.tex").exists() and "\\ref{tab:scale}" in TEX, "scale table not in paper")
    M = macros(PAPER / "tables" / "numbers_ctl.tex")
    v = s["rows"]["dense"]["all_support"]
    need(abs(v - num(M["ScaleFullDAll"])) < 0.06, "scale macro mismatch")
    print("SCALE_OK")


BANNED = ["keep their dense ranking", "keep their dense order", "statistically tied", "ties a cross", "matches a cross-encoder", "are untouched", "cannot demote"]


def claims():
    for b in BANNED:
        need(b not in TEX, f"banned phrase: {b}")
    need("their rank in the final list can fall" in TEX or "can still lose rank" in TEX, "unlinked-rank correction missing")
    need("\\TypMQGateTwohop" in TEX.split("\\end{abstract}")[0], "abstract lacks the MuSiQue two-hop numbers")
    need("over all passages" in TEX and "exhaustive" in TEX, "expanded-query scoring not stated")
    need("do not claim reduced hallucination" in TEX or "not claim reduced hallucination" in TEX, "restrained hallucination wording lost")
    need("\\nOverlapAB" in TEX, "sample overlap not disclosed")
    need("two independent samples" not in TEX and "independent samples" not in TEX, "'independent samples' still claimed")
    need("not significantly different" in TEX or "no significant" in TEX, "equivalence wording check")
    need("pooled" in TEX.split("\\end{abstract}")[0], "abstract does not scope the claims to pooled corpora")
    print("CLAIMS_OK")


def sampleids():
    d = ROOT / "results_ids"
    a = (d / "sample_ids_hotpot_A.txt").read_text().split()
    b = (d / "sample_ids_hotpot_B.txt").read_text().split()
    need(len(a) == 1000 and len(b) == 1000, "id files wrong size")
    ov = len(set(a) & set(b))
    M = macros(PAPER / "tables" / "numbers_ctl.tex")
    need(int(M["nOverlapAB"]) == ov, "overlap macro differs from recomputed overlap")
    need("\\CtlVDHXGateD" in TEX or "\\CtlVDHXTwoD" in TEX, "disjoint-subset result not reported in text")
    rel = ROOT / "release" / "results" / "sample_ids"
    need(rel.exists() and len(list(rel.glob("*.txt"))) >= 4, "sample ids not in release bundle")
    print("SAMPLEIDS_OK")


def cover():
    from docx import Document

    t = "\n".join(p.text for p in Document(ROOT / "Cover_Letter_AI_Open.docx").paragraphs)
    for must in ("no significant All@5 difference", "not evidence that the two are equivalent", "pooled corpora", "two-hop questions", "three- and four-hop", "Claude", "drafting the text"):
        need(must in t, "cover letter lacks: " + must)
    need("ties a cross" not in t and "19–25 October 2026]" not in t.replace("[Date: one day between 19 and 25 October 2026]", ""), "old wording")
    print("COVER_OK")


def build():
    p = subprocess.run("cd paper && pdflatex -interaction=nonstopmode main >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main", shell=True, cwd=ROOT, capture_output=True, text=True, errors="ignore")
    out = p.stdout
    need("Output written" in out, "no pdf")
    need("undefined" not in out.lower().replace("overfull", ""), "undefined reference/citation")
    log = (PAPER / "main.log").read_text(errors="ignore")
    need("Citation" not in "".join(re.findall(r"Citation.*undefined.*", log)), "undefined citation")
    t = subprocess.run("cd code && python -m unittest discover -s tests 2>&1 | tail -3", shell=True, cwd=ROOT, capture_output=True, text=True)
    need("OK" in t.stdout, "unit tests: " + t.stdout)
    print("BUILD_OK")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
