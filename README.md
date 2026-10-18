# twohop-graph-rerank-rag

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23445104-blue)](https://doi.org/10.5281/zenodo.23445104)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code, results and paper source for **"Graph-Augmented Retrieval for Multi-Hop Question Answering: Lightweight Entity-Graph Two-Hop Re-Ranking"**.

A passage–entity graph, built without any LLM calls, restricts which passages a dense retriever may boost after a first hop. The boost is a second dense scoring pass with a query expanded by the first-hop passage. A question-type gate decides when to apply it. **TitleLink**, a title-only variant, is included as a control and baseline.

## Main results (retrieval, All@5 = all supporting passages in the top 5, %)

| System | HotpotQA A | HotpotQA B | 2WikiMultiHopQA | MuSiQue |
|---|---|---|---|---|
| Dense (BGE-small) | 70.2 | 69.8 | 38.2 | 15.3 |
| SPRIG-style (re-implementation) | 70.3 | 69.9 | 38.3 | 15.3 |
| LinearRAG-style (re-implementation) | 62.0 | 62.0 | 44.0 | 13.0 |
| Two-hop + gate (ours) | 79.6 | 81.6 | 60.5 | 21.8 |
| TitleLink | 82.3 | 81.6 | 75.2 | 16.5 |

Settings were tuned on HotpotQA **train** only and frozen; 2WikiMultiHopQA and MuSiQue are zero-shot transfer. The SPRIG- and LinearRAG-style systems are simplified, untuned re-implementations (`SprigLike`, `LinearLike` in `code/kgrag/retrievers.py`), not the authors' code. Negative results are reported in the paper.

## Repository layout
```
code/kgrag/      graph, retrievers (BM25, dense, PPR hybrid, two-hop, TitleLink, baselines),
                 generation, metrics, runner, table and figure generation
code/tests/      unit tests
code/run_h200_*.sh   full reproduction scripts
results/         tuned hyper-parameters, summaries and per-question files of every reported run
results/controls/    controls, candidate coverage, audit and corpus-scale runs
results/sample_ids/  exact question ids of samples A and B (159 in common)
paper/           LaTeX source, generated tables and figures
```

## Hardware and software
All reported numbers come from one machine with a single NVIDIA H200 NVL GPU (driver 595.71.05, `results/gpu.txt`). Models: `BAAI/bge-small-en-v1.5` (dense), `BAAI/bge-reranker-base` (cross-encoder), `Qwen/Qwen2.5-1.5B-Instruct` (generator), spaCy `en_core_web_sm`. HotpotQA, 2WikiMultiHopQA and MuSiQue are downloaded from the Hugging Face hub at run time; no external API is used.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
python -m spacy download en_core_web_sm
cd code && python -m unittest discover -s tests
bash run_h200_all.sh     # tuning on train, all systems on the four sets, answers, controls, coverage, scale, latency
bash run_h200_fix.sh     # full tuning grid (records the train tie) and the main HotpotQA run with all baselines
bash run_h200_abl.sh     # retrieval-only ablations of the earlier PPR hybrid
```
Each script is resumable and writes a tarball of its results.

## Regenerate tables and figures
```bash
cd code; H=../results
python -m kgrag.make_tables --runs "HotpotQA=$H/hotpot" --eval "SampleA=$H/eval_v2_seed0" "SampleB=$H/eval_v2_seed1" \
  --signif "HotpotQA (sample A)=$H/hotpot" "HotpotQA (sample B)=$H/eval_v2_seed1" \
  --tuned $H/tuned.json --tuned2 $H/tuned2.json --out ../paper
python -m kgrag.extended --hotpot $H/hotpot --a $H/eval_v2_seed0 --b $H/eval_v2_seed1 --wiki $H/eval_2wiki --musique $H/eval_musique --out ../paper
python -m kgrag.analysis --a $H/eval_v2_seed0 --b $H/eval_v2_seed1 --out ../paper
python -m kgrag.report_controls --dir ../results/controls --out ../paper --timing "Gpu small=$H/timing_h200_1k.json" "Gpu full=$H/timing_h200_full.json"
python -m kgrag.report_gpu --res $H --out ../paper
python -m kgrag.make_figures --res $H --out ../paper/figs
python check_review3.py claims       # also: related cover build sampleids
```
Every table and number in the paper is generated from the stored per-question files.

## Notes on protocol
- Hyper-parameters were chosen on HotpotQA train questions only (`results/tuned*.json`). On train, (m=2, γ=0.2) and (m=3, γ=0.3) tie at 85.8%; the setting frozen before evaluation (m=3, γ=0.3) was kept and the tie is recorded in `tuned2.json`.
- Samples A (seed 0) and B (seed 1) are 1,000 HotpotQA questions each; sample B was reused after the gate and cross-encoder were added (see the paper's evaluation protocol).
- Latency is from a single device and software stack and is indicative.

## Citation
Please cite the archived release: https://doi.org/10.5281/zenodo.23445104 (version 1.1.0), and the paper once published.

## License
MIT.
