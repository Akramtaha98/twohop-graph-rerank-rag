# twohop-graph-rerank-rag

Code and results for **"Graph-Augmented Retrieval for Multi-Hop Question Answering: Lightweight Entity-Graph Two-Hop Re-Ranking"** (anonymised for review).

A passage–entity graph (no LLM calls at indexing time) restricts which passages a dense retriever may boost after a first hop; the boost is a second dense scoring pass with a query expanded by the first-hop passage. Evaluated on HotpotQA (distractor dev, pooled corpus) with BGE-small and Qwen2.5-1.5B-Instruct.

## Layout
- `code/kgrag/` retrievers (BM25, dense, PPR hybrid, two-hop re-ranker, TitleLink, SPRIG-/LinearRAG-style baselines), graph, metrics, runner, table and figure generation
- `code/tests/` unit tests
- `results/` tuned hyper-parameters (`tuned*.json`), summaries and per-question files of every run reported in the paper; `results/controls/` holds the controls, coverage, corpus-scale and audit runs; `results/sample_ids/` the exact question ids
- `paper/` LaTeX source, generated tables and figures

## Hardware
Every number in the paper comes from one machine with a single NVIDIA H200 NVL GPU (driver 595.71.05; see `results/gpu.txt`).

## Reproduce (CUDA, e.g. H200)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt && python -m spacy download en_core_web_sm
cd code && python -m unittest discover -s tests
bash run_h200_all.sh     # tuning on train, all systems on the four sets, answers, controls, coverage, scale, latency
bash run_h200_fix.sh     # re-runs tune2/tune3 with the full grid (records the train tie) and the main HotpotQA run with all baselines
bash run_h200_abl.sh     # retrieval-only ablations of the earlier PPR hybrid
```
Settings were tuned on HotpotQA **train** questions only (`results/tuned2.json`; the train tie between m=2, gamma=0.2 and m=3, gamma=0.3 is recorded there and the frozen setting was kept).

## Tables and figures
```bash
cd code
H=../results
python -m kgrag.make_tables --runs "HotpotQA=$H/hotpot" --eval "SampleA=$H/eval_v2_seed0" "SampleB=$H/eval_v2_seed1" \
  --signif "HotpotQA (sample A)=$H/hotpot" "HotpotQA (sample B)=$H/eval_v2_seed1" --tuned $H/tuned.json --tuned2 $H/tuned2.json --out ../paper
python -m kgrag.extended --hotpot $H/hotpot --a $H/eval_v2_seed0 --b $H/eval_v2_seed1 --wiki $H/eval_2wiki --musique $H/eval_musique --out ../paper
python -m kgrag.analysis --a $H/eval_v2_seed0 --b $H/eval_v2_seed1 --out ../paper
python -m kgrag.report_controls --dir ../results/controls --out ../paper --timing "Gpu small=$H/timing_h200_1k.json" "Gpu full=$H/timing_h200_full.json"
python -m kgrag.report_gpu --res $H --out ../paper
python -m kgrag.make_figures --res $H --out ../paper/figs
python check_review3.py claims      # also: related cover build sampleids
```

## Baselines
`SprigLike` and `LinearLike` (`kgrag/retrievers.py`) are simplified re-implementations of the LinearRAG and SPRIG ideas with untuned defaults; they are not the authors' code.

## Data
HotpotQA, 2WikiMultiHopQA and MuSiQue are downloaded from the Hugging Face hub at run time. Models: `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`, `Qwen/Qwen2.5-1.5B-Instruct`, spaCy `en_core_web_sm`.

## License
MIT.
