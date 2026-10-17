# twohop-graph-rerank-rag

Code and results for **"Graph-Augmented Retrieval for Multi-Hop Question Answering: Lightweight Entity-Graph Two-Hop Re-Ranking"** (anonymised for review).

A passage–entity graph (no LLM calls at indexing time) restricts which passages a dense retriever may boost after a first hop; the boost is a second dense scoring pass with a query expanded by the first-hop passage. Evaluated on HotpotQA (distractor dev, pooled corpus) with BGE-small and Qwen2.5-1.5B-Instruct.

## Layout
- `code/kgrag/` retrievers (BM25, dense, PPR hybrid, two-hop re-ranker), graph, metrics, runner, table generation
- `code/tests/` unit tests (metrics, entity extraction, PPR bounds, re-ranker)
- `results/` tuned hyper-parameters (`tuned*.json`), summaries, and per-question retrieval/answer files for dense and the re-ranker
- `paper/` generated LaTeX tables

## Reproduce (Apple silicon or CUDA)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements-mac.txt        # or code/requirements.txt
cd code && python -m unittest discover -s tests
bash run_mac.sh          # tune on train, run sample A with the generator
bash run_mac_v2.sh       # two-hop method: samples A and B, ablations, bootstrap
```
Settings were tuned on HotpotQA **train** questions only (`results/tuned2.json`); sample B (seed 1) was evaluated once.

## Controls, coverage, scale and timing (review revision)
```bash
cd code && bash run_controls.sh      # expansion-only and title-link-only controls, sample B, 2Wiki, MuSiQue, corpus-scale pools (resumable)
python -m kgrag.timing --n 100 --repeats 3 --warmup 10 --ce --cache ../ctl/emb --out ../ctl/timing_cpu_1k.json
python -m kgrag.report_controls --dir ../ctl --out ../paper --timing cpuk=../ctl/timing_cpu_1k.json
python check_review3.py claims      # also: related controls audit coverage timing scale sampleids cover build
```
Exact sample identifiers are in `results/sample_ids/` (samples A and B overlap in 159 questions). Per-question control results are in `results/controls/`.

## GPU run for the generator-dependent items (CUDA, e.g. H200)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate sentence-transformers numpy scipy scikit-learn matplotlib datasets spacy rank_bm25
python -m spacy download en_core_web_sm
cd code && python -m unittest discover -s tests && bash run_gpu_v4.sh      # TitleLink retrieval, sample-B answers, TitleLink answers, cross-encoder common pool
```

After the GPU run: `python -m kgrag.report_gpu --res ../results --out ../paper` regenerates the TitleLink, common-pool and answer tables (`paper/tables/gpu_*.tex`).

## Data
HotpotQA is downloaded from the Hugging Face hub at run time. Models: `BAAI/bge-small-en-v1.5`, `Qwen/Qwen2.5-1.5B-Instruct`, spaCy `en_core_web_sm`.

## License
MIT.

## Baselines and like-for-like answers (GPU v5)
`code/run_gpu_v5.sh` runs simplified re-implementations of the LinearRAG and SPRIG ideas (`SprigLike`, `LinearLike` in `kgrag/retrievers.py`; not the authors' code, untuned defaults) on all four sets, and all systems' answers on sample A on one GPU (`results/answersA_gpu`). `python -m kgrag.report_gpu --res results --out paper` regenerates the tables.

Figures: `python -m kgrag.make_figures --res results --out paper/figs` (vector PDF and 300 dpi PNG from the stored per-question files).

## One-machine reproduction (NVIDIA H200)
`bash code/run_h200_all.sh` reruns every experiment in the paper (tuning, all systems on the four sets, answers, controls, coverage, scale, latency) on a single GPU server and writes `results_h200_all.tar.gz`. Retrieval results are device independent; generated answers can differ in under 1% of questions between devices.
