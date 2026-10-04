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

## Data
HotpotQA is downloaded from the Hugging Face hub at run time. Models: `BAAI/bge-small-en-v1.5`, `Qwen/Qwen2.5-1.5B-Instruct`, spaCy `en_core_web_sm`.

## License
MIT.
