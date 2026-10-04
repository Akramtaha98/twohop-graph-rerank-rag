<div align="center">

# Two-Hop Graph Re-Ranking for Multi-Hop RAG

**Lightweight entity-graph re-ranking on top of a dense retriever. No LLM calls at indexing time.**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![task](https://img.shields.io/badge/task-multi--hop%20QA-orange)
![dataset](https://img.shields.io/badge/dataset-HotpotQA-purple)
![status](https://img.shields.io/badge/status-under%20review-lightgrey)

</div>

Code, tuned settings and per-question results for the paper *"Graph-Augmented Retrieval for Multi-Hop Question Answering: Lightweight Entity-Graph Two-Hop Re-Ranking"* (anonymised for review).

## Idea

Multi-hop questions often need a second passage that shares little vocabulary with the question. We find it in two steps:

1. **First hop.** A dense retriever (BGE-small) returns the top-*m* passages.
2. **Bridge entities.** Entities in those passages that are not in the question link to candidate passages through an offline passage-entity graph (rule-based and spaCy NER, IDF weights, hub removal).
3. **Second hop.** Candidates are re-scored by the same dense encoder with a query expanded by the first-hop passage. The boost is added to the dense score, so passages with no link keep their dense rank.

Cost: *m* extra dense query encodings per question, no extra LLM calls.

## Results (HotpotQA distractor dev, 1,000 questions per sample, pooled corpus)

| Metric | Dense (BGE-small) | Two-hop re-ranking |
|---|---|---|
| All@5, sample A | 70.2 | **78.6** |
| All@5, sample B (run once) | 69.8 | **80.3** |
| EM, Qwen2.5-1.5B, sample A | 37.5 | **42.6** |
| F1, Qwen2.5-1.5B, sample A | 47.6 | **53.2** |

All retrieval and answer gains: paired bootstrap, p < 0.001. Honest caveats, all reported in the paper:

- The gain comes from **bridge** questions. **Comparison** questions get worse.
- The lexical unsupported-answer rate does **not** change significantly, so we make no hallucination claim.
- A personalized-PageRank variant of the same graph gave **no gain** over dense retrieval (negative result).
- Only HotpotQA with a pooled corpus was tested.

## Repository layout

```
code/kgrag/    retrievers (BM25, dense, PPR hybrid, two-hop), graph, metrics, runner, tables
code/tests/    unit tests (metrics, entity extraction, PPR bounds, re-ranker)
results/       tuned hyper-parameters, summaries, per-question outputs
paper/         generated LaTeX tables
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements-mac.txt      # or code/requirements.txt (CUDA)
python -m spacy download en_core_web_sm
cd code && python -m unittest discover -s tests

bash run_mac.sh        # tune on train, run sample A with the generator
bash run_mac_v2.sh     # two-hop method: samples A and B, ablations, bootstrap
```

Hyper-parameters were tuned on HotpotQA **train** questions only (`results/tuned2.json`). Sample B (seed 1) was evaluated once.

## Data and models

HotpotQA is downloaded from the Hugging Face hub at run time. Models: `BAAI/bge-small-en-v1.5`, `Qwen/Qwen2.5-1.5B-Instruct`, spaCy `en_core_web_sm`. Every number in the paper's tables is regenerated from the stored results by `code/kgrag/make_tables.py`.

## License

MIT.
