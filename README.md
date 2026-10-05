<div align="center">

# Two-Hop Graph Re-Ranking for Multi-Hop RAG

**Lightweight entity-graph re-ranking on top of a dense retriever. No LLM calls at indexing time.**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![task](https://img.shields.io/badge/task-multi--hop%20QA-orange)
![datasets](https://img.shields.io/badge/datasets-HotpotQA%20%7C%202Wiki%20%7C%20MuSiQue-purple)
![status](https://img.shields.io/badge/status-under%20review-lightgrey)

</div>

Code, tuned settings and per-question results for the paper *"Graph-Augmented Retrieval for Multi-Hop Question Answering: Lightweight Entity-Graph Two-Hop Re-Ranking"*.

## Idea

Multi-hop questions often need a second passage that shares little vocabulary with the question. We find it in three steps:

1. **First hop.** A dense retriever (BGE-small) returns the top-*m* passages.
2. **Bridge entities.** Entities in those passages that are not in the question link to candidate passages through an offline passage-entity graph (rule-based and spaCy NER, IDF weights, hub removal).
3. **Second hop.** Candidates are re-scored by the same dense encoder with a query expanded by the first-hop passage. The boost is added to the dense score, so unlinked passages keep their dense rank.

A **question-type gate**, chosen on training data, skips the step for comparison-style questions. Cost: *m* extra dense query encodings per question, no LLM calls.

## Results (All@5 = share of questions with all supporting passages in the top 5, %)

| System | HotpotQA A | HotpotQA B | 2WikiMultiHopQA | MuSiQue |
|---|---|---|---|---|
| Dense (BGE-small) | 70.2 | 69.8 | 38.2 | 15.3 |
| Cross-encoder on dense top-20 | 81.2 | 81.0 | 38.7 | 18.0 |
| Two-hop re-ranking | 78.6 | 80.3 | 59.4 | 21.6 |
| Two-hop + question-type gate | 79.6 | 81.6 | 60.5 | 21.8 |
| Cross-encoder on two-hop top-20 | 85.3 | 86.8 | 47.4 | 22.9 |

Answers (Qwen2.5-1.5B-Instruct, HotpotQA sample A): EM 37.5 → 42.3, F1 47.6 → 52.8 (gated two-hop vs dense).

Caveats, all reported in the paper:

- On HotpotQA the two-hop method is statistically tied with the cross-encoder in All@5 and about 11x faster in retrieval time.
- The gain comes from bridge-type questions; the gate fixes most of the loss on comparison questions but is a small effect (+1 point).
- On MuSiQue the gain is limited to 2-hop questions; 3- and 4-hop questions are not helped.
- A cross-encoder placed after two-hop hurts on 2WikiMultiHopQA.
- The lexical unsupported-answer rate does **not** change significantly: no hallucination claim.
- A personalized-PageRank variant of the same graph gave **no gain** (negative result).
- Pooled corpora of 1,000 questions, Wikipedia-based data, one encoder.

## Repository layout

```
code/kgrag/    retrievers (BM25, dense, PPR hybrid, two-hop, gate, cross-encoder), graph, metrics, runner, tables
code/tests/    unit tests (metrics, entity extraction, PPR bounds, re-ranker, gate)
results/       tuned settings, summaries, per-question outputs, v3_report.txt (paired bootstrap)
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
bash run_mac_v3.sh     # gate, cross-encoder, 2WikiMultiHopQA, MuSiQue (about 1-2 h on an M4)
```

Settings were tuned on HotpotQA **train** questions only (`results/tuned2.json`, gate in `results/tuned3.json`). 2WikiMultiHopQA and MuSiQue are zero-shot transfer.

## Data and models

Datasets are downloaded from the Hugging Face hub at run time. Models: `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`, `Qwen/Qwen2.5-1.5B-Instruct`, spaCy `en_core_web_sm`. Every number in the paper's tables is regenerated from the stored results by `code/kgrag/make_tables.py` and `code/kgrag/extended.py`.

## License

MIT.
