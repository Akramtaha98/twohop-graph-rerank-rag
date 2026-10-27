# JIS revision experiments (H200)

Everything here adds to your released repo; nothing existing is changed except two small, idempotent patches (`patch_jis.py`):
`data.py` (MultiHop-RAG loader, optional blind titles via `KG_BLIND_TITLES=1`) and `run.py` (adds `multihoprag`, stores each system's ranked top-5).

## Setup (once)
Already applied on the `jis-experiments` branch (no copying or patching needed):
```bash
git fetch origin && git checkout jis-experiments && git pull
cd code && source ../.venv/bin/activate
```

## 1. MultiHop-RAG (most valuable; ~10 min)
`bash run_jis_experiments.sh mhrag` -> `../results_jis/mhrag` (+ `mhrag_blind`). Frozen settings, no tuning, no training on it.
Look at the first printed lines: field names and "evidence items without a matching article title" should be ~0. If the loader fails, download the two JSON files from the dataset page and pass `--path corpus.json,MultiHopRAG.json` (edit the `run` calls).
Expected honest outcome: news headlines are rarely mentioned in other articles, so TitleLink may give little. Report whatever happens; that is the test of the title-metadata thesis.

## 2. Full dev sets (hours)
`bash run_jis_experiments.sh fulldev`. Run each dataset in its own `tmux`/`nohup`. For 2Wiki the entity extraction dominates; `NER=regex bash ...` is allowed if stated (regex-only equalled spaCy+regex in our ablation).

## 3. Official HippoRAG 2
```bash
pip install hipporag vllm        # separate venv recommended (python 3.10)
vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 8192 --port 8000 &     # or Llama-3.1-8B-Instruct
bash run_jis_experiments.sh hipporag
```
Runs on the same 1,000-question pooled samples as the paper (HippoRAG's own protocol). Indexing makes one LLM OpenIE call per passage (about 10k passages per dataset). The paper must say which LLM/embedding model you used; an 8B indexer is weaker than the GPT-4o-mini / 70B setups in HippoRAG's papers. `retrieve()` is not in the README: if the package version differs, edit the lines marked `# API:` in `hipporag_baseline.py`.

## 4. MDR (official code, Python 3.6 env, separate)
```bash
bash run_jis_experiments.sh mdr_export            # writes ../mdr_in/<set>/{corpus.jsonl,questions.jsonl}
git clone https://github.com/facebookresearch/multihop_dense_retrieval && cd multihop_dense_retrieval && bash setup.sh
bash ./scripts/download_hotpot.sh                 # models/ (~2 GB) -- the 20 GB data download is not needed
for S in hotpotA 2wiki musique mhrag; do
  python scripts/encode_corpus.py --do_predict --predict_batch_size 1000 --model_name roberta-base \
     --predict_file ../mdr_in/$S/corpus.jsonl --init_checkpoint models/q_encoder.pt \
     --embed_save_path ../mdr_in/$S/index --fp16 --max_c_len 300 --num_workers 8
  python scripts/eval/eval_mhop_retrieval.py ../mdr_in/$S/questions.jsonl <index .npy written above> <id2doc json written above> models/q_encoder.pt \
     --batch-size 100 --beam-size 20 --topk 20 --shared-encoder --model-name roberta-base --gpu --save-path ../mdr_out/$S.json
done
bash run_jis_experiments.sh mdr_import
```
Three things MDR's README does not specify and you must check once: the file names `encode_corpus.py` writes, the question-file schema `eval_mhop_retrieval.py` reads (compare with `data/hotpot/hotpot_qas_val.json`), and the key under which `--save-path` stores chains (`mdr_import.py` prints the keys it finds and stops if it cannot match). MDR is trained on HotpotQA, so 2Wiki, MuSiQue and MultiHop-RAG are zero-shot for it, as for our method; say so.

## 5. Tables
`bash run_jis_experiments.sh report` prints R@2, R@5, All@2, All@5 (and document-level All@5 for MultiHop-RAG) with paired bootstrap against dense retrieval and writes `table.tex` per dataset. Send me the `results_jis` folder (or the printed tables) and I will fold them into the paper and rewrite the claims to match, whichever way they come out.
