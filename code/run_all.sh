#!/usr/bin/env bash
# Full pipeline on real data: tune -> evaluate (retrieval + LLM answers) -> tables -> PDF.
# Needs: GPU (for the generator), `pip install -r code/requirements.txt`, internet for HotpotQA / models.
# Optional env vars:
#   N=1000            number of evaluation questions per dataset
#   DENSE=bge         dense scorer: bge (BAAI/bge-small-en-v1.5) or tfidf
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct
#   MUSIQUE=path/to/musique_ans_v1.0_dev.jsonl     WIKI=path/to/2wiki_dev.json   (skipped if unset)
#   LORA=1            also fine-tune a LoRA adapter and evaluate the hybrid with it
set -euo pipefail
cd "$(dirname "$0")"
N=${N:-1000}; DENSE=${DENSE:-bge}; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
R=../results

echo "== 1/5 tune hyper-parameters on a training sample"
python -m kgrag.tune --n 500 --dense "$DENSE" --out $R/tuned.json

RUNS=()
echo "== 2/5 HotpotQA"
python -m kgrag.run --dataset hotpot --n "$N" --dense "$DENSE" --generator hf --model "$MODEL" --tuned $R/tuned.json --out $R/hotpot
RUNS+=("HotpotQA=$R/hotpot")

if [[ -n "${MUSIQUE:-}" ]]; then
  echo "== MuSiQue"
  python -m kgrag.run --dataset musique --path "$MUSIQUE" --n "$N" --dense "$DENSE" --generator hf --model "$MODEL" --tuned $R/tuned.json --out $R/musique
  RUNS+=("MuSiQue=$R/musique")
fi
if [[ -n "${WIKI:-}" ]]; then
  echo "== 2WikiMultiHopQA"
  python -m kgrag.run --dataset 2wiki --path "$WIKI" --n "$N" --dense "$DENSE" --generator hf --model "$MODEL" --tuned $R/tuned.json --out $R/2wiki
  RUNS+=("2Wiki=$R/2wiki")
fi

if [[ "${LORA:-0}" == "1" ]]; then
  echo "== 3/5 LoRA fine-tuning and evaluation"
  python -m kgrag.train_lora --model "$MODEL" --n_train 2000 --dense "$DENSE" --out ../adapters/lora_hotpot
  python -m kgrag.run --dataset hotpot --n "$N" --dense "$DENSE" --generator hf --model "$MODEL" --adapter ../adapters/lora_hotpot \
    --tuned $R/tuned.json --systems hybrid_lora:hybrid:alpha=$(python -c "import json;print(json.load(open('$R/tuned.json'))['best']['alpha'])"),lam=$(python -c "import json;print(json.load(open('$R/tuned.json'))['best']['lam'])"),beta=$(python -c "import json;print(json.load(open('$R/tuned.json'))['best']['beta'])") \
    --out $R/hotpot_lora
fi

echo "== 4/5 tables and figure"
python -m kgrag.make_tables --runs "${RUNS[@]}" --tuned $R/tuned.json --out ../paper

echo "== 5/5 compile the paper"
(cd ../paper && latexmk -pdf -interaction=nonstopmode main.tex)
echo "DONE: paper/main.pdf"
