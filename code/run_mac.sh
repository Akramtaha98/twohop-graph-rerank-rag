#!/usr/bin/env bash
# Apple-silicon (M1-M4) pipeline: tune -> retrieval for ALL systems -> LLM answers for the key systems -> tables -> PDF.
# Usage:   bash code/run_mac.sh                 (HotpotQA, no LoRA)
#          LORA=1 bash code/run_mac.sh          (adds LoRA fine-tune + eval)
#          N=300 bash code/run_mac.sh           (smaller / faster)
# Optional: MUSIQUE=/path/musique_ans_v1.0_dev.jsonl  WIKI=/path/2wiki_dev.json
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
N=${N:-1000}; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}; NTRAIN=${NTRAIN:-1000}
R=../results; GEN="none bm25 dense hybrid oracle"

python -m kgrag.tune --n 500 --dense bge --out $R/tuned.json
RUNS=()
run_ds () {  # name path label dir
  local extra=(); [[ -n "$2" ]] && extra=(--path "$2")
  python -m kgrag.run --dataset "$1" ${extra[@]+"${extra[@]}"} --n "$N" --dense bge --generator hf --model "$MODEL" \
    --tuned $R/tuned.json --gen_systems $GEN --out "$R/$4"
  RUNS+=("$3=$R/$4")
}
run_ds hotpot "" HotpotQA hotpot
[[ -n "${MUSIQUE:-}" ]] && run_ds musique "$MUSIQUE" MuSiQue musique
[[ -n "${WIKI:-}" ]] && run_ds 2wiki "$WIKI" 2Wiki 2wiki

if [[ "${LORA:-0}" == "1" ]]; then
  python -m kgrag.train_lora --model "$MODEL" --n_train "$NTRAIN" --dense bge --out ../adapters/lora_hotpot
  SPEC=$(python -c "import json;b=json.load(open('$R/tuned.json'))['best'];print('hybrid_lora:hybrid:alpha=%s,lam=%s,beta=%s'%(b['alpha'],b['lam'],b['beta']))")
  python -m kgrag.run --dataset hotpot --n "$N" --dense bge --generator hf --model "$MODEL" --adapter ../adapters/lora_hotpot \
    --systems "$SPEC" --append --out $R/hotpot
fi

python -m kgrag.make_tables --runs "${RUNS[@]}" --tuned $R/tuned.json --out ../paper
(cd ../paper && latexmk -pdf -interaction=nonstopmode main.tex) || echo "LaTeX not installed: open paper/ in Overleaf instead"
echo "DONE"
