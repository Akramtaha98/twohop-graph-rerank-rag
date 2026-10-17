#!/usr/bin/env bash
# Apple silicon: the three extra experiments requested by reviewers (all settings frozen or tuned on TRAIN only).
#   1) question-type gate chosen on HotpotQA train (tune3)
#   2) HotpotQA samples A and B (retrieval): gate, oracle-gate upper bound, cross-encoder (bge-reranker-base on top-20 of dense), cross-encoder on top-20 of the two-hop re-ranker
#   3) second datasets 2WikiMultiHopQA and MuSiQue (retrieval, zero-shot transfer of the HotpotQA settings, 1000 questions each)
#   4) answers (Qwen2.5-1.5B) for the gated method and the cross-encoder on sample A
#   5) report with paired bootstrap and per-type All@5  -> results/v3_report.txt
# Usage: bash code/run_mac_v3.sh      (about 60-120 min on an M4; safe to re-run, it appends)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
R=../results; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
source ../.venv/bin/activate
pip install -q sentence-transformers datasets huggingface_hub "spacy>=3.7" && python -m spacy download en_core_web_sm >/dev/null
python -m unittest discover -s tests 2>&1 | tail -3

python -m kgrag.tune3 --tuned2 $R/tuned2.json --n 500 --dense bge --out $R/tuned3.json | tee $R/tune3.log
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")
echo "selected gate: $GATE"

B="ner=both,m=3,gamma=0.3,tb=0.5"     # frozen from tuned2.json
NEW=("hybrid2_gate:hybrid2:$B,gate=$GATE" "hybrid2_oracle:hybrid2:$B,gate=oracle"
     "ce_dense:ce:top=20" "ce_hybrid2:ce_h2:$B,top=20" "ce_hybrid2_gate:ce_h2:$B,gate=$GATE,top=20")
for S in 0 1; do
  python -m kgrag.run --dataset hotpot --n 1000 --seed $S --dense bge --generator none --append --out $R/eval_v2_seed$S --systems "${NEW[@]}"
done

BASE=(bm25 dense "hybrid2:hybrid2:$B" oracle)
python -m kgrag.run --dataset 2wiki    --n 1000 --seed 0 --dense bge --generator none --out $R/eval_2wiki   --systems "${BASE[@]}" "${NEW[@]}"
python -m kgrag.run --dataset musique  --n 1000 --seed 0 --dense bge --generator none --out $R/eval_musique --systems "${BASE[@]}" "${NEW[@]}"

python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator hf --model "$MODEL" \
  --systems "hybrid2_gate:hybrid2:$B,gate=$GATE" "ce_dense:ce:top=20" --gen_systems hybrid2_gate ce_dense --append --out $R/hotpot

python -m kgrag.report_v3 --dirs HotpotA=$R/eval_v2_seed0 HotpotB=$R/eval_v2_seed1 2Wiki=$R/eval_2wiki MuSiQue=$R/eval_musique --out $R/v3_report.txt
echo DONE
