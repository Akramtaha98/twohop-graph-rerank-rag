#!/usr/bin/env bash
# H200 revision run r6 (~30-40 min, retrieval only): (1) BM25-dense fusion and direct question-title baselines on the four sets;
# (2) other BGE encoders (base, large) with dense / TitleLink / gated two-hop (settings frozen from train, not re-tuned);
# (3) an additional HotpotQA sample C (seed 2) evaluated once with the frozen methods.
# Usage (repo root, venv active):  nohup bash code/run_h200_r6.sh > run_h200_r6.log 2>&1 &     Output: ../results_h200_r6.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
R=../results_h200
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")
B="ner=both,m=3,gamma=0.3,tb=0.5"; T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"
FUS=("fusion_rrf:bm25dense:mode=rrf" "fusion_linear:bm25dense:mode=linear" "qtitle:qtitle:tb=0.5,ner=both")
CORE=(bm25 dense "hybrid2:hybrid2:$B" "hybrid2_gate:hybrid2:$B,gate=$GATE" "titlelink:hybrid2:$T" "${FUS[@]}")
# (1) new baselines appended to the existing result folders
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_v2_seed0 --systems "${FUS[@]}"
python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --generator none --append --out $R/eval_v2_seed1 --systems "${FUS[@]}"
python -m kgrag.run --dataset 2wiki   --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_2wiki   --systems "${FUS[@]}"
python -m kgrag.run --dataset musique --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_musique --systems "${FUS[@]}"
# (2) other encoders (frozen settings)
for M in base large; do
  ENC=BAAI/bge-$M-en-v1.5
  for DS in "hotpot 0 enc_${M}_hotpotA" "hotpot 1 enc_${M}_hotpotB" "2wiki 0 enc_${M}_2wiki" "musique 0 enc_${M}_musique"; do
    set -- $DS
    python -m kgrag.run --dataset $1 --n 1000 --seed $2 --dense bge --dense_model $ENC --generator none --out $R/$3 --systems "${CORE[@]}" oracle
  done
done
# (3) untouched sample C (seed 2), evaluated once
python -m kgrag.run --dataset hotpot --n 1000 --seed 2 --dense bge --generator none --out $R/eval_v2_seed2 --systems "${CORE[@]}" "ce_dense:ce:top=20" oracle
tar czf ../results_h200_r6.tar.gz -C .. results_h200/eval_v2_seed0 results_h200/eval_v2_seed1 results_h200/eval_2wiki results_h200/eval_musique results_h200/eval_v2_seed2 results_h200/enc_*
echo DONE
