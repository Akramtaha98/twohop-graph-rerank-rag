#!/usr/bin/env bash
# CUDA GPU (e.g. H200): the review items that need the generator or a heavy cross-encoder pass.
#   1) TitleLink variant (graph + title link, no expanded query): retrieval on A/B/2Wiki/MuSiQue, plus a cross-encoder after it
#   2) Qwen2.5-1.5B answers: dense, gated two-hop, cross-encoder and TitleLink on sample B; TitleLink on sample A
#   3) cross-encoder common-pool comparison on samples A and B
# Usage: bash run_gpu_v4.sh      (safe to re-run, it appends; ~30-60 min on an H200, most of it graph building on CPU)
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
R=../results; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
B="ner=both,m=3,gamma=0.3,tb=0.5"
T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"   # gamma chosen on HotpotQA train (plateau from 1.0 upwards)
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")

for S in 0 1; do
  python -m kgrag.run --dataset hotpot --n 1000 --seed $S --dense bge --generator none --append --out $R/eval_v2_seed$S \
    --systems "titlelink:hybrid2:$T" "ce_titlelink:ce_h2:$T,top=20"
done
python -m kgrag.run --dataset 2wiki   --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_2wiki   --systems "titlelink:hybrid2:$T" "ce_titlelink:ce_h2:$T,top=20"
python -m kgrag.run --dataset musique --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_musique --systems "titlelink:hybrid2:$T" "ce_titlelink:ce_h2:$T,top=20"

python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --generator hf --model "$MODEL" \
  --systems dense "hybrid2_gate:hybrid2:$B,gate=$GATE" "ce_dense:ce:top=20" "titlelink:hybrid2:$T" --gen_systems dense hybrid2_gate ce_dense titlelink --append --out $R/eval_v2_seed1
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator hf --model "$MODEL" \
  --systems "titlelink:hybrid2:$T" --gen_systems titlelink --append --out $R/hotpot

for S in 0 1; do
  python -m kgrag.controls --dataset hotpot --n 1000 --seed $S --gammas 0.3 --cache $R/emb --ce --out $R/controls_ce/hotpot_$S
done
echo DONE
