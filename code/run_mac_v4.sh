#!/usr/bin/env bash
# Apple silicon: the review items that need the generator or a heavy cross-encoder pass (everything else was run by run_controls.sh).
#   1) title-link variant (graph + title link, no expanded query): retrieval on A/B/2Wiki/MuSiQue with a cross-encoder after it
#   2) answers (Qwen2.5-1.5B) for dense, gated two-hop, cross-encoder and the title-link variant on sample B, and for the title-link variant on sample A
#   3) cross-encoder common-pool comparison (same candidate pool ordered by dense, two-hop and cross-encoder) on samples A and B
#   4) latency with warm-up and repeats on the laptop, for the 1k pool and the full 66k HotpotQA pool
# Usage: bash code/run_mac_v4.sh      (about 2 h on an M4; safe to re-run, it appends)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
R=../results; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
source ../.venv/bin/activate
B="ner=both,m=3,gamma=0.3,tb=0.5"
T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"      # gamma selected on HotpotQA train (plateau from 1.0 upwards)
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

python -m kgrag.timing --n 1000 --repeats 5 --warmup 10 --ce --cache $R/emb --out $R/timing_mac_1k.json
python -m kgrag.timing --n 1000 --pool_questions 7405 --repeats 3 --warmup 10 --cache $R/emb --out $R/timing_mac_full.json
echo DONE
