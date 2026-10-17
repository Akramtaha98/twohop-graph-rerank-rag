#!/usr/bin/env bash
# H200 (CUDA): (1) like-for-like answers on sample A for every system on the same GPU, (2) re-implemented LinearRAG-style and SPRIG-style baselines
# on A, B, 2Wiki and MuSiQue (retrieval) and answers on A. Resumable-ish: re-run overwrites only these labels. About 40 min.
# Usage: bash code/run_gpu_v5.sh        (from the repo root, venv active)
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
R=${R:-../results}; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
B="ner=both,m=3,gamma=0.3,tb=0.5"
T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")
BL=("spriglike:spriglike:alpha=0.15,seeds_k=5" "linearlike:linearlike:alpha=0.15,beta=0.5,hops=2,bridge_k=10")

# 1) sample A answers, all systems on this GPU
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --ner both --generator hf --model "$MODEL" --out $R/answersA_gpu \
  --systems dense "hybrid2:hybrid2:$B" "hybrid2_gate:hybrid2:$B,gate=$GATE" "ce_dense:ce:top=20" "titlelink:hybrid2:$T" "${BL[@]}" \
  --gen_systems dense hybrid2 hybrid2_gate ce_dense titlelink spriglike linearlike

# 2) baselines, retrieval only, other sets
python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --ner both --generator none --append --out $R/eval_v2_seed1 --systems "${BL[@]}"
python -m kgrag.run --dataset 2wiki --n 1000 --seed 0 --dense bge --ner both --generator none --append --out $R/eval_2wiki --systems "${BL[@]}"
python -m kgrag.run --dataset musique --n 1000 --seed 0 --dense bge --ner both --generator none --append --out $R/eval_musique --systems "${BL[@]}"
cp $R/answersA_gpu/spriglike.jsonl $R/answersA_gpu/linearlike.jsonl $R/eval_v2_seed0/ 2>/dev/null || true
tar czf results_gpu_v5.tar.gz -C $R answersA_gpu eval_v2_seed0 eval_v2_seed1 eval_2wiki eval_musique
echo DONE
