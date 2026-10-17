#!/usr/bin/env bash
# H200 correction run (about 40 min): (1) tune2 with the full gamma grid used for the paper, then tune3; (2) HotpotQA sample A main run again
# (Graph-PPR / PPR hybrid with the regex extractor as in the original runs); (3) the LinearRAG-/SPRIG-style baselines with ner=both on all four sets.
# Usage (repo root, venv active):  nohup bash code/run_h200_fix.sh > run_h200_fix.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
R=../results_h200; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
python -m kgrag.tune2 --n 500 --dense bge --out $R/tuned2.json
python - <<'PY'
import json
d = json.load(open("../results_h200/tuned2.json"))
grid = d["grid"]; top = max(r["all_support"] for r in grid)
ties = [r for r in grid if abs(r["all_support"] - top) < 1e-12]
print("best train All@5", top, "| tied configurations:", [(r["ner"], r["m"], r["gamma"], r["tb"]) for r in ties])
# The paper froze m=3, gamma=0.3, tb=0.5, ner=both (selected on the train grid); the full grid has an exact tie with m=2, gamma=0.2. Keep the frozen entry and record the tie.
fz = [r for r in grid if (r["ner"], r["m"], r["gamma"], r["tb"]) == ("both", 3, 0.3, 0.5)][0]
assert abs(fz["all_support"] - top) < 1e-12, "frozen setting is not among the best on train"
d["best_by_argmax"] = d["best"]; d["best"] = fz; d["ties"] = ties
json.dump(d, open("../results_h200/tuned2.json", "w"), indent=2)
PY
python -m kgrag.tune3 --tuned2 $R/tuned2.json --n 500 --dense bge --out $R/tuned3.json
python - <<'PY'
import json
g=json.load(open("../results_h200/tuned3.json"))["best"]; print("tuned3 best", g)
assert g["gate"]=="titles", "gate changed"
PY
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")
B="ner=both,m=3,gamma=0.3,tb=0.5"; T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"
BL=("spriglike:spriglike:alpha=0.15,seeds_k=5,ner=both" "linearlike:linearlike:alpha=0.15,beta=0.5,hops=2,bridge_k=10,ner=both")
NEW=("hybrid2_gate:hybrid2:$B,gate=$GATE" "hybrid2_oracle:hybrid2:$B,gate=oracle" "ce_dense:ce:top=20" "ce_hybrid2:ce_h2:$B,top=20"
     "ce_hybrid2_gate:ce_h2:$B,gate=$GATE,top=20" "titlelink:hybrid2:$T" "ce_titlelink:ce_h2:$T,top=20" "${BL[@]}")
rm -rf $R/hotpot
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator hf --model "$MODEL" --tuned $R/tuned.json --tuned2 $R/tuned2.json \
  --systems none bm25 dense graph "hybrid:hybrid:alpha=0.15,lam=0.5,beta=0.3" "hybrid2:hybrid2:$B" "${NEW[@]}" oracle \
  --gen_systems none bm25 dense hybrid hybrid2 hybrid2_gate ce_dense titlelink spriglike linearlike oracle --out $R/hotpot
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_v2_seed0 --systems "${BL[@]}"
python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --generator none --append --out $R/eval_v2_seed1 --systems "${BL[@]}"
python -m kgrag.run --dataset 2wiki   --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_2wiki   --systems "${BL[@]}"
python -m kgrag.run --dataset musique --n 1000 --seed 0 --dense bge --generator none --append --out $R/eval_musique --systems "${BL[@]}"
tar czf ../results_h200_fix.tar.gz -C .. results_h200/tuned2.json results_h200/tuned3.json results_h200/hotpot results_h200/eval_v2_seed0 results_h200/eval_v2_seed1 results_h200/eval_2wiki results_h200/eval_musique
echo DONE
