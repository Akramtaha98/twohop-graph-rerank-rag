#!/usr/bin/env bash
# Full reproduction of every experiment in the paper on ONE NVIDIA H200 (no laptop, no cloud CPU):
# tuning on train, HotpotQA A/B, 2WikiMultiHopQA, MuSiQue, answers, cross-encoder, TitleLink, baselines, controls, coverage, scale, latency.
# Usage (repo root, venv active):   nohup bash code/run_h200_all.sh > run_h200_all.log 2>&1 &
# Resumable: finished steps leave a marker in results_h200/.done and are skipped. Output: results_h200_all.tar.gz (about 2-3 h).
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
R=../results_h200; C=../ctl_h200; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
mkdir -p $R/.done $C
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee $R/gpu.txt
python -c "import torch,sys;assert torch.cuda.is_available(),'CUDA not available'"
step() { n=$1; shift; if [ -f "$R/.done/$n" ]; then echo "skip $n"; else echo "== $n"; "$@"; touch "$R/.done/$n"; fi; }

step tests python -m unittest discover -s tests
step tune1 python -m kgrag.tune  --n 500 --dense bge --out $R/tuned.json
step tune2 python -m kgrag.tune2 --n 500 --dense bge --out $R/tuned2.json
step tune2fz python - <<'PY'
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
step tune3 python -m kgrag.tune3 --tuned2 $R/tuned2.json --n 500 --dense bge --out $R/tuned3.json
GATE=$(python -c "import json;print(json.load(open('$R/tuned3.json'))['best']['gate'])")
B="ner=both,m=3,gamma=0.3,tb=0.5"                       # frozen from tuned2.json
T="ner=both,m=3,gamma=1.0,tb=0.5,mode=title_only"       # gamma chosen on train (control experiments)
BL=("spriglike:spriglike:alpha=0.15,seeds_k=5,ner=both" "linearlike:linearlike:alpha=0.15,beta=0.5,hops=2,bridge_k=10,ner=both")
NEW=("hybrid2_gate:hybrid2:$B,gate=$GATE" "hybrid2_oracle:hybrid2:$B,gate=oracle" "ce_dense:ce:top=20" "ce_hybrid2:ce_h2:$B,top=20"
     "ce_hybrid2_gate:ce_h2:$B,gate=$GATE,top=20" "titlelink:hybrid2:$T" "ce_titlelink:ce_h2:$T,top=20" "${BL[@]}")
BASE=(bm25 dense "hybrid:hybrid:alpha=0.15,lam=0.5,beta=0.3" "hybrid2:hybrid2:$B" oracle)
ABL=("abl2_no_title:hybrid2:ner=both,m=3,gamma=0.3,tb=0" "abl2_one_seed:hybrid2:ner=both,m=1,gamma=0.3,tb=0.5" "abl2_regex_ner:hybrid2:ner=regex,m=3,gamma=0.3,tb=0.5")

# main HotpotQA sample A: all systems, answers (Qwen2.5-1.5B) for the systems compared in the paper
step hotpotA python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator hf --model "$MODEL" --tuned $R/tuned.json --tuned2 $R/tuned2.json \
  --systems none bm25 dense graph "hybrid:hybrid:alpha=0.15,lam=0.5,beta=0.3" "hybrid2:hybrid2:$B" "${NEW[@]}" oracle \
  --gen_systems none bm25 dense hybrid hybrid2 hybrid2_gate ce_dense titlelink spriglike linearlike oracle --out $R/hotpot
# retrieval-only copy of sample A with ablations (same layout as before)
step evalA python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator none --out $R/eval_v2_seed0 --systems "${BASE[@]}" "${ABL[@]}" "${NEW[@]}"
# sample B: retrieval for everything, answers for dense, gated two-hop, cross-encoder, TitleLink
step evalB python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --generator hf --model "$MODEL" --out $R/eval_v2_seed1 --systems "${BASE[@]}" "${ABL[@]}" "${NEW[@]}" \
  --gen_systems dense hybrid2_gate ce_dense titlelink
step wiki  python -m kgrag.run --dataset 2wiki   --n 1000 --seed 0 --dense bge --generator none --out $R/eval_2wiki   --systems "${BASE[@]}" "${NEW[@]}"
step musq  python -m kgrag.run --dataset musique --n 1000 --seed 0 --dense bge --generator none --out $R/eval_musique --systems "${BASE[@]}" "${NEW[@]}"

# common-pool comparison and the controls (train-selected gamma), candidate coverage, 2Wiki audit, corpus scale (66,635 passages), all with the cross-encoder where relevant
for S in 0 1; do
  step ctlce$S python -m kgrag.controls --dataset hotpot --n 1000 --seed $S --gammas 0.3 --cache $C/emb --ce --out $R/controls_ce/hotpot_$S
done
export CE=1
step controls bash run_controls.sh $C

# latency on this machine (encoder and cross-encoder on the GPU, graph on the host CPU); warm-up and repeats
step time1k  python -m kgrag.timing --n 1000 --repeats 5 --warmup 10 --ce --cache $C/emb --out $R/timing_h200_1k.json
step timefull python -m kgrag.timing --n 1000 --pool_questions 7405 --repeats 3 --warmup 10 --ce --cache $C/emb --out $R/timing_h200_full.json

touch $R/.done/ALL
tar czf ../results_h200_all.tar.gz -C .. --exclude='ctl_h200/emb' results_h200 ctl_h200
ls -la ../results_h200_all.tar.gz
echo DONE
