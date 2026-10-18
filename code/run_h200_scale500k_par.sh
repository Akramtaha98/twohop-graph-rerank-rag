#!/usr/bin/env bash
# Same experiment as run_h200_scale500k.sh, one configuration per GPU (4 GPUs in parallel). Run from the repo root with the venv active:
#   setsid nohup bash code/run_h200_scale500k_par.sh > run_h200_scale500k.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
OUT=../results/scale500k
mkdir -p "$OUT" ../cache
CHK=""; [ -f ../results_h200/eval_v2_seed2/dense.jsonl ] && CHK="--check_ids ../results_h200/eval_v2_seed2/dense.jsonl"
run() { # gpu, configuration
  CUDA_VISIBLE_DEVICES=$1 python -m kgrag.scale500k --pool_size 500000 --also_blind --only $2 --spacy_procs 8 --cache ../cache --out "$OUT" $CHK > "$OUT/run_$2.log" 2>&1 &
}
run 0 500k_std
run 1 500k_blind
run 2 66k_std
run 3 66k_blind
wait
python -m kgrag.report_scale --res "$OUT" --out "$OUT/tables"
(cd .. && tar czf results_scale500k.tar.gz results/scale500k)
echo ALL_DONE
