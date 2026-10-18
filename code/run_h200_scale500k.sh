#!/usr/bin/env bash
# Large-pool (~500k passages) and blind-title validation on one H200. Run from the repo root:
#   source .venv/bin/activate
#   setsid nohup bash code/run_h200_scale500k.sh > run_h200_scale500k.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
OUT=../results/scale500k
mkdir -p "$OUT" ../cache
# 1) pool (dev + train), spaCy, BGE index, graph; 2) standard sample-C evaluation; 3) blind-title run; also the dev-only 66k pool for comparison
CHK=""; [ -f ../results_h200/eval_v2_seed2/dense.jsonl ] && CHK="--check_ids ../results_h200/eval_v2_seed2/dense.jsonl"
python -m kgrag.scale500k --pool_size 500000 --also_dev_pool --also_blind --cache ../cache --out "$OUT" $CHK 2>&1 | tee "$OUT/run.log"
# 4) tables
python -m kgrag.report_scale --res "$OUT" --out "$OUT/tables"
(cd .. && tar czf results_scale500k.tar.gz results/scale500k)
echo ALL_DONE
