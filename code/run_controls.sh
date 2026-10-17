#!/usr/bin/env bash
# Controls, candidate coverage and corpus-scale runs (retrieval only). Settings frozen from tuned2.json; extra gamma values are for
# train-only selection. Resumable: a finished run (summary.json present) is skipped.
# Usage: bash code/run_controls.sh [results_dir]   (add CE=1 to also run the cross-encoder common-pool comparison)
set -euo pipefail
cd "$(dirname "$0")"
R=${1:-../ctl}; G="0.3 0.1 0.2 0.5 1.0 1.5 2.0 3.0 5.0"; CEF=""; [ "${CE:-0}" = "1" ] && CEF="--ce"
export TOKENIZERS_PARALLELISM=false
run() { out=$1; shift; if [ -f "$R/$out/summary.json" ]; then echo "skip $out"; else python -m kgrag.controls "$@" --cache $R/emb --out $R/$out; fi; }
run train      --dataset hotpot --split train --n 500 --seed 0 --gammas $G
run hotpot_A   --dataset hotpot --n 1000 --seed 0 --gammas $G $CEF
run hotpot_B   --dataset hotpot --n 1000 --seed 1 --gammas $G $CEF
run 2wiki      --dataset 2wiki --n 1000 --seed 0 --gammas $G
run musique    --dataset musique --n 1000 --seed 0 --gammas $G
# corpus scale: the SAME 1000 questions (sample A) against pools built from more HotpotQA dev questions
run scale_7405 --dataset hotpot --n 1000 --seed 0 --pool_questions 7405 --gammas 0.3 1.0
run scale_3000 --dataset hotpot --n 1000 --seed 0 --pool_questions 3000 --gammas 0.3 1.0
echo ALL_DONE
