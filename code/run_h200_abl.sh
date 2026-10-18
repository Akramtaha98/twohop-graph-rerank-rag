#!/usr/bin/env bash
# H200: retrieval-only ablations of the earlier PPR hybrid on sample A (seed 0). ~5 min. Output: ../results_h200_abl.tar.gz
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
R=../results_h200; B="alpha=0.15,lam=0.5,beta=0.3"
python - <<'PY'
import json;b=json.load(open("../results_h200/tuned.json"))["best"];print("tuned.json best:",b)
assert (b["alpha"],b["lam"],b["beta"])==(0.15,0.5,0.3)
PY
rm -rf $R/abl_ppr
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator none --out $R/abl_ppr --systems dense \
 "hybrid:hybrid:$B" "abl_no_entity_seed:hybrid:alpha=0.15,lam=0,beta=0.3" "abl_no_dense_seed:hybrid:alpha=0.15,lam=1,beta=0.3" \
 "abl_no_fusion:hybrid:alpha=0.15,lam=0.5,beta=0" "abl_no_idf:hybrid:$B,idf=0" "abl_alpha05:hybrid:alpha=0.5,lam=0.5,beta=0.3"
tar czf ../results_h200_abl.tar.gz -C .. results_h200/abl_ppr
echo DONE
