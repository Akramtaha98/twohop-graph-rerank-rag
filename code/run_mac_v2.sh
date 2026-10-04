#!/usr/bin/env bash
# Apple-silicon (M1-M4): evaluate the v2 method (spaCy NER + two-hop re-ranking) once, with settings frozen from train-split tuning.
#   1) retrieval-only on two independent 1000-question samples (seed 0 and seed 1), all systems + ablations
#   2) answer generation (Qwen2.5-1.5B) for hybrid2 only, appended to results/hotpot (dense/hybrid/... already there)
#   3) tables + figure
# Usage: bash code/run_mac_v2.sh        (about 30-60 min on M4)
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
R=../results; MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
source ../.venv/bin/activate
pip install -q "spacy>=3.7" && python -m spacy download en_core_web_sm
python -m unittest discover -s tests 2>&1 | tail -3

B="ner=both,m=3,gamma=0.3,tb=0.5"   # frozen: tuned on HotpotQA TRAIN only (results/tuned2.json)
SYS=(bm25 dense "hybrid:hybrid:alpha=0.15,lam=0.5,beta=0.3" "hybrid2:hybrid2:$B"
     "abl2_no_title:hybrid2:ner=both,m=3,gamma=0.3,tb=0" "abl2_one_seed:hybrid2:ner=both,m=1,gamma=0.3,tb=0.5"
     "abl2_regex_ner:hybrid2:ner=regex,m=3,gamma=0.3,tb=0.5" oracle)
for S in 0 1; do
  python -m kgrag.run --dataset hotpot --n 1000 --seed $S --dense bge --generator none --out $R/eval_v2_seed$S --systems "${SYS[@]}"
done

# LLM answers for the new method only (same seed-0 questions as results/hotpot)
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator hf --model "$MODEL" \
  --systems "hybrid2:hybrid2:$B" --gen_systems hybrid2 --append --out $R/hotpot

python -m kgrag.make_tables --runs "HotpotQA=$R/hotpot" --tuned $R/tuned.json --out ../paper
python - <<'PY'
import json,numpy as np
from kgrag import metrics as M
R="../results"
for s in (0,1):
    def L(n): return [json.loads(l) for l in open(f"{R}/eval_v2_seed{s}/{n}.jsonl")]
    for key in ("all_support","recall"):
        a=[r[key] for r in L("hybrid2")]; b=[r[key] for r in L("dense")]
        d,lo,hi,p=M.paired_bootstrap(a,b)
        print(f"seed{s} {key}: hybrid2 {100*np.mean(a):.1f} vs dense {100*np.mean(b):.1f}  diff {100*d:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.4f}")
PY
echo DONE
