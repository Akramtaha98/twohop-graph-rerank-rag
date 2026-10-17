#!/bin/bash
B="ner=both,m=3,gamma=0.3,tb=0.5"
python -m kgrag.run --dataset hotpot --n 1000 --seed 0 --dense bge --generator none --out results/eval_v2_seed0b --systems dense "abl2_regex_ner:hybrid2:ner=regex,m=3,gamma=0.3,tb=0.5" oracle
python -m kgrag.run --dataset hotpot --n 1000 --seed 1 --dense bge --generator none --out results/eval_v2_seed1 \
 --systems bm25 dense "hybrid:hybrid:alpha=0.15,lam=0.5,beta=0.3" "hybrid2:hybrid2:$B" "abl2_no_title:hybrid2:ner=both,m=3,gamma=0.3,tb=0" "abl2_one_seed:hybrid2:ner=both,m=1,gamma=0.3,tb=0.5" "abl2_regex_ner:hybrid2:ner=regex,m=3,gamma=0.3,tb=0.5" oracle
