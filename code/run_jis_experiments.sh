#!/usr/bin/env bash
# JIS revision experiments on one H200. Run from the repo's code/ folder (venv active), one stage at a time:
#   bash run_jis_experiments.sh mhrag      # MultiHop-RAG (non-Wikipedia news), 5 min - 1 h, retrieval only
#   bash run_jis_experiments.sh fulldev    # full dev sets of HotpotQA / 2Wiki / MuSiQue, several hours
#   bash run_jis_experiments.sh hipporag   # official HippoRAG 2 (needs a vLLM server, see README_JIS_EXPERIMENTS.md)
#   bash run_jis_experiments.sh mdr_export # writes inputs for the official MDR code; run MDR separately, then: bash run_jis_experiments.sh mdr_import
#   bash run_jis_experiments.sh report
# Settings are the FROZEN ones from the paper (tuned on HotpotQA train only). Nothing is re-tuned on any new dataset.
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
R=${RESULTS:-../results_jis}; NER=${NER:-both}
GATE=titles                                   # chosen on train (results/tuned3.json: gate=titles)
B="ner=$NER,m=3,gamma=0.3,tb=0.5"; T="ner=$NER,m=3,gamma=1.0,tb=0.5,mode=title_only"
CORE=(bm25 dense "two_hop:hybrid2:$B" "two_hop_gate:hybrid2:$B,gate=$GATE" "titlelink:hybrid2:$T"
      "ce_dense:ce:top=20" "ce_gate:ce_h2:$B,gate=$GATE,top=20" sprig:spriglike linear:linearlike "qtitle:qtitle:tb=0.5,ner=$NER" oracle)
run() { python -m kgrag.run --dataset "$1" --n "$2" --seed "$3" --dense bge --ner "$NER" --generator none --out "$4" --systems "${CORE[@]}" "${@:5}"; }
STAGE=${1:-}
case "$STAGE" in
mhrag)
  run multihoprag 100000 0 $R/mhrag
  KG_BLIND_TITLES=1 run multihoprag 100000 0 $R/mhrag_blind          # same corpus, headlines replaced by doc_<i>
  python -m kgrag.rank_report --dataset multihoprag --n 100000 --dir $R/mhrag --systems dense bm25 two_hop two_hop_gate titlelink ce_dense ce_gate sprig linear --latex $R/mhrag/table.tex ;;
fulldev)
  run hotpot  100000 0 $R/full_hotpot     # all 7,405 dev questions, pooled corpus (66,635 passages)
  run musique 100000 0 $R/full_musique    # all 2,417 answerable dev questions
  run 2wiki   100000 0 $R/full_2wiki      # all 12,576 dev questions (largest pool; slowest: use NER=regex to cut extraction time, and say so in the paper)
  for d in hotpot musique 2wiki; do
    python -m kgrag.rank_report --dataset $d --n 100000 --dir $R/full_$d --systems dense bm25 two_hop two_hop_gate titlelink ce_dense ce_gate sprig linear --latex $R/full_$d/table.tex
  done ;;
hipporag)
  # protocol of HippoRAG/LinearRAG: 1,000 questions per dataset, pooled corpus. Same samples as our paper (seed 0).
  LLM=${LLM:-Qwen/Qwen2.5-7B-Instruct}; EMB=${EMB:-nvidia/NV-Embed-v2}
  for cfg in "hotpot 1000 0 $R/hotpotA" "2wiki 1000 0 $R/2wiki" "musique 1000 0 $R/musique" "multihoprag 100000 0 $R/mhrag"; do
    set -- $cfg
    # make sure our systems exist for the same sample (skip if already run), then add HippoRAG 2
    [ -f "$4/dense.jsonl" ] || run "$1" "$2" "$3" "$4"
    python -m kgrag.hipporag_baseline --dataset "$1" --n "$2" --seed "$3" --out "$4" --llm_model "$LLM" --embed_model "$EMB"
  done ;;
mdr_export)
  for cfg in "hotpot 1000 0 hotpotA" "2wiki 1000 0 2wiki" "musique 1000 0 musique" "multihoprag 100000 0 mhrag"; do
    set -- $cfg; python -m kgrag.mdr_export --dataset "$1" --n "$2" --seed "$3" --out ../mdr_in/$4
  done ;;
mdr_import)
  for cfg in "hotpot 1000 0 hotpotA hotpotA" "2wiki 1000 0 2wiki 2wiki" "musique 1000 0 musique musique" "multihoprag 100000 0 mhrag mhrag"; do
    set -- $cfg; python -m kgrag.mdr_import --dataset "$1" --n "$2" --seed "$3" --mdr_out ../mdr_out/$4.json --id2doc ../mdr_in/$4/id2doc.json --out $R/$5
  done ;;
report)
  for d in hotpotA 2wiki musique mhrag; do
    ds=$d; [ $d = hotpotA ] && ds=hotpot; n=1000; [ $d = mhrag ] && { ds=multihoprag; n=100000; }
    S="dense bm25 two_hop two_hop_gate titlelink ce_dense ce_gate sprig linear"
    for x in hipporag2 mdr; do [ -f "$R/$d/$x.jsonl" ] && S="$S $x"; done
    echo "== $d"; python -m kgrag.rank_report --dataset $ds --n $n --dir $R/$d --systems $S --latex $R/$d/table.tex
  done ;;
*) echo "usage: bash run_jis_experiments.sh {mhrag|fulldev|hipporag|mdr_export|mdr_import|report}"; exit 1 ;;
esac
echo "STAGE $STAGE DONE"
