#!/usr/bin/env bash
# Offline end-to-end smoke test on synthetic data. Verifies the plumbing only; never report these numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/build/smoke"
cd "$ROOT/code"
python -m kgrag.run --dataset toy --n 200 --generator extractive --out ../build/smoke/toy > ../build/smoke/toy.log 2>&1
python -m kgrag.make_tables --runs "Toy=../build/smoke/toy" --out ../build/smoke/paper > ../build/smoke/tables.log 2>&1
python - <<'EOF'
import json
s = json.load(open("../build/smoke/toy/summary.json"))["systems"]
need = ["none", "bm25", "dense", "graph", "hybrid", "abl_no_entity_seed", "abl_no_dense_seed", "abl_no_fusion", "abl_no_idf", "abl_alpha05", "oracle"]
keys = ["recall", "all_support", "em", "f1", "support", "unsupported"]
assert all(n in s for n in need), "missing system"
assert all(k in s[n] for n in need for k in keys), "missing metric"
assert s["oracle"]["recall"] == 1.0 and s["none"]["recall"] == 0.0
print("SMOKE_OK")
EOF
