#!/bin/zsh
# Production runs: stage2 refits (CPU, all cores) and NN members (GPU), then build.py.
#   ./run_prod.sh lb     -> ../outputs/courageous-onion_v11.parquet   (leaderboard set)
#   ./run_prod.sh final  -> ../outputs/courageous-onion_final_new.parquet (final ranking set)
set -e
PY=../.venv/bin/python
P=../data/processed
O=${OUT:-../outputs/prod}
mkdir -p $O
if [[ $1 == lb ]]; then
  FEAT=$P/feat${G:-}.parquet; COMP=$P/compact.parquet; TPL=../data/raw/submitting.parquet; SUF=lb
  RUNS=("7 3126,300,346" "11 2077,300,218" "23 2550,300,575" "31 2550,300,575" "47 2550,300,575")
else
  FEAT=$P/feat_final${G:-}.parquet; COMP=$P/compact_final.parquet; TPL=../data/raw/final_submitting.parquet; SUF=fin
  RUNS=("7 3126,300,346" "11 2077,300,218" "23 2550,300,575" "31 2550,300,575" "47 2550,300,575" "59 2550,300,575" "71 2550,300,575")
fi
parts=()
for r in "${RUNS[@]}"; do
  s=${r%% *}; rd=${r##* }
  dump=(); [[ $s == 7 ]] && dump=(--dump-all $O/mat_${SUF})
  $PY stage2.py --feat $FEAT --compact $COMP --template $TPL --out $O --name run_${SUF}$s --tag ${SUF}$s \
      --stages 3 --seed $s --lr 0.05 --rounds $rd "${dump[@]}" 2>&1 | grep -vE "Warning"
  parts+=($O/parts_${SUF}$s.npz)
done
for s in 7 23 31; do
  $PY nn.py --mat $O/mat_${SUF}_fin --out $O/nn_${SUF}_s$s.npz --seed $s --final 2>&1 | grep -E "epoch 12|wrote|Error"
done
$PY -c "
import numpy as np
z=[np.load('$O/nn_${SUF}_s%d.npz'%s) for s in (7,23,31)]
assert all((x['id']==z[0]['id']).all() for x in z)
np.savez('$O/nn_${SUF}_avg.npz', id=z[0]['id'], nn=np.mean([x['nn'] for x in z],0))"
echo "PARTS ${(j:,:)parts}"
