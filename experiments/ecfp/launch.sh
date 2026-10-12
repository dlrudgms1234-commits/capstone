#!/bin/bash
# ECFP + MLP 12개: [타깃별 쌍, 타깃별 분자별, 공용 쌍(누수 있음), 공용 쌍(누수 분자 빼기)] × 시드 3
# CPU 28–47번 코어, 우선순위 낮춤. 각 작업은 작은 MLP라 GPU 메모리 몇 GB
D=/home/bising0922/Capstone/runs/tmp/ecfp; P=/home/bising0922/micromamba/envs/aienv/bin/python
cd $D
i=0; GPUS=(5 5 5 0 0 2 2 4 5 0 2 4)
for a in "target pair none" "target mol none" "shared pair none" "shared pair drop_leak"; do
  for s in 0 1 2; do
    set -- $a
    n=ec_${1}_${2}_${3}_s$s
    ~/micromamba/bin/tmux new-session -d -s $n "cd $D && CUDA_VISIBLE_DEVICES=${GPUS[$i]} taskset -c 28-47 nice -n 10 $P ecfp_mlp.py $1 $2 $3 $s > $n.log 2>&1; echo \"exit \$? \$(date +%m-%d_%H:%M)\" >> $n.log"
    echo "$n GPU ${GPUS[$i]}"; i=$((i+1))
  done
done
