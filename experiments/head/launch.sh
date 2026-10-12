#!/bin/bash
# 헤드 맞추기 6개 띄우기: 빈 메모리가 많은 GPU 순서로, CPU 28–47번 코어·우선순위 낮춤
D=/home/bising0922/Capstone/runs/tmp/head; P=/home/bising0922/micromamba/envs/aienv/bin
cd $D && $P/python make_head.py
GPUS=$($P/python -c "
import torch
f = sorted(((torch.cuda.mem_get_info(g)[0], g) for g in range(torch.cuda.device_count())), reverse=True)
print(' '.join(str(g) for _, g in f))" 2>/dev/null)
set -- $GPUS
G=($1 $2 $3 $1 $2 $3)   # 빈 카드 세 장에 둘씩
i=0
for n in head_mb_s0 head_mb_s1 head_mb_s2 head_gine_s0 head_gine_s1 head_gine_s2; do
  ~/micromamba/bin/tmux new-session -d -s $n "cd $D && export TMPDIR=/home/bising0922/Capstone/runs/tmp MPLCONFIGDIR=/home/bising0922/Capstone/runs/tmp && CUDA_VISIBLE_DEVICES=${G[$i]} taskset -c 28-47 nice -n 10 $P/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 $n.ipynb --output ${n}_out.ipynb > $n.log 2>&1; echo \"exit \$? \$(date +%m-%d_%H:%M)\" >> $n.log"
  echo "$n GPU ${G[$i]}" >> launch_result.txt; i=$((i+1))
done
echo "GPU 빈 순서: $GPUS" >> launch_result.txt
cp launch_result.txt /home/bising0922/Capstone/runs/_diag.txt
