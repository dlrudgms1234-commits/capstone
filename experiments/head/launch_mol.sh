#!/bin/bash
# 분자별 큰 헤드 학습 1개 (GINE → Mole-BERT 순서, 시드 3) + 끝나면 보고서 다시 만들기
D=/home/bising0922/Capstone/runs/tmp/head; P=/home/bising0922/micromamba/envs/aienv/bin
cd $D && $P/python make_molhead.py
G=$($P/python -c "
import torch
print(max(range(torch.cuda.device_count()), key=lambda g: torch.cuda.mem_get_info(g)[0]))" 2>/dev/null)
~/micromamba/bin/tmux new-session -d -s head_mol "cd $D && export TMPDIR=/home/bising0922/Capstone/runs/tmp MPLCONFIGDIR=/home/bising0922/Capstone/runs/tmp && CUDA_VISIBLE_DEVICES=$G taskset -c 28-47 nice -n 10 $P/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 head_mol.ipynb --output head_mol_out.ipynb > head_mol.log 2>&1; echo \"exit \$? \$(date +%m-%d_%H:%M)\" >> head_mol.log; cd /home/bising0922/Capstone && $P/python runs/tmp/ecfp/build_report.py > runs/tmp/head/rebuild_mol.log 2>&1"
echo "head_mol GPU $G $(date +%H:%M)" > /home/bising0922/Capstone/runs/_diag.txt
