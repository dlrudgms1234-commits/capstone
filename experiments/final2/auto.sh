#!/bin/bash
# 원인 찾기 3개(r_frag, r_head, r_valall)가 끝나면 → decide.py(규칙대로 설정 결정) → make_final2.py → 학습 대기열(동시 6개)
R=/home/bising0922/Capstone/runs/tmp/revert; F=/home/bising0922/Capstone/runs/tmp/final2; P=/home/bising0922/micromamba/envs/aienv/bin/python
cd $F
until [ $(grep -l "^exit " $R/r_frag.log $R/r_head.log $R/r_valall.log 2>/dev/null | wc -l) -eq 3 ]; do sleep 60; done
echo "$(date '+%m-%d %H:%M') 원인 찾기 끝: $(tail -qn 1 $R/r_frag.log $R/r_head.log $R/r_valall.log | tr '\n' ' ')" >> auto.log
$P decide.py >> auto.log 2>&1 || { echo "decide 실패" >> auto.log; exit 1; }
$P make_final2.py >> auto.log 2>&1 || { echo "make_final2 실패" >> auto.log; exit 1; }
# 설정이 예전 설계(조각 인코더 있음, 평균, 헤드 1024, 검증 쌍 전체)와 같으면 Mole-BERT 쌍은 예전 설계 시드 0–4와 같은 모델이므로 다시 학습하지 않는다
if $P -c "import json,sys; c=json.load(open('config.json')); sys.exit(0 if c==dict(readout='mean', frag=True, head='big', val_pairs='all') else 1)"; then
  grep -v "final2_mb_s" jobs.txt > jobs.tmp && mv jobs.tmp jobs.txt
  echo "$(date '+%m-%d %H:%M') 예전 설계와 같은 설정 → Mole-BERT 쌍은 예전 설계 결과 사용, 대기열에서 뺌" >> auto.log
fi
# 과적합 대책 후보 3개(Mole-BERT 쌍 시드 0)를 대기열 맨 앞에 넣는다
REG=$($P make_reg.py) || { echo "make_reg 실패" >> auto.log; exit 1; }
{ echo "$REG"; cat jobs.txt; } > jobs.tmp && mv jobs.tmp jobs.txt
echo "$(date '+%m-%d %H:%M') 대기열: $(tr '\n' ' ' < jobs.txt)" >> auto.log
$P run_queue.py 6 > run_queue.log 2>&1
echo "$(date '+%m-%d %H:%M') 최종 설계 2 학습 끝" >> auto.log
