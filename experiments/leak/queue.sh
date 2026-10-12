#!/bin/bash
# 누수 실험 대기열: 동시에 최대 MAXJ개, 남은 메모리가 넉넉할 때만 다음 작업 시작 (공용 서버)
# 모든 학습이 끝나면 조건별 전체 노트북(예측값·표)을 차례로 실행
CAP=/home/bising0922/Capstone; D=$CAP/runs/tmp/leak; L=$D/queue.log; MAXJ=7
cd $D; export TMPDIR=$CAP/runs/tmp MPLCONFIGDIR=$CAP/runs/tmp
J=/home/bising0922/micromamba/envs/aienv/bin/jupyter
log() { echo "$(date '+%m-%d %H:%M') $*" >> $L; }
running() { ~/micromamba/bin/tmux ls 2>/dev/null | grep -c '^lk_'; }
pick_gpu() {   # 내 작업(살아 있는 것 + 팀원 노트북 GPU 1)이 가장 적은 카드, 같으면 남은 메모리가 많은 카드 (15GB 이상 남은 것만)
  live=$( (for s in $(~/micromamba/bin/tmux ls -F '#{session_name}' 2>/dev/null | grep '^lk_'); do grep "^${s#lk_} " assign.txt; done | awk '{print $2}'; echo 1) )
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    while IFS=', ' read g u t; do
      [ $((t-u)) -gt 15000 ] || continue
      echo "$(echo "$live" | grep -cx $g) $((100000-(t-u))) $g"
    done | sort -n -k1,1 -k2,2 | head -1 | cut -d' ' -f3
}
log "대기열 시작 (CPU 28–47번 코어로 제한)"
for name in $(cat jobs.txt); do
  if ~/micromamba/bin/tmux has-session -t lk_$name 2>/dev/null || [ -f $name.log ]; then continue; fi   # 이미 시작한 작업
  while true; do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    gpu=$(pick_gpu)
    if [ "$(running)" -lt $MAXJ ] && [ "$avail" -gt 100 ] && [ -n "$gpu" ]; then break; fi
    sleep 120
  done
  ~/micromamba/bin/tmux new-session -d -s lk_$name "cd $D && CUDA_VISIBLE_DEVICES=$gpu taskset -c 28-47 nice -n 10 $J nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 $name.ipynb --output ${name}_out.ipynb > $name.log 2>&1; echo \"exit \$? \$(date +%m-%d_%H:%M)\" >> $name.log"
  echo "$name $gpu" >> assign.txt
  log "시작 $name (GPU $gpu, 남은 메모리 ${avail}GB)"
  sleep 300   # 앞 작업이 메모리를 다 잡을 때까지 기다렸다가 다음 판단
done
while [ "$(running)" -gt 0 ]; do sleep 120; done
bad=$(for n in $(cat jobs.txt); do grep -q '^exit 0' $n.log || echo $n; done)
if [ -n "$bad" ]; then log "중단: 실패한 학습 $bad"; exit 1; fi
log "학습 모두 끝남 → 조건별 전체 실행"
for f in rnd_full leak_full; do
  gpu=$(pick_gpu)
  CUDA_VISIBLE_DEVICES=${gpu:-0} taskset -c 28-47 nice -n 10 $J nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 $f.ipynb --output ${f}_out.ipynb > $f.log 2>&1
  log "$f exit $?"
done
log "끝"
