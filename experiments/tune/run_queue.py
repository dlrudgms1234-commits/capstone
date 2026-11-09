"""작업 폴더의 jobs.txt 노트북을 동시에 최대 MAXJ개 실행한다. GPU는 내 작업이 적은 카드 → 빈 메모리가 큰 카드 순.
끝난 작업(로그 마지막 줄 'exit 0')은 건너뛰므로 다시 실행하면 이어서 한다. 저장소 최상위에서 tmux 안에서 실행 권장
사용: python experiments/tune/run_queue.py <작업 폴더> [MAXJ=3]
환경변수 (공용 서버에서 다른 사용자·내 다른 작업에 영향이 가지 않게):
  CPUS=28-47       taskset으로 쓸 CPU 코어 제한 (nice 10은 항상)
  GPUS=0,1,2       쓸 GPU만
  MEM_SHARE=0.5    한 카드에서 이 대기열의 작업 전체가 쓸 수 있는 메모리 몫. 작업마다 MEM_SHARE / MAX_PER_GPU로 강제한다
  MAX_PER_GPU=2    한 카드에 올릴 이 대기열의 작업 수 상한
  BUSY_MEM=0.5     카드 메모리가 (누가 쓰든) 이 비율 이상 차 있으면 새 작업을 올리지 않는다
  BUSY_UTIL=80     내 작업이 없는 카드의 사용률(%)이 이 이상이면 (다른 사람이 쓰는 중) 피한다
조건에 맞는 카드가 없으면 기다렸다가 다시 확인한다. GPU 상태는 nvidia-smi로만 본다 (이 프로세스는 GPU 메모리를 잡지 않는다)"""
import os, subprocess, sys, time

OUT = os.path.abspath(sys.argv[1]); MAXJ = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BIN = os.path.dirname(sys.executable)
MEM_SHARE, MAX_PER_GPU = float(os.environ.get('MEM_SHARE', 0.5)), int(os.environ.get('MAX_PER_GPU', 2))
BUSY_MEM, BUSY_UTIL = float(os.environ.get('BUSY_MEM', 0.5)), float(os.environ.get('BUSY_UTIL', 80))
FRAC = MEM_SHARE / MAX_PER_GPU   # 작업 하나의 메모리 상한 (노트북이 GPU_MEM_FRAC로 읽어 강제)

def smi():
    """{GPU 번호: (사용 메모리 비율, 사용률 %)}"""
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): (float(u) / float(t), float(p)) for i, u, t, p in (l.split(', ') for l in out.strip().splitlines())}

jobs = open(f'{OUT}/jobs.txt').read().split()
gpus = [int(g) for g in os.environ['GPUS'].split(',')] if os.environ.get('GPUS') else sorted(smi())
pre = (f"taskset -c {os.environ['CPUS']} " if os.environ.get('CPUS') else '') + 'nice -n 10 '
log = lambda *m: print(time.strftime('%m-%d %H:%M:%S'), *m, flush=True)
done = lambda j: os.path.exists(f'{OUT}/logs/{j}.log') and open(f'{OUT}/logs/{j}.log').read().rstrip().endswith('exit 0')
os.makedirs(f'{OUT}/logs', exist_ok=True)
tmp = os.path.join(os.getcwd(), 'runs', 'tmp'); os.makedirs(tmp, exist_ok=True)
env = dict(os.environ, TMPDIR=tmp, MPLCONFIGDIR=tmp, GPU_MEM_FRAC=f'{FRAC:.4f}')
running, waiting = {}, False
todo = [j for j in jobs if not done(j)]
log(f'작업 {len(jobs)}개 중 남은 것 {len(todo)}개, 동시 {MAXJ}개, GPU {gpus} | 카드당 내 작업 {MAX_PER_GPU}개·작업당 메모리 {FRAC:.0%} 이하, '
    f'카드 메모리 {BUSY_MEM:.0%} 이상 또는 남의 사용률 {BUSY_UTIL:.0f}% 이상이면 피함')
while todo or running:
    for j, (p, g) in list(running.items()):
        if p.poll() is not None:
            with open(f'{OUT}/logs/{j}.log', 'a') as f: f.write(f'\nexit {p.returncode}\n')
            log(f'끝 {j} GPU {g} exit {p.returncode}'); del running[j]
    while todo and len(running) < MAXJ:
        st = smi()
        mine = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
        ok = [g for g in gpus if mine[g] < MAX_PER_GPU and st[g][0] < BUSY_MEM and (mine[g] > 0 or st[g][1] < BUSY_UTIL)]
        if not ok:
            if not waiting: log('조건에 맞는 GPU가 없어 기다린다 (' + ', '.join(f'GPU {g}: 메모리 {st[g][0]:.0%}, 사용률 {st[g][1]:.0f}%, 내 작업 {mine[g]}' for g in gpus) + ')')
            waiting = True; break
        waiting = False
        j = todo.pop(0)
        g = min(ok, key=lambda g: (mine[g], st[g][0]))
        cmd = (f'cd {OUT}/jobs && CUDA_VISIBLE_DEVICES={g} {pre}{BIN}/jupyter nbconvert --to notebook --execute '
               f'--ExecutePreprocessor.timeout=-1 {j}.ipynb --output {j}_out.ipynb > {OUT}/logs/{j}.log 2>&1')
        running[j] = (subprocess.Popen(cmd, shell=True, env=env), g); log(f'시작 {j} GPU {g}')
        time.sleep(20)   # 방금 올린 작업이 메모리를 잡은 뒤에 다음 카드 상태를 본다
    time.sleep(30)
log('모두 끝')
