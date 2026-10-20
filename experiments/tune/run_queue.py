"""작업 폴더의 jobs.txt 노트북을 동시에 최대 MAXJ개 실행한다. GPU는 내 작업이 적은 카드 → 빈 메모리가 큰 카드 순.
끝난 작업(로그 마지막 줄 'exit 0')은 건너뛰므로 다시 실행하면 이어서 한다. 저장소 최상위에서 tmux 안에서 실행 권장
사용: python experiments/tune/run_queue.py <작업 폴더> [MAXJ=3]
환경변수: CPUS=28-47 (taskset으로 쓸 CPU 코어 제한, 공용 서버에서 권장), GPUS=0,1,2 (쓸 GPU만)"""
import os, subprocess, sys, time
import torch

OUT = os.path.abspath(sys.argv[1]); MAXJ = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BIN = os.path.dirname(sys.executable)
jobs = open(f'{OUT}/jobs.txt').read().split()
gpus = [int(g) for g in os.environ['GPUS'].split(',')] if os.environ.get('GPUS') else list(range(torch.cuda.device_count()))
pre = (f"taskset -c {os.environ['CPUS']} " if os.environ.get('CPUS') else '') + 'nice -n 10 '
log = lambda *m: print(time.strftime('%m-%d %H:%M:%S'), *m, flush=True)
done = lambda j: os.path.exists(f'{OUT}/logs/{j}.log') and open(f'{OUT}/logs/{j}.log').read().rstrip().endswith('exit 0')
os.makedirs(f'{OUT}/logs', exist_ok=True)
tmp = os.path.join(os.getcwd(), 'runs', 'tmp'); os.makedirs(tmp, exist_ok=True)
env = dict(os.environ, TMPDIR=tmp, MPLCONFIGDIR=tmp)
running = {}
todo = [j for j in jobs if not done(j)]
log(f'작업 {len(jobs)}개 중 남은 것 {len(todo)}개, 동시 {MAXJ}개, GPU {gpus}')
while todo or running:
    for j, (p, g) in list(running.items()):
        if p.poll() is not None:
            with open(f'{OUT}/logs/{j}.log', 'a') as f: f.write(f'\nexit {p.returncode}\n')
            log(f'끝 {j} GPU {g} exit {p.returncode}'); del running[j]
    while todo and len(running) < MAXJ:
        j = todo.pop(0)
        mine = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
        g = min(mine, key=lambda g: (mine[g], -torch.cuda.mem_get_info(g)[0]))
        cmd = (f'cd {OUT}/jobs && CUDA_VISIBLE_DEVICES={g} {pre}{BIN}/jupyter nbconvert --to notebook --execute '
               f'--ExecutePreprocessor.timeout=-1 {j}.ipynb --output {j}_out.ipynb > {OUT}/logs/{j}.log 2>&1')
        running[j] = (subprocess.Popen(cmd, shell=True, env=env), g); log(f'시작 {j} GPU {g}')
    time.sleep(30)
log('모두 끝')
