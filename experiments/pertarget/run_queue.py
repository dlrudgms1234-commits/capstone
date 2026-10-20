"""jobs.txt의 노트북을 순서대로, 동시에 최대 MAXJ개 실행한다. GPU는 내 작업이 적은 카드 → 빈 메모리가 큰 카드 순으로 고른다.
끝난 작업(로그 마지막 줄 'exit 0')은 건너뛰므로 다시 실행해도 이어서 한다. 사용: python queue.py [MAXJ]"""
import os, subprocess, sys, time
import torch

OUT = '/home/bising0922/Capstone/runs/tmp/pertarget'; P = '/home/bising0922/micromamba/envs/aienv/bin'
MAXJ = int(sys.argv[1]) if len(sys.argv) > 1 else 6
jobs = open(f'{OUT}/jobs.txt').read().split()
log = lambda *m: print(time.strftime('%m-%d %H:%M:%S'), *m, flush=True)
done = lambda j: os.path.exists(f'{OUT}/logs/{j}.log') and open(f'{OUT}/logs/{j}.log').read().rstrip().endswith('exit 0')
os.makedirs(f'{OUT}/logs', exist_ok=True)
env = dict(os.environ, TMPDIR='/home/bising0922/Capstone/runs/tmp', MPLCONFIGDIR='/home/bising0922/Capstone/runs/tmp')
running = {}   # 작업 이름 → (프로세스, GPU)
todo = [j for j in jobs if not done(j)]
log(f'작업 {len(jobs)}개 중 남은 것 {len(todo)}개, 동시 {MAXJ}개')
while todo or running:
    for j, (p, g) in list(running.items()):
        if p.poll() is not None:
            with open(f'{OUT}/logs/{j}.log', 'a') as f: f.write(f'\nexit {p.returncode}\n')
            log(f'끝 {j} GPU {g} exit {p.returncode}'); del running[j]
    while todo and len(running) < MAXJ:
        j = todo.pop(0)
        mine = {g: sum(1 for _, gg in running.values() if gg == g) for g in range(torch.cuda.device_count())}
        g = min(mine, key=lambda g: (mine[g], -torch.cuda.mem_get_info(g)[0]))
        cmd = (f'cd {OUT}/jobs && CUDA_VISIBLE_DEVICES={g} taskset -c 28-47 nice -n 10 {P}/jupyter nbconvert --to notebook --execute '
               f'--ExecutePreprocessor.timeout=-1 {j}.ipynb --output {j}_out.ipynb > {OUT}/logs/{j}.log 2>&1')
        running[j] = (subprocess.Popen(cmd, shell=True, env=env), g); log(f'시작 {j} GPU {g}')
    time.sleep(30)
log('모두 끝')
