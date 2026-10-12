# 재현에 필요한 캐시 목록 만들기 (최종 결과에 쓰인 것만)
import glob, os
os.chdir('/home/bising0922/Capstone')
T = 't30_h64_dr0.3_wd0.0001'
pats = {
    '원본 데이터 (MoleculeACE CSV)': ['data/moleculeace/*.csv'],
    '쌍 캐시 (우리 분할 v2, 팀원 분할 v3)': ['data/pairs/*_v2.pkl', 'data/pairs/*_v3.pkl'],
    'Mole-BERT 사전학습 가중치': ['runs/pretrained/Mole-BERT.pth'],
    # 가중치·학습 기록: ① 누수 있음(v2) ② 팀원(v3) ③ 우리(leakdrop) + 대조(rnddrop) ④ 큰 헤드(head1024_leakdrop) / 편집 정보 빼기(noedit, nonchanged)
    '쌍 모델 가중치·기록': [f'runs/{k}_a0_{T}_exp1_anchored_{v}_seed?.{x}' for k, x in [('gnn', 'pt'), ('hist', 'csv')]
                     for v in ['v2', 'v2_molebert', 'v3', 'v3_molebert', 'v2_leakdrop', 'v2_leakdrop_molebert', 'v2_rnddrop', 'v2_rnddrop_molebert',
                               'v2_head1024_leakdrop', 'v2_head1024_leakdrop_molebert', 'v2_noedit', 'v2_nonchanged']],
    '분자별 모델 가중치·기록': [f'runs/{k}_{T[:3]}_h64_l3_dr0.3_wd0.0001_ep1000{v}_seed?.{x}' for k, x in [('gnnmol', 'pt'), ('histmol', 'csv')]
                       for v in ['', '_molebert', '_v3', '_v3_molebert', '_leakdrop', '_leakdrop_molebert', '_rnddrop', '_rnddrop_molebert', '_leakdrop_head1024', '_leakdrop_head1024_molebert']],
    '예측값 저장본 (노트북 빠른 재실행)': ['runs/preds_*_v2_*.pkl', 'runs/preds_*_v3_*.pkl', 'runs/preds_mol_*ep1000_[0-9]*.pkl', 'runs/preds_molpt_*ep1000_molebert_*.pkl',
                              'runs/preds_mol*_v3*.pkl', 'runs/preds_mol*leakdrop*.pkl', 'runs/preds_mol*head1024*.pkl', 'runs/preds_mol*rnddrop*.pkl'],
    '실험 예측값 (ECFP + MLP, 헤드, 편집 정보, XAI)': ['runs/tmp/ecfp/pred_*_s?.pkl', 'runs/tmp/ecfp/report3.pkl', 'runs/tmp/head/pred_head_*_s?.pkl',
                                          'runs/tmp/ablC/pred_ablC_*_s?.pkl', 'runs/tmp/xai/xai_*.pkl', 'runs/tmp/mace/mlp_*.yml'],
}
files, out = [], []
for name, ps in pats.items():
    fs = sorted(set(f for p in ps for f in glob.glob(p)))
    fs = [f for f in fs if 'smoke' not in f]
    sz = sum(os.path.getsize(f) for f in fs)
    out.append(f'{name}: {len(fs)}개, {sz / 2**20:.0f}MB'); files += fs
files = sorted(set(files))
out.append(f'합계 {len(files)}개, {sum(os.path.getsize(f) for f in files) / 2**20:.0f}MB')
open('runs/tmp/pack_files.txt', 'w').write('\n'.join(files) + '\n')
open('runs/_diag.txt', 'w').write('\n'.join(out) + '\n')
