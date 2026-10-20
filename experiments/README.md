# 추가 실험 재현 방법

`ac_pipeline.ipynb` 밖에서 돌린 실험(누수 실험, ECFP + MLP, 예측 헤드 크기, 편집 정보 빼 보기, XAI)과
결과 보고 PDF(`누수_비교_보고서.pdf`, `결과_보고_헤드수정.pdf`)를 만드는 코드다. PDF 자체는 저장소에서 뺐고, 아래 2절 명령으로 다시 만들 수 있다.

## 0. 준비

1. **캐시 받기** — 학습된 가중치·쌍 캐시·예측값 묶음(`capstone_cache_*.tar.gz`, 약 640MB)을 저장소 최상위에 풀면
   `data/`, `runs/` 아래 같은 자리에 들어간다. 이게 있으면 학습 없이 표·그림·PDF를 다시 만들 수 있다.
   ```bash
   tar -xzf capstone_cache_2026-10-12.tar.gz      # 저장소 최상위에서
   sha256sum -c capstone_cache_2026-10-12.sha256   # 받은 파일 확인 (선택)
   ```
2. **스크립트를 작업 위치로 복사** — 스크립트는 `runs/tmp/<폴더>/` 에서 서로를 참조하도록 짜여 있다.
   ```bash
   mkdir -p runs/tmp && cp -r experiments/. runs/tmp/
   ```
3. **경로 바꾸기** — 원래 서버 경로(`/home/bising0922/Capstone`, 파이썬 환경 경로)가 박혀 있다. 내 환경에 맞게 한 번 바꾼다.
   ```bash
   cd runs/tmp
   sed -i "s#/home/bising0922/Capstone#$(cd ../..; pwd)#g; s#/home/bising0922/micromamba/envs/aienv/bin/#$(dirname $(which python))/#g" \
       *.py */*.py */*.sh
   ```
4. 환경: `aienv` (rdkit, torch, torch_geometric, scikit-learn, pandas, matplotlib, pyyaml). PDF 변환에 LibreOffice(`soffice`)가 필요하다.

## 1. 실험별 위치와 순서

| 실험 | 노트북 옵션 / 스크립트 | 결과 |
|---|---|---|
| 누수 실험 (③ 우리 방식, 대조: 무작위로 빼기) | 노트북 `LEAK_MODE` = `'drop_leak'` / `'drop_random'`. `make_leak.py`로 분할 노트북 생성 → `leak/queue.sh`로 학습 → `leak/report_final.py` | 가중치 `_leakdrop` / `_rnddrop` |
| 팀원 방식 (②) | git main의 `ac_pipeline.ipynb` (분자 분할을 타깃 전체에서 하나로, 쌍 캐시 `_v3`) | 가중치 `_v3` |
| ECFP + MLP (MoleculeACE 설정) | `ecfp/ecfp_mlp.py <target|shared> <pair|mol> <none|drop_leak|v3> <시드>`, `ecfp/launch.sh`. MoleculeACE 타깃별 MLP 설정은 `mace/` | `ecfp/pred_*.pkl` |
| 예측 헤드 크기 (④) | 노트북 `HEAD = 'big'` (1024 → 1024 → 1, 드롭아웃 0). `head/make_head.py` → `head/launch.sh` → `head/report_head.py` | 가중치 `_head1024`, `head/pred_head_*.pkl` |
| 편집 정보 빼 보기 | 노트북 `USE_EDIT = False` / `DROP_NCHANGED = True`. `make_ablC.py` | 가중치 `_noedit` / `_nonchanged` |
| XAI (Integrated Gradients) | `xai/make_xai.py` → `xai/xai.ipynb` 실행 → `xai/xai_track.py` | `xai/xai_*.pkl` |
| **실험 계획과 결정 기록** | `계획_고정_2026-10-18.md` — valid로만 고르고 test는 한 번만 연다는 규칙, 모든 결정과 그 시점 | – |
| 그래프 모델 타깃별 학습 (기각: valid에서 네 모델 모두 30개 타깃 한 모델보다 나쁨) | `pertarget/make_pertarget.py` → `run_queue.py` → `collect.py val` | `runs/pertarget/<타깃>/` |
| 최종 설계 1 (합 읽기·헤드 2048·조각 인코더 없음·한 군데만 다른 valid 쌍으로 멈춤) | `final/make_final.py`, `final/final_eval.py` | 가중치 `_h300_…_sum_valpass` |
| 원인 찾기 (설계 1에서 하나씩 되돌림, Mole-BERT 쌍 시드 3) | `revert/make_revert.py`, `revert/compare.py` | `revert/pred_r_*.pkl` |
| 최종 설계 2 + 과적합 대책 후보 (드롭아웃·AdamW·인코더 학습률) | `final2/auto.sh` (decide.py → make_final2.py → make_reg.py → run_queue.py), `final2/reg_compare.py` | `final2/pred_*.pkl` |
| **최종 설계 3 = 예전 설계 + GINE 폭 300 + AdamW 0.01** (노트북 기본값) | `final3/make_final3.py`, `final3/adamw_s0.py` | 가중치 `_h300_dr0.3_wd0.01_…_head1024_adamw` |
| 검증 후보 비교 (중단, 참고용) | `screen/make_screen.py`, `screen/screen_val.py` | – |
| 학습 곡선 (시드별 train vs valid, cliff·non-cliff) | `curves/plot_curves.py`, `curves/plot_curves_split.py` | – |
| ECFP + Tanimoto 커널 SVR 분자별·쌍 (타깃별) | `svr/svr_delta.py mol` (C는 검증 분자 활성 오차로) / `svr/svr_delta.py pair <타깃,...> 1,10,100,1000 <이름>` (C는 검증 쌍 Δ 오차로, 쌍 커널 T(A,C)·T(B,D), 두 방향 학습, 반대칭 평균). 표·검정은 `svr/table.py` | `svr/pred_svr_*.pkl` |

## 2. 표·PDF 다시 만들기 (캐시만 있으면 학습 없이)

```bash
python runs/tmp/ecfp/report3.py 1 2 3      # 세 설정 비교표 → runs/tmp/ecfp/report3.pkl
python runs/tmp/ecfp/build_pdf.py          # 누수_비교_보고서.pdf
python runs/tmp/ecfp/build_report.py       # 결과_보고_헤드수정.pdf (④ 큰 헤드 포함)
```
`report3.py`의 설정 번호: 1 = 누수 있음, 2 = 우리 방식, 3 = 팀원 방식 (PDF 표기 ①·③·② 순서와 다르니 주의).

## 3. 공용 서버에서 돌릴 때

- 분할 학습 스크립트는 CPU 코어를 `taskset -c 28-47`로, 우선순위를 `nice -n 10`으로 제한한다. 서버에 맞게 고친다
- 노트북 실행 시 `TMPDIR`, `MPLCONFIGDIR`를 홈 쪽 폴더로 지정한다 (루트 디스크가 차 있을 때 실패 방지)
