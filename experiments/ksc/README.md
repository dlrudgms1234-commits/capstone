# KSC2026 추가 실험 (서버 실행분)

`paper/KSC2026_서버이관_2026-10-16.zip` 안의 `서버_인계.md` 를 서버에서 실행한 기록이다.
맥에서 만든 `paper/eval_pairs_predictions_leakfree.csv` 와 같은 평가 쌍 38,711개(cliff 8,778개,
타깃 30개)를 쓴다.

실행 환경: GPU 서버(`capstone-ml` 환경), 2026-10-17 ~ 10-18.

## 0. 평가 쌍 복원과 대조 (`build_eval.py`)

서버에는 위 CSV 가 없어서 캐시(`runs/tmp/head`, `runs/tmp/ecfp`, `runs/preds_*`)에서 같은 표를
다시 만들었다. 그 뒤 이 브랜치의 CSV 를 받아 **행 38,711개 전부를 대조했고 완전히 일치**했다
(delta·예측값 모두 최대 차 5e-6 = CSV 소수 5자리 반올림 폭, `is_cliff` 불일치 0).
논문 표 1 의 다섯 행도 소수 셋째 자리까지 그대로 재현된다.

## 1. 실험 B: 쌍 입력 ECFP + SVR (`svr_pair.py`) — 완료

인계 문서 2절 그대로다. ECFP4 2,048비트, 쌍 커널 K((A,B),(C,D)) = T(A,C)·T(B,D),
예측 Δ̂ = [g(A,B) − g(B,A)]/2, C 는 {1,10,100,1000} 중 검증 쌍 RMSE 최소, ε = 0.1, 타깃별 학습.
같은 스크립트가 분자별 SVR 도 다시 계산하며, 그 값이 논문 표 1 의 0.914 / 0.455 / 0.591 과 같다
(게이트).

| 입력 | 모델 | cliff | non-cliff | 전체 |
|---|---|---|---|---|
| 분자별 | ECFP + SVR | **0.914** | 0.455 | **0.591** |
| 쌍 | ECFP + SVR | 0.994 | **0.428** | 0.605 |

- 쌍 − 분자별, cliff: **+0.080** (95% CI [0.067, 0.093], 쌍이 나은 타깃 3/30, Wilcoxon p = 4.4e-05)
- 쌍 − 분자별, 전체: +0.014 (CI [0.007, 0.021], 6/30, p = 0.0012)
- Mole-BERT 쌍 − 쌍 SVR, cliff: −0.105 (CI [−0.127, −0.083], 28/30, p = 1.3e-07)

즉 **SVR 에서는 쌍 입력이 cliff 를 유의하게 나쁘게 한다.** non-cliff 는 반대로 좋아진다.
인계 문서 4절의 사전 해석대로, 쌍 입력의 이득은 그래프 신경망에 한정해서 쓰는 것이 맞다.

주석: 학습 쌍 상한 20,000(시드 0 무작위 추출)은 `CHEMBL234_Ki`(25,940쌍)와
`CHEMBL244_Ki`(20,159쌍) 두 타깃에만 적용됐다. 나머지 28개는 전부 썼다.
차이 커널(선택 대안)은 돌리지 않았다.

## 2. 실험 A: 변화 부위 인코더 뺀 쌍 입력 모델 (`make_nofrag.py`) — 실행 중

`USE_FRAG, USE_EDIT = False, True`, `LEAK_MODE = 'drop_leak'`, `HEAD = 'big'`, 타깃 30개,
시드 0·1·2 를 GPU 하나씩 맡는다. 한 노트북이 GINE → Mole-BERT 순으로 학습한다.
결과는 `out_nb/pred_{gine,mb}_nofrag_s*.pkl` 로 떨어지고 `ksc_stats.py` 가 읽는다.

노트북에서 뺀 셀: `concat-example`(그림. `_m.frag` 를 호출해 이 조건에서는 죽는다),
`gnnmol-md`·`gnnmol-train`·`5acb601f`(분자별 GNN 비교군. 이 ablation 에 필요 없다).

## 3. 통계 (`ksc_stats.py`)

논문 `stats/ksc_stats.py` 와 같은 양식이다 — cliff / non-cliff / 전체 RMSE(시드 평균),
비교는 차이 A − B 와 **평가 분자 단위 군집 부트스트랩 2,000회** 95% CI, 타깃 30개 Wilcoxon.
평가 분자는 쌍 (i, j) 중 split 이 test 인 쪽이다. 결과는 `ksc_results.json`.

## 재현

```bash
python build_eval.py                      # eval_pairs.pkl (+ 표 1 재현 확인)
for t in $(cat targets.txt); do python svr_pair.py $t; done
python make_nofrag.py && bash out_nb/launch.sh
python ksc_stats.py                       # → ksc_results.json
```

큰 산출물(`*.pkl`, `out/`, `out_nb/`, `runs/`)은 올리지 않는다 — 위 명령으로 다시 만들어진다.
