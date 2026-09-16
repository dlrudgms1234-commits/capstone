# 쌍 기반 ΔpKi 예측 파이프라인 (`ac_pipeline.ipynb`)

한 곳만 다른 분자쌍 (A, B)를 입력받아 **Δ = p(B) − p(A)** 를 직접 예측한다.
3D CNN 인코더가 준비되기 전까지 인코더 자리는 GNN으로 채워두고, 그 앞뒤(입력 · 손실 · 평가 · 설명)를 먼저 완성하는 것이 목표다.

## 노트북 구성
| 섹션 | 내용 |
|---|---|
| 1–2 | MoleculeACE 데이터 내려받기, 분자 단위 train/val/test split |
| 3–4 | "한 곳만 다른 쌍" 판정(MCS 기반)과 쌍 생성 |
| 5–7 | 쌍 통계, cliff/non-cliff 시각화, easy/hard 시각화 |
| 8 | Baseline: ECFP + SVR (분자별 예측 후 차이) |
| 9–10 | 그래프 입력, GNN 쌍 모델, 가중 손실 학습 |
| 11–12 | α 선택, baseline 비교, 결과 해석 |

## AC-aware 적용 지점
| 아이디어 | 구현 |
|---|---|
| ① 표현 | 바뀐 원자(MCS 밖)를 별도 인코더로, 편집 종류(replace/insert/delete) one-hot |
| ② 손실 | `w = 1 + α·|Δ|`, 배치별 `Σw` 정규화. α는 validation cliff RMSE로 선택 |
| ③ 설명 | 원자별 기여도 → 편집 부위 비율 (미구현) |
| ④ 평가 | {train, easy, hard} × {cliff, non-cliff} RMSE |

## 실행
```bash
CUDA_VISIBLE_DEVICES=4 jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 ac_pipeline.ipynb
```
- 환경: `aienv` (rdkit, torch, torch_geometric, scikit-learn)
- `data/`와 `runs/`는 커밋하지 않는다. 노트북이 데이터를 내려받고 쌍을 다시 만든다
- 쌍 생성은 MCS 계산 때문에 개발용 타깃 4개 기준 약 10분 걸린다. 결과는 `data/pairs/`에 캐시된다
- 학습 가중치는 `runs/`에 저장되고, 있으면 다시 학습하지 않는다

## 현재 상태 (2026-09-16, 타깃 4개, 시드 1개)
test RMSE 기준으로 **쌍 모델이 아직 baseline을 이기지 못했다.**

| test split | | zero | SVR | GNN α=4 |
|---|---|---|---|---|
| easy | cliff | 1.825 | **0.926** | 0.940 |
| | non-cliff | **0.523** | 0.532 | 0.773 |
| hard | cliff | 1.856 | **1.269** | 1.293 |
| | non-cliff | **0.540** | 0.620 | 0.892 |

train cliff RMSE 0.187 vs val 0.944로 과적합이 크다. 자세한 해석은 노트북 12번 섹션에 있다.

다음 단계: 양방향 평균 예측 → 정규화 강화 → 나머지 26개 타깃으로 데이터 확대
