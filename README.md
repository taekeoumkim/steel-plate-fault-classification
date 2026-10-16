# Steel Plate Fault Classification

강판 표면 결함의 측정값 27개로 결함 7종을 분류하는 머신러닝 프로젝트입니다.
팀 프로젝트(5인)에서 모델링을 맡아 진행한 작업을 개인 저장소로 다시 정리했습니다.

| 항목 | 내용 |
|---|---|
| 데이터 | [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) · 1,941건, 결함 7종 |
| 문제 | 다중 분류 (가장 많은 유형과 가장 적은 유형의 건수 차이 약 12배) |
| 평가 지표 | Macro F1 (모든 유형을 같은 비중으로 평가) |
| 검증 | 80:20 층화 분할, Train 내부 5-fold CV + 반복 CV(5×3) / Test는 최종 1회 평가 |
| 최종 모델 | ExtraTrees + RandomForest + GBM **Soft Voting** · 피처 36개(원본 26 + 파생 10) · 클래스 가중치 |

## 결과

| 모델 | 5-fold CV | 반복 CV (5×3) | Test |
|---|---:|---:|---:|
| Baseline (XGBoost 기본값) | 0.8109 | 0.8143 | 0.8165 |
| **최종 모델 (Soft Voting)** | **0.8441** | **0.8389** | **0.8324** |

- Test Macro F1 95% 신뢰구간: 0.7904 ~ 0.8686 (부트스트랩), Accuracy 0.8098
- Test 유형별 F1: K_Scratch 0.9536 · Z_Scratch 0.9730 · Dirtiness 0.9524 · Stains 0.8889 · Other_Faults 0.7714 · Bumps 0.7421 · Pastry 0.5455

<img src="figures/confusion_matrix.png" width="420">

## 접근 과정

| 단계 | 내용 | 5-fold Macro F1 |
|---|---|---:|
| Baseline | 6개 모델 비교 → 부스팅 계열 우세 (GBM 기본값) | 0.7958 |
| 불균형 처리 | 클래스 가중치·SMOTENC·언더샘플링 비교 → GBM은 클래스 가중치 채택 | 0.8095 |
| 파생변수 + 튜닝 | 형상·밝기·위치·강판 맥락 파생변수 10개, RandomizedSearchCV로 다시 튜닝 | 0.8401 |
| 앙상블 | 5개 튜닝 모델의 Soft Voting 조합 비교 → 배깅 2 + 부스팅 1 | **0.8441** |

- **누수 방지:** 전처리·샘플링·가중치·집계형 파생변수는 모두 Pipeline 안에 넣어 학습 fold로만 계산했습니다.
- **모델 선택:** 5-fold 최고값은 여러 설정 중 1등을 고른 값이라 낙관적이므로, 후보 간 비교는 반복 CV(15 fold) 평균으로 했습니다.
- **불균형 처리:** 언더샘플링은 학습 데이터가 1/5로 줄어 크게 나빠졌습니다. 오버샘플링·가중치는 소수 유형 Recall을 올리는 대신 Other_Faults를 더 놓치는, 판정 경계를 옮기는 효과였습니다.
- **파생변수:** 파라미터를 고정한 채 변수만 추가하면 효과가 거의 없었고, 다시 튜닝했을 때 개선됐습니다.

## 검증과 해석

**과적합 점검**

<img src="figures/learning_curve.png" width="640">

- 학습 데이터를 20%에서 100%로 늘리면 검증 F1은 0.7430에서 0.8441로 오르고, 검증 Log Loss는 0.6907에서 0.4827로 내려갑니다.
- GBM 단독은 후반에 확률을 과신해 손실이 다시 오릅니다. Soft Voting은 배깅 모델이 이를 상쇄해 끝까지 손실이 감소합니다.

**오류 구조**

- Test 오분류 74건 중 32건이 Bumps ↔ Other_Faults, 21건이 Pastry ↔ Other_Faults입니다.
- 오류가 서로 닮은 세 유형의 경계에 집중되어 있습니다. 건수가 가장 적은 Stains·Dirtiness는 오히려 잘 맞힙니다.

**변수 중요도 (Permutation)**

- 상위 3개가 강판 두께·폭·강종(생산 조건)입니다.
- 결함의 생김새보다 "어느 강판에서 나온 결함인가"에 크게 의존합니다.

**일반화 점검: 처음 보는 생산 조건**

- 같은 생산 조건(폭·두께·강종)의 결함이 Train과 Test에 섞여 있어, Test 결함의 93.8%가 Train에 같은 조건이 있습니다.
- 같은 조건이 학습·검증 한쪽에만 들어가도록 나누는 그룹 교차검증(StratifiedGroupKFold)으로 새 조건에서의 성능을 따로 측정했습니다.
- 생산 조건별 결함 수를 세는 `plate_defect_count`의 효과도 이 변수를 뺀 구성을 다시 튜닝해 비교했습니다.

| 모델 | 무작위 분할 (5×3) | 그룹 분할 (5×3) |
|---|---:|---:|
| Baseline XGBoost | 0.8143 | (05 실행 후 기입) |
| 최종 모델 | 0.8389 | (05 실행 후 기입) |
| 최종 모델 − `plate_defect_count` (재튜닝) | (05 실행 후 기입) | (05 실행 후 기입) |

## 구조

```
├── data/Faults.csv                 # UCI 원본
├── src/steel.py                    # 데이터 로드·전처리·파생변수·모델 래퍼·최종 모델
├── notebooks/
│   ├── 01_data_eda.ipynb                  # 품질 점검, 클래스 분포, 생산 조건 구조
│   ├── 02_baseline_imbalance.ipynb        # Baseline 6개 모델, 불균형 처리 비교
│   ├── 03_features_tuning_ensemble.ipynb  # 파생변수, 튜닝, Soft Voting 조합, 반복 CV
│   ├── 04_final_model.ipynb               # 최종 모델 CV·과적합 점검·Test 평가·저장
│   └── 05_generalization.ipynb            # plate_defect_count 효과, 그룹 교차검증
├── figures/                        # 결과 그림
└── results/                        # 최종 파라미터, 계산 캐시
```

## 실행

```bash
pip install -r requirements.txt
```

노트북은 01부터 순서대로 실행합니다. 04는 단독으로 실행할 수 있습니다(최종 파라미터는 `src/steel.py`에 고정).
튜닝·CV 결과는 `results/cache/`에 저장되어 두 번째 실행부터는 빠르게 끝납니다.

## 배운 점

- **점수는 재는 방법과 함께 읽어야 합니다.** 한 번의 5-fold에서 보인 차이가 반복 CV에서는 사라지기도 했고, 무작위 분할과 그룹 분할의 차이는 모델 간 차이보다 컸습니다.
- **"적은 유형"과 "어려운 유형"은 다릅니다.** 성능을 막는 것은 건수가 적은 유형이 아니라 서로 닮은 유형의 경계였습니다.
- **변수를 바꾸면 하이퍼파라미터도 다시 맞춰야 합니다.** 파생변수의 효과는 재튜닝 후에 나타났습니다.
