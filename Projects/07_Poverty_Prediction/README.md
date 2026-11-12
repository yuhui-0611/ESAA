# Poverty Prediction | 소비 예측과 다중 빈곤선 평가를 결합한 실험 설계

> **핵심:** 소비액 예측을 조사별 빈곤율 추정으로 변환. 19개 빈곤선 복합 평가함수, 그룹 단위 검증, 앙상블 가중치·임계값 보정 실험.

## 문제 정의
가구 소비 조사는 비용이 높지만 빈곤 추정에는 중요한 근거임. 가구 특성으로 **소비(`cons_ppp17`)를 예측**하고 이를 이용해 빈곤선별 빈곤율(`pct_hh_below_*`)을 재구성하는 접근을 실험함.

## 데이터 처리 및 모델
- 가구 수준 특성과 조사(`survey_id`) 식별자를 활용함.
- 결측값 대체, 범주형 처리, 파생 변수·로그 변환 등 여러 전략을 버전별로 비교함.
- **CatBoost / LightGBM / XGBoost Regressor**로 소비액을 예측하고 로그 스케일 학습 및 역변환(`log1p` / `expm1`)을 실험함.
- Baseline부터 Ver1~Ver4, 추가 전처리까지 노트북에 시행착오를 기록함.

## 심화 실험 1 — 소비 예측 → 빈곤율 집계
19개 기준 빈곤선(노트북 기준 **3.17~27.37**) 각각에 대해, 조사별 소비 예측이 임계값 미만인 가구의 비율을 산출함.

```python
# 특정 survey_id에 속한 가구의 소비 예측 cons: (n,)
# poverty_thresholds: (19,)
poverty_rates = (cons[:, None] < poverty_thresholds[None, :]).mean(axis=0)
```

이렇게 **개별 가구 예측을 집계 지표로 변환**해 단일 소비 회귀 RMSE만으로는 드러나지 않는 빈곤율 추정 오차를 평가함.

## 심화 실험 2 — 복합 평가 지표 및 앙상블 최적화

```text
poverty_error     = 조사별 · 빈곤선별 빈곤율 상대오차 평균
consumption_error = 가구별 소비 예측 상대오차 평균
final_error       = 0.9 × poverty_error + 0.1 × consumption_error
```

- **그룹 기반 검증:** `survey_id` 기준 GroupShuffleSplit(`test_size=0.2`, `random_state=42`) 적용. 동일 조사 그룹의 학습·검증 중복 방지.
- **회귀 모델 비교:** LightGBM·XGBoost·CatBoost 단일 모델과 2·3모델 가중 평균 앙상블 실험.
- **가중치 탐색:** 2모델 조합의 가중치를 0.05 간격으로 변경하며 복합 오차 비교. 모델 버전별 전처리·분할 조건 분리.

### 저장된 동일 버전 내부 검증 예시

| `03_Model_Ver1_Plus.ipynb` 실험 | 복합 오차 |
|---|---:|
| XGBoost 단독 | **0.073779** |
| XGBoost 90% + CatBoost 10% | **0.073279** |

- **앙상블 실험 결과:** 동일 검증 분할에서 XGBoost 단독 0.073779 → XGBoost 90% + CatBoost 10% 결합 0.073279 기록.
- **평가 기준:** `03_Model_Ver1_Plus.ipynb`의 내부 검증 데이터에서 가중치별 복합 오차 계산.
- **해석:** 개인별 소비액 오차만 낮추는 전략과 조사별 빈곤율 오차를 함께 낮추는 전략의 차이 확인.

## 심화 실험 3 — 빈곤선 주변 판정 보정

- **대표 기준:** 빈곤선 `7.06`에 대한 ±10% 인접 구간 식별.
- **보정 모델:** `RandomForestClassifier(class_weight='balanced')`의 빈곤 확률 사용.
- **후처리:** 회귀 예측값·분류 확률·임계값 규칙을 결합하고 복합 오차 재평가.
- **시나리오 실험:** 빈곤선 주변 구간과 가중치·판정 규칙을 바꾸며 빈곤율 집계 오차 변화 확인.

## 재현 가능한 실험 기록

| Notebook | 내용 |
|---|---|
| [`01_Baseline.ipynb`](./01_Baseline.ipynb) | 데이터 구조, 결측값 처리, baseline |
| [`02_Model_Ver1.ipynb`](./02_Model_Ver1.ipynb) | 3종 회귀기 초기 비교, 빈곤율 평가 |
| [`03_Model_Ver1_Plus.ipynb`](./03_Model_Ver1_Plus.ipynb) | 확장 모델링 및 앙상블 |
| [`04_Model_Ver2.ipynb`](./04_Model_Ver2.ipynb) | 전처리·설정 변경 실험 |
| [`05_Model_Ver3.ipynb`](./05_Model_Ver3.ipynb) | 로그 타깃, 복합 평가, 앙상블·보정 |
| [`06_Model_Ver4.ipynb`](./06_Model_Ver4.ipynb) | 복합 목적함수, 가중치 탐색, Random Forest 보정 |
| [`07_Additional_Preprocessing.ipynb`](./07_Additional_Preprocessing.ipynb) | 결측·특징 구성 개선 |

## 분석 결론

- **평가 목적의 차이:** 소비액 예측값이 유사해도 빈곤선 주변에서 이진 판정이 바뀌면 조사별 빈곤율 오차가 달라지는 구조 확인.
- **모델 선택:** 단순 소비액 회귀 성능뿐 아니라 조사 수준의 추정 오차를 함께 반영한 가중치 비교.
- **분석 결과:** 가구별 회귀 예측값을 조사 단위 지표로 재집계하고, 빈곤선 인근 예측값의 작은 변화가 빈곤율 오차에 미치는 영향 실험.

**Stack** `Python` `Pandas` `NumPy` `Scikit-learn` `LightGBM` `XGBoost` `CatBoost` `Feature Engineering`
