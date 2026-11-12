# Poverty Prediction Challenge | 소비 예측 · 빈곤율 시나리오 분석

**가구별 소비액을 예측한 뒤 19개 빈곤선에 적용해 빈곤율을 산출했습니다. 소비 예측 오차와 빈곤율 오차를 함께 평가하고, 앙상블 가중치·임계값 보정 실험을 진행했습니다.**

## 프로젝트 개요

- **목표:** 가구 소비(`cons_ppp17`) 예측 및 조사별 빈곤선 이하 가구 비율 추정
- **핵심 기술:** CatBoost, LightGBM, XGBoost, Custom Evaluation, Ensemble Weight Search, Threshold Analysis

## 주요 구현

### 1. 소비액 회귀 예측
- 가구 특성과 조사 ID를 기준으로 결측값·범주형·파생 변수를 정리했습니다.
- **CatBoost·LightGBM·XGBoost Regressor**로 소비액을 예측했습니다.
- 로그 타깃 학습(`log1p`)과 역변환(`expm1`)을 포함해 전처리·모델 설정을 버전별로 비교했습니다.

### 2. 19개 빈곤선별 빈곤율 계산
- 예측 소비액을 **19개 빈곤선(3.17~27.37)**에 대입했습니다.
- 조사(`survey_id`)별로 임계값 미만인 가구의 비율을 집계해 빈곤율을 산출했습니다.
- 가구 단위 회귀 예측 결과를 **조사 단위 정책 지표**로 변환했습니다.

### 3. 복합 평가 지표 설계
- **빈곤율 상대오차 90% + 소비 예측 상대오차 10%**를 결합한 검증 지표를 사용했습니다.
- 소비액 예측 오차와 집계 빈곤율 오차를 동시에 확인하며 모델을 비교했습니다.

### 4. 앙상블 가중치·임계값 시나리오 실험
- LightGBM·XGBoost·CatBoost의 예측값을 가중 결합하고 **2모델·3모델 가중치**를 탐색했습니다.
- 2모델 가중치 탐색 결과, 기록된 LGB+XGB 실험에서는 **XGBoost 단독 선택(alpha=0.00, 복합 오차 0.077469)**이 최적이었습니다.
- 대표 빈곤선 **7.06의 ±10% 구간**에서 Random Forest의 빈곤 확률을 반영하는 후처리 규칙을 실험했습니다.
- 빈곤선과 결합 가중치, 판정 기준을 바꿔가며 예측 결과의 변화를 비교했습니다.

## 분석 결과

- 소비액 예측 모델의 결과를 **19개 빈곤선별 빈곤율**로 변환해 함께 평가했습니다.
- 복합 지표를 기준으로 앙상블 가중치를 비교했고, **모델을 결합하는 것이 항상 유리하지는 않다**는 결과를 확인했습니다.
- 빈곤선 근처의 예측값에 별도 분류 확률을 적용해 **임계값 민감 구간의 예측 보정**을 실험했습니다.

## 코드

| Notebook | 내용 |
|---|---|
| [01_Baseline.ipynb](./01_Baseline.ipynb) | 초기 전처리 및 기준 모델 |
| [02_Model_Ver1.ipynb](./02_Model_Ver1.ipynb) | 회귀 모델 비교 및 빈곤율 평가 |
| [03_Model_Ver1_Plus.ipynb](./03_Model_Ver1_Plus.ipynb) | 모델 확장 및 앙상블 |
| [04_Model_Ver2.ipynb](./04_Model_Ver2.ipynb) | 전처리·학습 설정 변경 |
| [05_Model_Ver3.ipynb](./05_Model_Ver3.ipynb) | 로그 변환·복합 지표·보정 실험 |
| [06_Model_Ver4.ipynb](./06_Model_Ver4.ipynb) | 가중치 탐색·Random Forest 보정 |
| [07_Additional_Preprocessing.ipynb](./07_Additional_Preprocessing.ipynb) | 추가 결측·특징 처리 |
