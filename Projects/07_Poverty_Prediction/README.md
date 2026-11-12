# Poverty Prediction Challenge | 가구 소비·빈곤율 예측

**가구 소비액 회귀 예측값을 19개 빈곤선에 적용하고, 조사별 빈곤율 오차를 포함한 복합 지표로 모델·앙상블·후처리 전략 비교**

## 문제 정의
- **예측 대상:** 가구 소비액 `cons_ppp17` 및 조사(`survey_id`)별 빈곤율
- **평가 관점:** 소비액 회귀 오차뿐 아니라 빈곤선 이하 가구 비율의 집계 오차 동시 고려
- **주요 기술:** XGBoost, CatBoost, LightGBM, GroupShuffleSplit, Custom Metric, Ensemble Search

## 주요 구현

### 1. 조사 단위 분할·소비액 예측
- `survey_id` 기준 **GroupShuffleSplit(80:20)** 적용으로 동일 조사 그룹의 학습·검증 중복 방지
- 가구 특성·범주형 변수 전처리 및 XGBoost·LightGBM·CatBoost 회귀 모델 비교
- 소비액의 `log1p` 변환 학습 및 `expm1` 역변환 실험

### 2. 빈곤율 계산 및 평가 지표
- 예측 소비액을 **19개 빈곤선**에 대입해 조사별 빈곤선 미만 가구 비율 산출
- **빈곤율 상대오차 90% + 소비액 상대오차 10%** 복합 지표 구성 및 실험별 평가
- 가구 단위 소비액 정확도와 조사 단위 빈곤율 정확도의 차이 분석

### 3. 앙상블·임계값 보정 실험
- LightGBM·XGBoost·CatBoost 단독 예측과 2모델 가중 평균 비교
- 가중치를 **0.05 간격**으로 탐색하고 복합 오차 기준으로 조합 선택
- 특정 빈곤선 인접 구간에서 Random Forest 분류 확률을 적용한 예측 보정 시도
- 모델 버전별 전처리·타깃 변환·예측 결합 방식 비교

## 분석 결과
- 소비액 예측값을 여러 빈곤선의 정책 지표로 변환하는 평가 로직 구현
- 단독 모델·가중 앙상블·후처리 규칙의 복합 지표 차이 확인
- **검증 범위:** 가중치 선택에 사용한 내부 검증 데이터에서 기록한 최저 복합 오차는 독립 테스트 점수가 아니므로 대표 성능에서 제외

## 코드
| Notebook | 내용 |
|---|---|
| [01_Baseline](./01_Baseline.ipynb) | 기본 전처리 및 기준 모델 |
| [02_Model_Ver1](./02_Model_Ver1.ipynb) | 회귀 모델·빈곤율 평가 |
| [03_Model_Ver1_Plus](./03_Model_Ver1_Plus.ipynb) | 로그 변환·가중 앙상블 |
| [04_Model_Ver2](./04_Model_Ver2.ipynb) | 모델 설정 비교 |
| [05_Model_Ver3](./05_Model_Ver3.ipynb) | 복합 지표 및 실험 확장 |
| [06_Model_Ver4](./06_Model_Ver4.ipynb) | 가중치·빈곤선 주변 보정 실험 |
| [07_Additional_Preprocessing](./07_Additional_Preprocessing.ipynb) | 추가 전처리 실험 |
