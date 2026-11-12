# Movie Box Office Prediction | 가설 검증 및 회귀 앙상블

> **핵심:** 개봉 전 영화 메타데이터 기반 관객 수 예측. 흥행 요인 가설 검토, 회귀 모델별 5-Fold CV, XGBoost·Ridge 예측 결합 실험

## 문제 정의 및 데이터
- 영화의 장르·감독 과거 흥행·배우/스태프 수·상영시간·개봉시기 등의 특성을 활용해 **관객 수(`box_off_num`)**를 예측
- 큰 흥행작이 분포를 왜곡할 수 있어 로그 변환과 이상치 처리의 영향을 검토

## 분석 과정
1. **EDA 및 가설 설정:** 감독의 이전 작품 관객 수, 장르, 개봉 시기 등이 흥행과 연관되는지 분포·산점도·회귀선 비교
2. **특징 엔지니어링:** 개봉일 연·월·일 분리, 감독 과거 실적 결측 처리, 수치/범주형 변수 인코딩, 상관성이 높은 변수 점검
3. **규제 회귀:** Ridge·Lasso·ElasticNet의 정규화 계수 비교, 타깃 `log1p` 변환 실험
4. **트리 모델:** Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM 비교
5. **평가 및 탐색:** 5-Fold CV RMSE와 GridSearchCV 활용
6. **최종 예측:** 복수 모델의 예측값을 결합하고 테스트셋 예측값의 범위를 점검

## 주요 실험 결과

| 실험 | 기록 지표 | 평가 조건 |
|---|---:|---|
| Ridge (`alpha=10`) | RMSE **2.136** | 규제 회귀 5-Fold CV 평균 |
| Random Forest (튜닝) | RMSE **1.9213** | GridSearchCV 5-Fold CV 최적 점수 |
| XGBoost (튜닝) | RMSE **1.8652** | 추가 파라미터 탐색의 5-Fold CV 최적 점수 |
| XGBoost + Ridge (가중 결합) | RMSE **2.0755** | 별도 검증 분할의 실험 결과 |

- **모델 비교:** Ridge·Random Forest·XGBoost를 교차검증으로 평가하고, 파라미터 탐색에서 XGBoost의 RMSE 1.8652 기록. 로그 변환과 비선형 모델 적용에 따른 오차 특성 분석
- **지표 단위:** `log1p(box_off_num)`를 타깃으로 사용한 교차검증의 RMSE. 원래 관객 수 단위의 절대 오차와 구분
- **특성 설계:** 감독의 기존 흥행 실적, 개봉 시점, 장르, 상영시간, 스태프 구성 등 정형 특성의 예측력 비교

## Files
| Notebook | 내용 |
|---|---|
| [`01_EDA.ipynb`](./01_EDA.ipynb) | 데이터 특성과 결측값 탐색 |
| [`02_Visualization.ipynb`](./02_Visualization.ipynb) | 흥행 관련 가설, 시각화 |
| [`03_Linear_Regression.ipynb`](./03_Linear_Regression.ipynb) | 규제 회귀와 성능 검증 |
| [`04_Regression_Tree.ipynb`](./04_Regression_Tree.ipynb) | 트리/부스팅/앙상블 비교 |
| [`05_Final_Model.ipynb`](./05_Final_Model.ipynb) | 최종 예측 및 결합 |

**Stack** `Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `GridSearchCV` `Matplotlib`
