# Movie Box Office Prediction | 가설 검증부터 회귀 앙상블까지

> **프로젝트 핵심:** 영화 메타데이터의 설명력을 가설과 시각화로 확인하고, **선형 회귀·규제 회귀·트리 앙상블을 교차검증**한 뒤 관객 수 예측을 개선했습니다.

## 문제 정의 및 데이터
영화의 장르·감독 과거 흥행·배우/스태프 수·상영시간·개봉시기 등의 특성을 활용해 관객 수(`box_off_num`)를 예측했습니다. 큰 흥행작이 분포를 왜곡할 수 있어 로그 변환과 이상치 처리의 영향을 검토했습니다.

## 분석 과정
1. **EDA 및 가설 설정:** 감독의 이전 작품 관객 수, 장르, 개봉 시기 등이 흥행과 연관되는지 분포·산점도·회귀선 비교
2. **특징 엔지니어링:** 개봉일 연·월·일 분리, 감독 과거 실적 결측 처리, 수치/범주형 변수 인코딩, 상관성이 높은 변수 점검
3. **규제 회귀:** Ridge·Lasso·ElasticNet의 정규화 계수 비교, 타깃 `log1p` 변환 실험
4. **트리 모델:** Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM 비교
5. **평가 및 탐색:** 5-Fold CV RMSE와 GridSearchCV 활용
6. **최종 예측:** 복수 모델의 예측값을 결합하고 테스트셋 예측값의 범위를 점검

## 주요 기술적 판단
- **왜 로그 변환인가?** 극단적인 고관객 수 사례가 학습을 지배하는 현상을 완화하고 분포의 비대칭성을 줄이기 위해서입니다.
- **왜 선형과 트리를 모두 비교했는가?** 규제 회귀의 단순·안정적 기준선과 트리 모델의 비선형 상호작용 학습 능력을 비교하기 위해서입니다.
- **왜 5-Fold 교차검증인가?** 단일 데이터 분할에 따른 평가 편차를 완화하기 위해서입니다.

## Files
| Notebook | 내용 |
|---|---|
| [`01_EDA.ipynb`](./01_EDA.ipynb) | 데이터 특성과 결측값 탐색 |
| [`02_Visualization.ipynb`](./02_Visualization.ipynb) | 흥행 관련 가설, 시각화 |
| [`03_Linear_Regression.ipynb`](./03_Linear_Regression.ipynb) | 규제 회귀와 성능 검증 |
| [`04_Regression_Tree.ipynb`](./04_Regression_Tree.ipynb) | 트리/부스팅/앙상블 비교 |
| [`05_Final_Model.ipynb`](./05_Final_Model.ipynb) | 최종 예측 및 결합 |

> 이 프로젝트는 **회귀 및 머신러닝 앙상블** 사례입니다.

**Stack** `Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `GridSearchCV` `Matplotlib`

