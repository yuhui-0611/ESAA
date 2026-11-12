# Movie Box Office Prediction | 영화 관객 수 예측

**장르·상영시간·감독 이력·개봉 시점 등 개봉 전 정보를 활용한 관객 수 회귀 모델 비교**

## 문제 정의
- **예측 대상:** 영화별 누적 관객 수 `box_off_num`
- **분석 범위:** 흥행 관련 변수 탐색 → 파생 변수 생성 → 규제 회귀·트리 앙상블 실험
- **주요 기술:** Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM, GridSearchCV

## 주요 구현
- **탐색 및 전처리:** 관객 수 분포·이상치 확인, 장르 및 감독 이력 분석, 개봉일 세분화 및 범주형 인코딩
- **타깃 변환:** 관객 수의 오른쪽 꼬리가 긴 분포를 고려한 `log1p` 변환과 역변환 실험
- **회귀 모델 비교:** Ridge·Lasso·ElasticNet의 규제 강도 비교 및 트리·부스팅 회귀 모델 확장
- **교차검증:** 5-Fold CV와 RMSE를 이용한 모델·하이퍼파라미터 비교
- **예측 결합:** 서로 다른 회귀 모델 예측값의 가중 결합 실험 및 제출용 예측 생성

## 분석 결과
- 선형 모델과 비선형 앙상블 모델 간 오차 차이 확인
- 타깃 변환, 특성 구성, 하이퍼파라미터 및 예측 결합에 따른 결과 비교

## 코드
| Notebook | 내용 |
|---|---|
| [01_EDA](./01_EDA.ipynb) | 원천 데이터 구조 및 결측·이상치 확인 |
| [02_Visualization](./02_Visualization.ipynb) | 흥행 요인 탐색 및 시각화 |
| [03_Linear_Regression](./03_Linear_Regression.ipynb) | 규제 회귀 및 CV 평가 |
| [04_Regression_Tree](./04_Regression_Tree.ipynb) | 트리·부스팅 모델 및 GridSearchCV |
| [05_Final_Model](./05_Final_Model.ipynb) | 최종 예측값 생성 및 모델 결합 |
