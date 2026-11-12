# ESAA | Data Analytics · Machine Learning · Deep Learning

**데이터 분석 프로젝트 7개 및 머신러닝·딥러닝 학습 기록**

정형 데이터 예측, 자연어 처리, 모델 해석, 임계값·앙상블 실험을 수행한 ESAA 활동 아카이브. 프로젝트별로 데이터 전처리부터 모델 학습·평가까지의 구현 코드와 분석 결과 정리.

## 주요 프로젝트

| 프로젝트 | 문제 정의 | 핵심 구현 |
|---|---|---|
| **[Olist E-commerce](./Projects/06_Olist_Ecommerce_Analysis)** | 브라질 이커머스 고객·판매자 행동 분석 및 셀러 이탈 예측 | **사전학습 XLM-RoBERTa 리뷰 감성 추론, PyTorch MLP 직접 구현, 트리 모델 비교, SHAP** |
| **[Poverty Prediction](./Projects/07_Poverty_Prediction)** | 가구 소비액 예측 및 조사별 빈곤율 산출 | **19개 빈곤선 평가, 그룹 단위 검증 분리, 복합 오차, 앙상블 가중치·임계값 실험** |
| **[Real Estate Investment](./Projects/05_Real_Estate_Investment)** | 다년도 가구 조사 기반 부동산 투자 의향 분류 | **2019–2025년 변수 체계 통합, XGBoost, SHAP 분석** |

## 전체 프로젝트

| 프로젝트 | 주요 기술 및 실험 |
|---|---|
| [01. Movie Box Office Prediction](./Projects/01_Movie_BoxOffice_Prediction) | 회귀 모델 비교, 로그 변환, 5-Fold CV, 트리 앙상블 |
| [02. Hotel Reservation Cancellation](./Projects/02_Hotel_Reservation_Cancellation) | 분류 모델 비교, 특성 선택, Optuna 튜닝, Voting |
| [03. Author Classification NLP](./Projects/03_Author_Classification_NLP) | TF-IDF·N-gram, GloVe 기반 텍스트 특징, OOF 확률, 앙상블 |
| [04. Bank Marketing Hackathon](./Projects/04_Bank_Marketing_Hackathon) | 불균형 분류, 클래스 가중치, F1 임계값 탐색, 확률 앙상블 |
| [05. Real Estate Investment](./Projects/05_Real_Estate_Investment) | 다년도 자료 통합, XGBoost, Feature Importance·SHAP |
| [06. Olist E-commerce Analysis](./Projects/06_Olist_Ecommerce_Analysis) | Transformer 감성 추론, PyTorch MLP, 고객·셀러 행동 분석 |
| [07. Poverty Prediction Challenge](./Projects/07_Poverty_Prediction) | 소비 회귀, 조사별 빈곤율, 가중 앙상블·후처리 실험 |

## Deep Learning & Advanced Study

프로젝트 코드와 별도로 `Study/OB`에 PyTorch 학습, CNN, RNN/LSTM, 시계열, 생성 모델 관련 실습 기록 정리

- [PyTorch](./Study/OB/04_PyTorch) · [Deep Learning](./Study/OB/05_Deep_Learning) · [RNN](./Study/OB/06_RNN)
- [Time Series](./Study/OB/07_Time_Series) · [Generative Models](./Study/OB/08_Generative_Models)
- [Study/OB 전체](./Study/OB) · [Study/YB 기초 머신러닝](./Study/YB)

## Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `PyTorch` · `Hugging Face Transformers` · `SHAP` · `Optuna`

