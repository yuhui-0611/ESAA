# ESAA | Data Analytics · Machine Learning · Deep Learning

**7개 프로젝트 | 예측 모델링 · 딥러닝 · NLP · 실험 설계 · 설명 가능한 AI(XAI)**

데이터 분석 학회 **ESAA**에서 수행한 프로젝트와 학습 기록. 데이터 전처리·가설 검증부터 머신러닝 모델 비교, **PyTorch 신경망 직접 구현**, **Transformer 기반 리뷰 감성 추론**, **빈곤율 중심 복합 평가**, **SHAP 해석**까지 수행.

## 대표 프로젝트

| 프로젝트 | 핵심 문제 | 주요 구현 및 검증 |
|---|---|---|
| **[Olist E-commerce](./Projects/06_Olist_Ecommerce_Analysis)** | 리뷰·거래 데이터 기반 고객 분석 및 셀러 이탈 분류 | **XLM-RoBERTa 감성 추론, PyTorch MLP(64→32→1), Early Stopping, 5회 시드 실험, XGBoost·LightGBM 비교, SHAP** |
| **[Poverty Prediction](./Projects/07_Poverty_Prediction)** | 가구 소비액과 조사별 빈곤율 동시 평가 | **19개 빈곤선, 빈곤율 90%+소비 오차 10% 복합 평가, GroupShuffleSplit, XGBoost·CatBoost 앙상블, 임계값 보정** |
| **[Real Estate Investment](./Projects/05_Real_Estate_Investment)** | 가구의 부동산 투자 의향 분류 | **2019–2025년 조사 데이터 통합, XGBoost, SHAP; 20% 무작위 홀드아웃 F1 0.6816 / ROC-AUC 0.7443** |

## 전체 프로젝트

| 프로젝트 | 기술적 초점 |
|---|---|
| [01. Movie Box Office Prediction](./Projects/01_Movie_BoxOffice_Prediction) | 흥행 요인 탐색, 타깃 로그 변환, 규제 회귀·XGBoost 비교, 5-Fold CV |
| [02. Hotel Reservation Cancellation](./Projects/02_Hotel_Reservation_Cancellation) | 고객 예약 취소 분류, 특성 선택, Optuna, 트리 앙상블 및 오류 분석 |
| [03. Author Classification NLP](./Projects/03_Author_Classification_NLP) | Word/Character TF-IDF, GloVe, OOF 확률, XGBoost·LightGBM |
| [04. Bank Marketing Hackathon](./Projects/04_Bank_Marketing_Hackathon) | 불균형 분류, 클래스 가중치, F1 임계값 탐색, 확률 앙상블 |
| [05. Real Estate Investment](./Projects/05_Real_Estate_Investment) | 다년도 변수 표준화, XGBoost, Feature Importance·SHAP |
| [06. Olist E-commerce Analysis](./Projects/06_Olist_Ecommerce_Analysis) | Transformer 추론, PyTorch 신경망 학습, 고객·판매자 분석 |
| [07. Poverty Prediction Challenge](./Projects/07_Poverty_Prediction) | 소비 예측→빈곤율 집계, 복합 평가·가중치 탐색, 임계값 민감도 실험 |

## 딥러닝·심화 학습

`Study/OB`에 **PyTorch, CNN, RNN/LSTM, 시계열, 생성 모델, NLP·추천 시스템** 학습 노트북 정리. 실전 프로젝트에서 직접 학습한 PyTorch MLP와 사전학습 Transformer 추론은 프로젝트별 README에 별도 수록.

| 영역 | 코드 |
|---|---|
| PyTorch / 신경망 / CNN | [PyTorch](./Study/OB/04_PyTorch) · [Deep Learning](./Study/OB/05_Deep_Learning) |
| 순환 신경망 / 시계열 | [RNN](./Study/OB/06_RNN) · [Time Series](./Study/OB/07_Time_Series) |
| 생성 모델 | [Generative Models](./Study/OB/08_Generative_Models) |
| NLP / 추천 / 기타 심화 실습 | [Study/OB](./Study/OB) |
| 기초 통계·머신러닝 | [Study/YB](./Study/YB) |

## 기술 스택

- **Deep Learning & NLP:** PyTorch, Hugging Face Transformers, GloVe, TF-IDF, LDA
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **Data & XAI:** Python, Pandas, NumPy, Matplotlib, Seaborn, SHAP

프로젝트별 README에 **문제 정의 → 데이터 처리 → 주요 모델·실험 → 성능 및 분석 결과 → 코드** 순서로 기록
