# ESAA | Data Science & Deep Learning Portfolio

> **데이터 전처리에서 예측 모델 구현을 넘어, 딥러닝 기반 비정형 데이터 분석·모델 성능 실험·설명 가능한 AI·의사결정 지표 설계까지**

데이터 분석 학회 **ESAA** 활동에서 작성한 프로젝트 및 학습 노트북을 정리한 포트폴리오입니다. 정형 데이터 회귀·분류부터 **사전학습 Transformer 감성 추론, PyTorch MLP 비교 실험, 다중 모델 앙상블 및 임계값 기반 빈곤율 추정**까지 문제 특성에 맞는 분석 방법을 적용했습니다.

## Selected Highlights

### 01 | Transformer NLP + PyTorch Deep Learning — Olist 셀러 이탈 분석
- Olist 관계형 거래 테이블을 주문·고객·셀러 단위로 연결하고, 구매·배송·리뷰 특징량을 생성했습니다.
- `cardiffnlp/twitter-xlm-roberta-base-sentiment` **사전학습 Transformer**를 사용해 리뷰 텍스트의 감성 라벨/점수를 추론하고 정형 데이터와 결합했습니다. (*파인튜닝이 아닌 사전학습 모델 활용*)
- PyTorch MLP (64→32→1, ReLU·Dropout)를 직접 구현하고, BCEWithLogitsLoss·Adam·Early Stopping 및 **5개 시드 반복 실험**으로 트리 모델과 비교했습니다.
- MLP의 F1 **0.646**, ROC-AUC **0.799**; 5회 반복 실험 평균 F1 **0.648±0.005**, AUC **0.797±0.002**. XGBoost·LightGBM과의 비교를 통해 모델 복잡도보다 데이터 특성에 적합한 접근의 중요성을 확인했습니다.
- [프로젝트 상세 보기](./Projects/06_Olist_Ecommerce_Analysis)

### 02 | Threshold-based Poverty Estimation — 소비·빈곤 동시 예측
- 가구별 소비액을 회귀 예측한 뒤, 19개 빈곤선(threshold)에 대한 빈곤율을 조사(survey) 단위로 재계산했습니다.
- 빈곤율 오차 90% + 소비 오차 10%의 복합 목적함수를 직접 구성하고, LightGBM·XGBoost·CatBoost의 **앙상블 가중치를 탐색**했습니다.
- 임계값 근처의 불확실한 예측을 **Random Forest 빈곤확률**로 재분류하는 규칙 기반 보정 실험을 진행했습니다.
- 이는 **임계값·가중치·예측 보정에 대한 시나리오 비교**이며, 확률표본을 반복 생성하는 *Monte Carlo 시뮬레이션*이라고 주장하지 않습니다.
- [프로젝트 상세 보기](./Projects/07_Poverty_Prediction)

### 03 | Explainable AI — 부동산 투자 의사결정
- 2019–2025년 가구 자료의 상이한 변수 체계를 정규화해 통합하고 XGBoost 분류 모델을 구축했습니다.
- **ROC-AUC 0.7443 / F1 0.6816**을 기록했으며, Feature Importance와 **SHAP**으로 변수의 영향 방향을 해석했습니다.
- [프로젝트 상세 보기](./Projects/05_Real_Estate_Investment)

## Projects

| 프로젝트 | 핵심 분석 및 기술 | 상세 |
|---|---|---|
| **Olist E-commerce Analysis** | **Transformer 감성 추론 · PyTorch MLP · 반복 시드 실험 · Optuna · SHAP** | [바로가기](./Projects/06_Olist_Ecommerce_Analysis) |
| **Poverty Prediction Challenge** | **복합 평가지표 · 19개 빈곤선 · 앙상블 가중치 탐색 · 임계값 보정** | [바로가기](./Projects/07_Poverty_Prediction) |
| **Real Estate Investment** | 다년도 데이터 통합 · XGBoost · SHAP · 해석 가능한 예측 | [바로가기](./Projects/05_Real_Estate_Investment) |
| **Author Classification NLP** | 단어/문자 벡터화 · GloVe 문장 표현 · OOF 예측 · 앙상블 | [바로가기](./Projects/03_Author_Classification_NLP) |
| **Bank Marketing Hackathon** | LightGBM · CatBoost · 불균형 분류 · F1 임계값 최적화 | [바로가기](./Projects/04_Bank_Marketing_Hackathon) |
| **Hotel Reservation Cancellation** | 분류기 비교 · 특성 선택 · Optuna · 앙상블 | [바로가기](./Projects/02_Hotel_Reservation_Cancellation) |
| **Movie Box Office Prediction** | 로그 변환 · 선형/트리 회귀 · 5-Fold CV · 예측 결합 | [바로가기](./Projects/01_Movie_BoxOffice_Prediction) |

## Deep Learning & Advanced Study

실전 프로젝트와 별도로 `Study/OB`에서 심화 모델을 구현·학습한 노트북을 정리했습니다. 아래 내용은 **학습 실습**입니다.

| 분야 | 주요 학습 내용 | 경로 |
|---|---|---|
| PyTorch | 텐서 연산, 모델 설계, 학습 루프 | [`Study/OB/04_PyTorch`](./Study/OB/04_PyTorch) |
| Deep Learning / CNN | 신경망, 합성곱, 최적화 | [`Study/OB/05_Deep_Learning`](./Study/OB/05_Deep_Learning) |
| Sequential Models | RNN, LSTM, GRU | [`Study/OB/06_RNN`](./Study/OB/06_RNN) |
| Time Series | 시계열 분석·예측 실습 | [`Study/OB/07_Time_Series`](./Study/OB/07_Time_Series) |
| Generative Models | Autoencoder, GAN, KerasNLP | [`Study/OB/08_Generative_Models`](./Study/OB/08_Generative_Models) |
| Other | NLP, 추천 시스템, 머신러닝 응용 | [`Study/OB`](./Study/OB) |

기초 학습 및 데이터 분석 노트북은 [`Study/YB`](./Study/YB)에서 확인할 수 있습니다.

## Tech Stack

**Deep Learning & NLP** `PyTorch` `Hugging Face Transformers` `TensorFlow / Keras (Study)` `GloVe` `TF-IDF`  
**Machine Learning** `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `Optuna`  
**Analytics & Interpretability** `Pandas` `NumPy` `Matplotlib` `Seaborn` `SHAP` `RFM` `LDA`

## Repository Guide

```text
ESAA/
├── Projects/           # 7개 분석 프로젝트: 각 폴더의 README에 분석 과정과 실험 근거 기록
│   ├── 01_Movie_BoxOffice_Prediction/
│   ├── 02_Hotel_Reservation_Cancellation/
│   ├── 03_Author_Classification_NLP/
│   ├── 04_Bank_Marketing_Hackathon/
│   ├── 05_Real_Estate_Investment/
│   ├── 06_Olist_Ecommerce_Analysis/
│   └── 07_Poverty_Prediction/
└── Study/
    ├── YB/              # 데이터 분석·머신러닝 기초
    └── OB/              # NLP·딥러닝·생성 모델 심화 실습
```
