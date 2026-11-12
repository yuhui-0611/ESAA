# Olist E-commerce Analysis | Transformer NLP · PyTorch MLP · 셀러 이탈 분석

**브라질 이커머스 주문·배송·리뷰를 연결해 고객 행동과 셀러 이탈 패턴 분석. 사전학습 Transformer 감성 추론과 PyTorch MLP를 각각 적용하고 트리 기반 모델과 비교**

## 문제 정의
- **데이터:** Brazilian E-Commerce Public Dataset by Olist
- **분석 과제:** 고객 행동·리뷰 분석, 셀러 단위 이탈 여부 분류
- **주요 기술:** XLM-RoBERTa, Hugging Face Transformers, PyTorch, XGBoost, LightGBM, SHAP, LDA

## 주요 구현

### 1. 거래·고객·셀러 데이터 결합
- 주문·상품·고객·셀러·결제·리뷰 데이터의 키 관계 정리
- 주문 수, 매출, 배송 기간·지연율, 리뷰 지표 등 분석 특징 생성
- 고객별 리뷰·구매 특징 및 셀러별 거래 특징 집계

### 2. 사전학습 Transformer로 리뷰 감성 추론
- Hugging Face **`cardiffnlp/twitter-xlm-roberta-base-sentiment`** 모델 로드
- 리뷰 텍스트의 배치 추론을 통해 긍정·중립·부정 라벨 및 확률 점수 생성
- 감성 점수의 고객·셀러 단위 집계, LDA 기반 리뷰 주제 탐색
- **사전학습 모델 추론 활용** — 별도 파인튜닝 수행으로 기술하지 않음

### 3. PyTorch MLP 직접 구현·학습
- 수치형 특징 표준화, **64 → 32 → 1** 완전연결 신경망 구성
- `ReLU`, `Dropout(0.3)`, `BCEWithLogitsLoss`, `Adam` 적용
- 검증 손실 기반 Early Stopping 및 **서로 다른 초기 시드 5회 학습·평가**
- 이진 분류 F1·ROC-AUC를 기준으로 XGBoost·LightGBM과 결과 비교

```text
입력 특징 → 표준화 → Linear(64) → ReLU → Dropout
                   → Linear(32) → ReLU → Dropout
                   → Linear(1) → 이탈 확률
```

### 4. 트리 앙상블 및 모델 해석
- XGBoost·LightGBM 학습, 하이퍼파라미터 탐색 및 확률 보정 실험
- 셀러 이탈 예측에 반영된 주요 특징을 SHAP으로 분석
- **예측 기준일 이전의 주문 수·매출·리뷰 수 재집계**를 통한 시점 관련 특성 보완 실험

## 분석 결과
- 자연어 리뷰를 감성 특징으로 수치화하고 거래·배송 데이터와 결합
- PyTorch 신경망 학습 루프·검증·Early Stopping·반복 시드 평가 직접 구현
- 트리 앙상블·MLP의 성능 차이 및 주요 예측 특징 비교
- **검증 범위:** 기준일 이후에 생성된 일부 특징의 완전한 배제 여부가 코드에서 확인되지 않아, 기존 단일 테스트 분할의 F1·ROC-AUC를 독립적인 최종 예측 성능으로 제시하지 않음

![Seller Churn SHAP Summary](./images/shap_summary.png)

## 코드
| Notebook | 내용 |
|---|---|
| [01_Data_Preprocessing](./01_Data_Preprocessing.ipynb) | 이커머스 원천 테이블 정리 |
| [02_Customer_Analysis](./02_Customer_Analysis.ipynb) | 고객 분석·Transformer 감성 추론·LDA |
| [03_Seller_Analysis](./03_Seller_Analysis.ipynb) | 셀러 이탈 정의·모델 학습·PyTorch MLP·SHAP |
