# Olist E-commerce Customer & Seller Analysis

브라질 E-commerce 플랫폼 **Olist**의 실제 거래 데이터를 활용하여 고객 및 셀러 행동을 분석하고, **셀러 이탈 가능성을 예측한 데이터 분석·머신러닝 프로젝트**입니다.

## Project Overview

E-commerce 플랫폼에서는 고객의 구매 행동뿐만 아니라 플랫폼에 입점한 셀러가 지속적으로 활동하는지 파악하는 것도 중요합니다.

본 프로젝트에서는 Olist의 고객, 주문, 상품, 결제, 리뷰, 셀러 데이터를 결합하여 분석용 데이터셋을 구축하고,

* 고객 구매 행동 분석
* RFM 기반 고객 Feature 생성
* 리뷰 텍스트 및 감성 정보 활용
* 셀러 운영 특성 분석
* 셀러 이탈 예측

까지 분석 범위를 확장했습니다.

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**를 활용했습니다.

주요 데이터:

* Customers
* Orders
* Order Items
* Payments
* Products
* Reviews
* Sellers

원본 데이터는 여러 관계형 테이블로 분리되어 있기 때문에 분석 목적에 맞게 직접 병합하여 사용했습니다.

## Data Engineering

### 1. 관계형 데이터 병합

주요 Key를 기준으로 데이터를 연결했습니다.

* `customer_id`
* `order_id`
* `seller_id`
* `product_id`

특히 `items`, `payments`처럼 하나의 주문에 여러 행이 존재하는 데이터는 바로 병합할 경우 주문 건수가 중복될 수 있으므로 **먼저 주문 단위로 집계한 뒤 병합**했습니다.

이를 통해 구매 횟수와 결제 금액이 중복 계산되는 문제를 방지했습니다.

### 2. Customer Features

고객의 구매 행동을 나타내기 위해 다음과 같은 Feature를 생성했습니다.

* Recency
* Frequency
* Monetary
* 지역 정보
* 배송 특성
* 주문 및 결제 특성
* 리뷰 점수 및 리뷰 수

### 3. Review & Sentiment Features

리뷰 데이터를 활용하여 고객 및 셀러의 평가 특성을 추가했습니다.

* 평균 리뷰 평점
* 리뷰 감성 점수
* 감성 점수 변동성
* 배송 관련 감성
* 가격 및 가치 관련 감성
* 제품 품질 관련 감성
* 구매 과정 관련 이슈

리뷰 텍스트에는 LDA 및 Aspect 기반 Feature를 적용하여 단순 평점 외에도 **어떤 유형의 문제가 발생했는지**를 분석할 수 있도록 구성했습니다.

## Seller Churn Prediction

셀러별 주문, 배송, 가격, 리뷰 및 감성 Feature를 구축하여 **셀러 이탈 여부를 예측하는 Binary Classification 모델**을 구축했습니다.

주요 모델:

* XGBoost
* LightGBM

Hyperparameter Optimization에는 **Optuna**를 활용했습니다.

## Model Performance

최종 모델 비교 결과:

| Model    | F1-score | ROC-AUC |
| -------- | -------: | ------: |
| XGBoost  |    0.692 |   0.832 |
| LightGBM |    0.698 |   0.827 |

F1-score 기준으로는 LightGBM이 소폭 높았으며, ROC-AUC 기준으로는 XGBoost가 더 높은 성능을 보였습니다.

모델 성능뿐만 아니라 실제 이탈과 연관된 Feature를 해석하기 위해 **SHAP 분석**을 추가로 수행했습니다.

## Key Findings

### 1. 배송 경험이 셀러 이탈과 밀접하게 연관됨

SHAP 분석에서 다음 배송 관련 변수가 높은 중요도를 나타냈습니다.

* `delay_rate`
* `avg_delivery_days`

배송 지연 발생률이 높거나 배송 시간이 길수록 셀러 이탈 방향으로 예측되는 경향이 나타났습니다.

이를 통해 **배송 서비스 품질이 셀러의 지속적인 플랫폼 활동과 관련된 중요한 운영 지표**임을 확인했습니다.

### 2. 리뷰의 평균뿐 아니라 감성의 안정성도 중요

`seller_sentiment_std`가 주요 Feature로 나타났습니다.

단순 평균 평점 외에도 셀러가 받는 리뷰의 감성 분포와 변동성이 셀러 특성을 설명하는 데 기여했습니다.

이는 리뷰 데이터를 단순 평점으로 축약하지 않고 **텍스트 감성 정보를 추가 Feature로 활용할 수 있음을 보여줍니다.**

### 3. 낮은 거래량의 셀러에서 이탈 위험이 높게 나타남

`total_orders` 역시 상위 중요 변수로 나타났으며, 주문량이 매우 적은 셀러에서 이탈 위험이 높게 나타나는 경향을 확인했습니다.

다만 활동 기간이 짧아 누적 주문이 적은 신규 셀러와 실제 저활동 셀러가 혼재할 수 있으므로 해당 변수 해석에는 주의가 필요합니다.

### 4. 리뷰 텍스트에서 추출한 이슈 Feature도 실제 예측에 기여

다음과 같은 리뷰 텍스트 기반 변수 역시 셀러 이탈 예측에 기여했습니다.

* 구매 과정 관련 문제
* 배송 관련 문제
* 가격 변동성

즉, 정형 거래 데이터뿐만 아니라 **비정형 리뷰 텍스트에서 추출한 정보도 셀러 이탈 예측에 활용할 수 있음**을 확인했습니다.

## Data Leakage Consideration

분석 과정에서 리뷰 평점과 리뷰 텍스트 감성 간 강한 연관성으로 인해 **Target Leakage 가능성**도 검토했습니다.

리뷰 평점을 예측할 때 리뷰 감성 Feature를 그대로 사용하는 경우 사실상 정답과 매우 가까운 정보를 모델에 제공할 수 있으므로, 분석 목적에 따라 해당 Feature를 제외해야 한다고 판단했습니다.

따라서 단순히 높은 모델 성능을 만드는 것보다 **예측 시점에서 실제 사용할 수 있는 변수인지, 비즈니스적으로 의미 있는 Feature인지 검토하는 과정**을 함께 수행했습니다.

## Project Highlights

* 여러 관계형 E-commerce 데이터를 직접 병합하여 분석 데이터 구축
* 주문 단위 중복을 고려한 데이터 집계 및 전처리
* RFM 기반 고객 행동 Feature 생성
* 리뷰 텍스트 감성 및 LDA/Aspect Feature 활용
* XGBoost·LightGBM 기반 셀러 이탈 예측
* Optuna 기반 Hyperparameter Tuning
* SHAP을 활용한 모델 결과 해석
* Target Leakage 가능성 검토

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `LightGBM` `Optuna` `SHAP` `NLP` `Matplotlib` `Seaborn`

## Repository Structure

```text
06_Olist_Ecommerce_Analysis/
├── README.md
├── 01_Data_Preprocessing.ipynb
├── 02_Customer_Analysis.ipynb
└── 03_Seller_Analysis.ipynb
```
