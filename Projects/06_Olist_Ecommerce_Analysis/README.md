# Olist E-commerce Customer & Seller Analysis

브라질 E-commerce 플랫폼 **Olist**의 실제 거래 데이터를 활용하여 고객과 셀러의 행동을 분석한 데이터 분석 프로젝트입니다.

## Project Overview

E-commerce 플랫폼에서는 신규 고객 확보뿐만 아니라 기존 고객의 행동과 판매자의 활동을 이해하는 것이 중요합니다.

본 프로젝트에서는 Olist의 고객, 주문, 상품, 결제 등 여러 데이터를 결합하여 E-commerce 플랫폼의 고객 및 셀러 특성을 분석했습니다.

여러 개의 관계형 데이터를 직접 병합하고 분석용 데이터셋을 구축하는 과정부터 시작했습니다.

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**를 활용했습니다.

주요 데이터는 다음과 같습니다.

* Customers
* Orders
* Order Items
* Payments
* Products
* Sellers

각 데이터의 공통 Key를 활용하여 여러 테이블을 병합하고 분석 목적에 맞는 데이터셋을 구축했습니다.

## Analysis Process

### 1. Data Preprocessing

* 데이터 구조 확인
* 여러 테이블 간 관계 파악
* 고객 및 주문 데이터 병합
* 결측치 및 이상치 처리
* 분석용 Feature 생성

### 2. Customer Analysis

고객의 구매 행동을 분석하고 고객별 특성을 파악했습니다.

고객의 최근 구매 시점, 구매 횟수, 구매 금액 등을 활용하는 **RFM 분석** 등을 통해 고객 행동을 정량적으로 분석했습니다.

### 3. Seller Analysis

판매자의 주문 및 거래 특성을 분석하여 셀러별 활동 패턴을 파악했습니다.

판매자의 거래량과 활동 특성을 기반으로 플랫폼 내 판매자의 행동을 분석했습니다.

## Key Point

Olist 데이터는 하나의 정제된 데이터셋이 아니라 **여러 개의 관계형 테이블로 구성되어 있기 때문에**, 분석 전에 필요한 데이터를 직접 연결하고 분석용 데이터셋을 구축해야 합니다.

따라서 이 프로젝트에서는 단순 모델링보다

* 데이터 구조 이해
* 데이터 병합
* Feature Engineering
* 고객 행동 분석
* 판매자 행동 분석

과 같은 실제 데이터 분석 과정에 중점을 두었습니다.

## Tech Stack

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Scikit-learn`

## Files

```text
06_Olist_Ecommerce_Analysis/
├── README.md
├── 01_Data_Preprocessing.ipynb
├── 02_Customer_Analysis.ipynb
└── 03_Seller_Analysis.ipynb
```
