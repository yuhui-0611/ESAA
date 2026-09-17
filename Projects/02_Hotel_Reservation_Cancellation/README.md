# Hotel Reservation Cancellation Prediction

호텔 예약 데이터를 활용하여 고객의 **예약 취소 여부를 예측**하는 머신러닝 프로젝트입니다.

## Project Overview

호텔 예약 취소는 객실 운영과 수익 관리에 영향을 줄 수 있습니다.
본 프로젝트에서는 고객의 예약 정보와 이용 특성을 기반으로 예약 취소 여부를 예측하고, 다양한 분류 모델의 성능을 비교했습니다.

## Dataset

호텔 예약 고객의 예약 및 이용 정보를 포함한 데이터를 사용했습니다.

주요 변수 예시:

* 숙박 일수
* 예약 리드타임(Lead Time)
* 객실 유형
* 평균 객실 가격
* 시장 세그먼트
* 특별 요청 횟수
* 이전 예약 및 취소 이력

**Target**

* `Canceled`
* `Not_Canceled`

## Analysis Process

1. 데이터 구조 및 분포 확인
2. Exploratory Data Analysis (EDA)
3. 데이터 전처리
4. 범주형 변수 처리
5. Feature Engineering
6. 머신러닝 모델 학습
7. Hyperparameter Tuning
8. 모델별 성능 비교

## Models

다양한 분류 모델을 활용하여 예약 취소 여부를 예측했습니다.

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

일부 모델의 Hyperparameter Optimization에는 **Optuna**를 활용했습니다.

## Evaluation

분류 모델의 성능을 비교하기 위해 다음과 같은 평가 지표를 활용했습니다.

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `CatBoost` `Optuna` `Matplotlib` `Seaborn`

## Files

```text
02_Hotel_Reservation_Cancellation/
├── README.md
└── Hotel_Reservation_Prediction.ipynb
```
