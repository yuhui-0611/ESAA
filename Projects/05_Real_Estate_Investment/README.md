# Real Estate Investment Decision Prediction

가구 및 경제 관련 데이터를 활용하여 **부동산 투자 의사결정을 예측하는 머신러닝 프로젝트**입니다.

## Project Overview

본 프로젝트는 다양한 가구 및 경제적 특성이 부동산 투자 의사결정과 어떠한 관계를 가지는지 분석하고, 이를 바탕으로 투자 여부를 예측하는 것을 목표로 진행했습니다.

프로젝트에서는 여러 시점의 데이터를 통합하고 분석에 필요한 변수를 구성한 뒤, 머신러닝 기반 분류 모델을 구축했습니다.

## Objective

다양한 개인 및 가구 특성을 활용하여 부동산 투자 의사결정을 나타내는 변수인 `investment_decision`을 예측했습니다.

이를 통해

* 어떤 특성이 투자 의사결정과 관련되는지
* 머신러닝으로 투자 의사결정을 어느 정도 분류할 수 있는지

를 분석하고자 했습니다.

## Analysis Process

1. 여러 연도의 데이터 통합
2. 분석 대상 변수 선정
3. 데이터 전처리
4. Exploratory Data Analysis
5. Feature Engineering
6. 머신러닝 모델 구축
7. XGBoost 기반 분류
8. 변수 중요도 및 결과 해석

## Model

주요 예측 모델로 **XGBoost Classifier**를 활용했습니다.

XGBoost는 여러 개의 Decision Tree를 순차적으로 학습하여 이전 모델의 오차를 보완하는 Boosting 기반 알고리즘으로, 복잡한 비선형 관계를 학습할 수 있다는 장점이 있습니다.

## Key Point

이 프로젝트에서는 단순히 예측 정확도를 높이는 것뿐만 아니라 **어떤 변수들이 부동산 투자 의사결정에 중요한 역할을 하는지 해석하는 것**에도 초점을 두었습니다.

따라서 머신러닝 모델 구축과 함께 Feature Importance를 활용하여 주요 변수의 영향력을 살펴보았습니다.

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`

## Files

```text
05_Real_Estate_Investment/
├── README.md
├── 01_Data_Analysis.ipynb
└── 02_Final_XGBoost_Model.ipynb
```
