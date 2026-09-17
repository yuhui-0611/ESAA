# Poverty Prediction Challenge

가구 설문 데이터를 활용하여 **가구의 소비 수준과 빈곤 정도를 예측하는 머신러닝 프로젝트**입니다.

## Project Overview

빈곤 수준을 정확하게 측정하는 것은 정책 수립과 지원 대상 선정에 중요하지만, 대규모 가구 소비 조사는 많은 시간과 비용이 필요합니다.

본 프로젝트에서는 가구 및 개인의 특성을 활용하여 소비 수준을 예측하고, 이를 기반으로 빈곤 수준을 추정하는 머신러닝 모델을 구축했습니다.

단일 모델만 구축하는 것이 아니라 여러 전처리 방식과 모델링 전략을 반복적으로 실험하며 성능 개선 과정을 비교했습니다.

## Objective

주요 목표는 다음과 같습니다.

* 가구 및 개인 특성을 활용한 소비 수준 예측
* 빈곤 정도를 설명할 수 있는 주요 변수 탐색
* 다양한 전처리 및 모델링 전략 비교
* 반복적인 실험을 통한 예측 성능 개선

## Analysis Process

1. 데이터 구조 및 변수 확인
2. 결측치 및 이상치 처리
3. 범주형·수치형 변수 전처리
4. Baseline 모델 구축
5. Feature Engineering
6. 모델 및 파라미터 변경
7. 추가 전처리 적용
8. 각 실험 결과 비교

## Experiment History

프로젝트 과정에서 여러 버전의 모델을 구축하며 개선 과정을 기록했습니다.

```text
07_Poverty_Prediction/
├── README.md
├── 01_Baseline.ipynb
├── 02_Model_Ver1.ipynb
├── 03_Model_Ver1_Plus.ipynb
├── 04_Model_Ver2.ipynb
├── 05_Model_Ver3.ipynb
├── 06_Model_Ver4.ipynb
└── 07_Additional_Preprocessing.ipynb
```

각 Notebook에서는 전처리 방식, Feature 구성, 모델링 전략 등을 변경하면서 성능 변화를 비교했습니다.

## Key Point

이 프로젝트에서는 하나의 최종 모델을 만드는 것보다 **실제 Competition 문제를 해결하는 과정에서 여러 가설을 세우고 반복적으로 실험한 과정**에 초점을 두었습니다.

특히 다음 과정을 경험했습니다.

* 데이터 전처리 전략 비교
* Feature Engineering
* 모델 변경 및 성능 비교
* 반복적인 실험을 통한 개선
* Competition 형태의 머신러닝 문제 해결

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Machine Learning` `Feature Engineering`

## Repository Structure

```text
07_Poverty_Prediction/
├── README.md
├── 01_Baseline.ipynb
├── 02_Model_Ver1.ipynb
├── 03_Model_Ver1_Plus.ipynb
├── 04_Model_Ver2.ipynb
├── 05_Model_Ver3.ipynb
├── 06_Model_Ver4.ipynb
└── 07_Additional_Preprocessing.ipynb
```
