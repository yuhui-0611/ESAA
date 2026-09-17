# Poverty Prediction Challenge

가구 설문 데이터를 활용하여 **가구의 소비 수준과 빈곤 정도를 예측하는 머신러닝 프로젝트**입니다.

## Project Overview

빈곤 수준을 정확하게 측정하는 것은 정책 수립과 지원 대상 선정에 중요하지만, 대규모 가구 소비 조사는 많은 시간과 비용을 필요로 합니다.

본 프로젝트에서는 가구 및 개인 특성을 활용하여 소비 수준을 예측하고, 다양한 데이터 전처리 및 모델링 전략을 반복적으로 실험했습니다.

하나의 모델만 구축하기보다 **Baseline부터 여러 모델 버전과 추가 전처리까지 단계적으로 개선하는 Competition 방식**으로 프로젝트를 진행했습니다.

---

## Objective

주요 목표는 다음과 같습니다.

* 가구 및 개인 특성을 활용한 소비 수준 예측
* 다양한 Feature Engineering 전략 비교
* 여러 모델과 파라미터 실험
* 전처리 방식에 따른 성능 변화 확인
* 반복적인 실험을 통한 예측 성능 개선

---

## Analysis Process

1. 데이터 구조 및 변수 확인
2. 결측치 및 이상치 처리
3. 범주형·수치형 변수 전처리
4. Baseline 모델 구축
5. Feature Engineering
6. 모델 변경 및 파라미터 조정
7. 추가 전처리 적용
8. 실험별 결과 비교

---

## Experiment History

프로젝트 과정에서 여러 버전의 모델을 구축하며 성능 개선을 반복했습니다.

```text
01_Baseline.ipynb
02_Model_Ver1.ipynb
03_Model_Ver1_Plus.ipynb
04_Model_Ver2.ipynb
05_Model_Ver3.ipynb
06_Model_Ver4.ipynb
07_Additional_Preprocessing.ipynb
```

각 Notebook에서는 전처리 방식, Feature 구성, 모델링 전략 등을 변경하면서 성능 변화를 비교했습니다.

---

## Key Point

이 프로젝트에서는 단순히 하나의 최종 모델을 만드는 것보다 **여러 가설을 세우고 반복적인 실험을 통해 모델을 개선하는 과정**에 중점을 두었습니다.

특히 다음과 같은 경험을 포함합니다.

* Baseline 모델 구축
* Feature Engineering
* 전처리 전략 비교
* 모델 변경
* Hyperparameter 조정
* 실험 결과 비교 및 개선

---

## Project Highlights

* Competition 형태의 머신러닝 문제 해결
* 여러 버전의 모델링 실험 수행
* 반복적인 Feature Engineering
* 전처리 전략에 따른 성능 변화 비교
* Baseline → 개선 모델로 이어지는 실험 과정 기록

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Machine Learning` `Feature Engineering`

---

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
