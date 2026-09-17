---

# Author Classification using NLP

소설 텍스트 데이터를 활용하여 **텍스트의 작가를 분류하는 NLP 프로젝트**입니다.

## Project Overview

작가마다 사용하는 단어와 표현 방식에는 일정한 특징이 존재할 수 있습니다.
본 프로젝트에서는 소설 텍스트를 정제하고 벡터화하여 텍스트에 나타나는 특징을 추출한 뒤, 이를 이용해 작가를 분류했습니다.

EDA와 Modeling 과정을 분리하여 텍스트 데이터의 특성을 먼저 탐색한 후 분류 모델을 구축했습니다.

## Analysis Process

### 1. Exploratory Data Analysis

* 데이터 구조 확인
* 작가별 데이터 분포 확인
* 텍스트 길이 분석
* 단어 및 표현 특성 탐색

### 2. Text Preprocessing

텍스트 데이터를 모델 학습에 사용할 수 있도록 정제했습니다.

### 3. Feature Extraction

텍스트를 수치형 데이터로 변환하기 위해 다음과 같은 방법을 활용했습니다.

* CountVectorizer
* TF-IDF

### 4. Modeling

벡터화된 텍스트 데이터를 활용하여 작가 분류 모델을 구축하고 성능을 비교했습니다.

## Key Point

일반적인 정형 데이터가 아니라 **텍스트 데이터를 직접 전처리하고 벡터화하여 머신러닝 모델의 입력으로 활용했다는 점**에 초점을 둔 프로젝트입니다.

텍스트 데이터의 표현 방법에 따라 모델의 성능이 달라질 수 있기 때문에 데이터 전처리와 Feature Extraction 과정이 중요했습니다.

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `NLP` `TF-IDF` `CountVectorizer` `Matplotlib` `Seaborn`

## Files

```text
03_Author_Classification_NLP/
├── README.md
├── 01_EDA.ipynb
└── 02_Modeling.ipynb
```

