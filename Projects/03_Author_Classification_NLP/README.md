# Author Classification | 텍스트 표현·임베딩·OOF 앙상블 비교

> **프로젝트 핵심:** 작가별 문체 특성을 수치화하고, **TF-IDF/Count 기반 특징과 사전학습 GloVe 임베딩**, 여러 분류기의 **OOF 예측 및 확률 앙상블**을 비교했습니다.

## 문제 정의
소설의 짧은 텍스트만으로 작가를 **5개 클래스 중 하나로 분류**하는 자연어 처리 경진대회 문제를 다뤘습니다. 텍스트 길이와 단어 분포뿐 아니라 문장부호·불용어·통계적 문체 특징이 식별력을 가질 수 있는지 검토했습니다.

## 분석 파이프라인
1. 작가별 텍스트 수·길이, 불용어, n-gram, WordCloud 등 EDA
2. 텍스트 전처리 및 문장부호·불용어·길이 등 문체 특징 생성
3. **Word/Character Count 및 TF-IDF** 표현 실험
4. **GloVe 사전학습 단어벡터**를 평균화한 문장 수준 표현(`sent2vec`) 생성
5. Logistic Regression, XGBoost, LightGBM 등 분류기를 학습하고 **LogLoss** 기반 성능 평가
6. **교차검증 OOF(out-of-fold) 확률**을 생성해 다중 모델의 확률 앙상블 비교

## 핵심 구현 포인트
- **희소 벡터와 밀집 벡터의 비교:** 표면적 단어 빈도/문체 특징과 단어 임베딩 기반 의미 표현의 차이를 실험했습니다.
- **OOF 예측:** 학습 폴드와 예측 폴드를 분리해 앙상블 실험에 사용할 클래스 확률을 생성했습니다.
- **LogLoss:** 정확도만이 아니라 5개 클래스에 대한 예측확률의 신뢰도를 평가했습니다.

## Files
- [`01_EDA.ipynb`](./01_EDA.ipynb): 데이터 탐색, 작가별 단어 특성 분석
- [`02_Modeling.ipynb`](./02_Modeling.ipynb): 텍스트 벡터화, GloVe, 분류기, OOF/앙상블

**Stack** `Python` `Scikit-learn` `TF-IDF` `CountVectorizer` `GloVe` `XGBoost` `LightGBM`
