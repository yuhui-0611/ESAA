# Author Classification | 텍스트 표현·임베딩·OOF 앙상블 비교

> **핵심:** 작가의 문체·어휘 정보를 희소 벡터와 사전학습 임베딩으로 수치화. OOF 예측과 다중 모델 확률 앙상블 구성

## 문제 정의
- 소설의 짧은 텍스트만으로 작가를 **5개 클래스 중 하나로 분류**하는 자연어 처리 경진대회 문제를 다룸
- 텍스트 길이와 단어 분포뿐 아니라 문장부호·불용어·통계적 문체 특징이 식별력을 가질 수 있는지 검토함

## 분석 파이프라인
1. 작가별 텍스트 수·길이, 불용어, n-gram, WordCloud 등 EDA
2. 텍스트 전처리 및 문장부호·불용어·길이 등 문체 특징 생성
3. **Word/Character Count 및 TF-IDF** 표현 실험
4. **GloVe 사전학습 단어벡터**를 합산·L2 정규화한 문장 수준 표현(`sent2vec`) 생성
5. Multinomial Naive Bayes, Logistic Regression, XGBoost, LightGBM 등 분류기 학습 및 **LogLoss** 평가
6. **교차검증 OOF(out-of-fold) 확률**을 생성해 다중 모델의 확률 앙상블 비교

## 모델링 및 실험 결과

- **Word/Character TF-IDF·Count 비교:** 어휘 및 철자 패턴이 분류에 미치는 영향 탐색. 8-Fold CV 기반 Multinomial Naive Bayes의 표현별 LogLoss 비교
- **GloVe 100차원 표현:** 사전학습 단어벡터 조회 → 문장별 벡터 합산 → L2 정규화 → 분류 모델 입력 특성 생성
- **OOF 메타 특징:** 폴드 외부 데이터에 대한 클래스별 확률을 생성하여 후속 앙상블 입력 특성으로 사용
- **평가 지표:** 5개 클래스의 확률 품질을 고려한 다중 클래스 LogLoss

| 비교 실험 | LogLoss | 측정 조건 |
|---|---:|---|
| Word TF-IDF + Multinomial NB | **1.0899** | 8-Fold CV 평균 |
| Character TF-IDF + Multinomial NB | **0.9831** | 8-Fold CV 평균 |
| Word Count + Multinomial NB | **0.9922** | 8-Fold CV 평균 |
| XGBoost 분류 실험 | **0.4534** | 단일 검증 분할, 다중 클래스 LogLoss |

**분석 결과:** Multinomial NB 8-Fold 실험에서 Character TF-IDF의 LogLoss 0.9831, Word TF-IDF 1.0899 기록. 해당 모델·평가 조건에서 문자 단위 표현의 분류 성능 우위 확인

## Files
- [`01_EDA.ipynb`](./01_EDA.ipynb): 데이터 탐색, 작가별 단어 특성 분석
- [`02_Modeling.ipynb`](./02_Modeling.ipynb): 텍스트 벡터화, GloVe, 분류기, OOF/앙상블

**Stack** `Python` `Scikit-learn` `TF-IDF` `CountVectorizer` `GloVe` `XGBoost` `LightGBM`
