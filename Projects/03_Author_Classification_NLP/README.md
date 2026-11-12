# Author Classification NLP | 문체 특징 기반 작가 분류

**소설 텍스트를 어휘·문체·사전학습 임베딩 특징으로 변환하고 여러 분류 모델을 결합한 5개 클래스 작가 분류**

## 문제 정의
- **예측 대상:** 주어진 텍스트의 작가 클래스 (5개)
- **주요 기술:** TF-IDF, CountVectorizer, Word/Character N-gram, GloVe, Logistic Regression, Naive Bayes, XGBoost, LightGBM

## 주요 구현
- **문체 특징 분석:** 작가별 단어 빈도, 텍스트 길이, 불용어·문장부호 분포 탐색
- **희소 텍스트 특징 생성:** Word·Character N-gram 기반 TF-IDF 및 CountVectorizer 적용
- **사전학습 표현 활용:** GloVe 100차원 단어 벡터를 조회하고, 합산·L2 정규화 방식으로 문장별 밀집 벡터 생성
- **OOF 메타 특징:** 교차검증에서 생성한 클래스별 예측 확률을 후속 모델 입력 특징으로 활용
- **분류·앙상블:** Logistic Regression, Naive Bayes, XGBoost, LightGBM 및 예측 확률 결합 비교
- **평가 지표:** 다중 클래스 LogLoss·Accuracy

## 분석 결과
- 단어 출현 빈도뿐 아니라 문자 단위 문체·사전학습 단어 임베딩을 분류 특징으로 활용
- OOF 기반 확률 특징 생성 및 모델별 확률 출력 비교

## 코드
| Notebook | 내용 |
|---|---|
| [01_EDA](./01_EDA.ipynb) | 텍스트 탐색 및 문체 분석 |
| [02_Modeling](./02_Modeling.ipynb) | TF-IDF·GloVe 특징 생성, 분류기·OOF·앙상블 |
