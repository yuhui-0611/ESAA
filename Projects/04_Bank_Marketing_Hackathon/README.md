# Bank Marketing Hackathon | 불균형 고객 반응 예측과 의사결정 임계값 탐색

> **핵심:** 정기예금 가입 반응 예측. 불균형 클래스 대응, 특성 선택, 예측확률 임계값·앙상블 가중치 조합 실험

## 문제 정의
- 은행 마케팅 캠페인의 정기예금 가입 반응을 예측하는 이진 분류 
- 클래스 불균형을 고려한 학습과 **분류 확률의 임계값 조정**을 중심으로 실험

## 분석 흐름
1. `CreditScore`, `AccountLengthYears`, `TenureWithBank`, `InvestmentPortfolioValue`, `CallResponseScore`, `MarketingScore`, `ResponsePropensity` 등 결측 확인
2. 변수형 변환·파생 변수 및 범주형 처리
3. **LightGBM**으로 baseline F1 평가, **CatBoost**로 범주형 데이터 반영
4. 특징 중요도 기반 제거 전후 성능 및 `class_weights` 처리 비교
5. Validation 예측 확률에 대해 임계값을 격자 탐색하고 **F1 최대 구간** 확인
6. CatBoost–LightGBM 확률 가중 결합과 임계값을 함께 탐색

## 심화 실험 — 확률과 판정 기준의 분리
기본 확률 임계값 `0.5`만 사용하는 대신 다음과 같이 검증셋에서 여러 기준을 비교함

```python
thresholds = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_valid, (proba_valid >= t).astype(int)) for t in thresholds]
```

또한 중요도 기준으로 피처를 선택하거나 모델 예측확률을 가중 평균하여, **특성 선택과 분류 임계값 변화에 따른 Precision·Recall·F1 차이**를 탐색함

## 실험 결과 및 분석

- **초기 검증 F1 0.5001:** LightGBM 기준 실험의 검증 분할 기록
- **임계값 탐색:** CatBoost 실험에서 검증 F1이 최대인 분류 기준 약 **0.423** 기록
- **핵심 관찰:** 학습된 확률을 동일한 모델이라도 어느 임계값에서 양성으로 분류하는지에 따라 Precision·Recall·F1 변화
- **앙상블 검토:** LightGBM·CatBoost의 예측확률을 혼합하고, 가중치와 분류 기준을 함께 탐색

**실험 요약:** 클래스 불균형을 고려한 모델 학습과 분류 임계값 조정을 별도 단계로 설계. F1을 기준으로 확률 앙상블 및 분류 기준 탐색

## File
- [`bank_marketing_prediction.ipynb`](./bank_marketing_prediction.ipynb): 전체 전처리, 분류 모델 및 임계값/앙상블 실험

**Stack** `Python` `Pandas` `Scikit-learn` `LightGBM` `CatBoost` `F1 optimization`
