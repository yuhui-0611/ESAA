# Bank Marketing Hackathon | 불균형 고객 반응 예측과 의사결정 임계값 탐색

> **프로젝트 핵심:** 은행 마케팅 고객의 반응/가입 여부를 분류하면서 **LightGBM·CatBoost, 특징 선택, 클래스 가중치, F1 최적화 임계값 및 확률 앙상블**을 실험했습니다.

## 문제 정의
은행의 마케팅 캠페인에서 반응 가능성이 높은 고객을 구별하는 이진 분류 과제입니다. 데이터 결측 및 클래스 분포를 확인하고, **분류 확률을 실제 의사결정에 어떻게 변환할지**에 초점을 맞췄습니다.

## 분석 흐름
1. `CreditScore`, `AccountLengthYears`, `TenureWithBank`, `InvestmentPortfolioValue`, `CallResponseScore`, `MarketingScore`, `ResponsePropensity` 등 결측 확인
2. 변수형 변환·파생 변수 및 범주형 처리
3. **LightGBM**으로 baseline F1 평가, **CatBoost**로 범주형 데이터 반영
4. 특징 중요도 기반 제거 전후 성능 및 `class_weights` 처리 비교
5. Validation 예측 확률에 대해 임계값을 격자 탐색하고 **F1 최대 구간** 확인
6. CatBoost–LightGBM 확률 가중 결합과 임계값을 함께 탐색

## 심화 실험 — 확률과 판정 기준의 분리
기본 확률 임계값 `0.5`만 사용하는 대신 다음과 같이 검증셋에서 여러 기준을 비교했습니다.

```python
thresholds = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_valid, (proba_valid >= t).astype(int)) for t in thresholds]
```

또한 중요도 기준으로 피처를 선택하거나 모델 예측확률을 가중 평균하여, **특징 차원과 운영상 판정 기준의 트레이드오프**를 탐색했습니다.

## File
- [`bank_marketing_prediction.ipynb`](./bank_marketing_prediction.ipynb): 전체 전처리, 분류 모델 및 임계값/앙상블 실험

**Stack** `Python` `Pandas` `Scikit-learn` `LightGBM` `CatBoost` `F1 optimization`

