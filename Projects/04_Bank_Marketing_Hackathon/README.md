# Bank Marketing Hackathon | 불균형 고객 반응 예측

**정기예금 가입 여부 예측에서 클래스 불균형, 분류 임계값 및 모델 결합 가중치의 영향을 실험**

## 문제 정의
- **예측 대상:** `TermDepositSubscribed` (정기예금 가입 여부)
- **주요 기술:** LightGBM, CatBoost, Class Weight, Stratified K-Fold, F1 Threshold Search

## 주요 구현
- **데이터 전처리:** 고객·캠페인 변수 정리, 범주형 처리 및 파생 변수 구성
- **불균형 대응:** 클래스 가중치와 `scale_pos_weight` 등 적용 및 검증 결과 비교
- **모델 학습:** LightGBM·CatBoost 분류기 학습, 특성 중요도 기반 변수 선택
- **임계값 탐색:** 예측 확률의 분류 임계값을 조정하며 Precision·Recall·F1 변화 확인
- **확률 앙상블:** 두 모델의 예측 확률을 가중 결합하고 가중치·임계값 조합 탐색

## 분석 결과
- 임계값 `0.5` 고정 시점과 최적화 시점의 F1 비교
- 모델별 클래스 가중치·특성 축소·확률 결합의 효과 검토

## 코드
[bank_marketing_prediction.ipynb](./bank_marketing_prediction.ipynb) — 전처리, 불균형 분류, 임계값·앙상블 탐색
