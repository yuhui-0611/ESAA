# Hotel Reservation Cancellation | 호텔 예약 취소 예측

**예약 고객 정보와 과거 이용 이력을 활용한 취소 여부 분류 및 분류기별 예측 성능 비교**

## 문제 정의
- **예측 대상:** 호텔 예약 취소 여부 (`Canceled` / `Not_Canceled`)
- **주요 변수:** 예약 리드타임, 숙박 일수, 객실 가격, 특별 요청, 고객·예약 이력
- **주요 기술:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Optuna

## 주요 구현
- **전처리 및 특성 검토:** 범주형 인코딩, 변수 간 상관관계·VIF 검토, 특성 제거 전후 비교
- **분류 모델 비교:** 로지스틱 회귀 기준 모델과 Random Forest·부스팅 계열 분류기 학습
- **하이퍼파라미터 탐색:** Optuna 및 교차검증을 활용한 모델별 설정 탐색
- **앙상블 실험:** VotingClassifier를 이용한 복수 분류기 결합
- **오류 분석:** Accuracy·Precision·Recall·F1·ROC-AUC와 혼동행렬을 함께 확인

## 분석 결과
- 모델 종류, 특성 축소 및 하이퍼파라미터 조정에 따른 예측 결과 비교
- 클래스별 오분류 유형과 재현율·정밀도 차이 확인

## 코드
[hotel_reservation_prediction.ipynb](./hotel_reservation_prediction.ipynb) — EDA, 특성 선택, 모델 학습, Optuna, Voting 및 성능 평가
