# Hotel Reservation Cancellation | 다중 분류기 비교 및 하이퍼파라미터 탐색

> **프로젝트 핵심:** 호텔 예약 취소의 위험을 예측하기 위해 **선형 모델 → 트리 기반 앙상블 → Optuna 탐색 → 확률 및 변수 중요도 비교**로 분석을 확장했습니다.

## 문제 정의
예약 시점에 파악 가능한 고객·숙박 정보를 바탕으로 예약 취소(`Canceled` / `Not_Canceled`) 가능성을 추정합니다. 취소율은 수익·운영에 영향을 주므로, 단순 정확도보다 **정밀도·재현율·F1·ROC-AUC**를 함께 살펴보았습니다.

## 데이터 및 전처리
- 주요 특성: lead time, 숙박 일수, 객실 유형, 객실 가격, 시장 세그먼트, 특별 요청 수, 고객 예약 이력 등
- EDA 후 수치형/범주형 변수 구분, 인코딩 및 모델별 입력 구성
- 상관계수/VIF 검토 및 변수 중요도 기반 축소 실험

## 모델 비교 및 최적화
- **Logistic Regression:** 해석 가능한 선형 기준 모델
- **Random Forest:** 비선형 상호작용을 포착하는 배깅 모델
- **XGBoost / LightGBM / CatBoost:** 부스팅 기반 분류기 비교
- **Optuna:** LightGBM·XGBoost·Random Forest 등의 후보 하이퍼파라미터 탐색
- 추가로 일부 모델의 **Voting 방식 결합**을 실험했습니다.

## 평가 설계
Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Classification Report를 확인했습니다. 모델의 성능 차이와 과적합 가능성을 비교하고, 단일 점수보다는 **취소 고객 탐지 목적에 적합한 지표 조합**을 고려했습니다.

## 포트폴리오 관점에서의 의미
- 범주형과 연속형 특성이 혼재된 현실 데이터를 다양한 모델 입력으로 변환
- Optuna 기반 자동 튜닝 및 Feature Selection
- 분류 성능 및 오류 유형을 함께 비교

> 본 프로젝트는 **머신러닝 앙상블 최적화** 사례입니다. 

## File
- [`hotel_reservation_prediction.ipynb`](./hotel_reservation_prediction.ipynb): 데이터 분석·전처리·모델 비교 및 탐색

**Stack** `Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `Optuna`
