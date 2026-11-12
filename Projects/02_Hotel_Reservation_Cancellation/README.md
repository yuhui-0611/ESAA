# Hotel Reservation Cancellation | 다중 분류기 비교 및 하이퍼파라미터 탐색

> **핵심:** 호텔 예약 취소 분류. 로지스틱 회귀 기준선부터 트리 앙상블, Optuna 튜닝, 변수 축소 및 Voting까지 단계적 비교.

## 문제 정의
예약 고객·숙박 정보를 바탕으로 예약 취소(`Canceled` / `Not_Canceled`) 가능성 추정. 예약 취소 탐지의 정밀도·재현율 간 차이를 고려해 **F1·ROC-AUC 및 혼동행렬** 평가.

## 데이터 및 전처리
- 주요 특성: lead time, 숙박 일수, 객실 유형, 객실 가격, 시장 세그먼트, 특별 요청 수, 고객 예약 이력 등
- EDA 후 수치형/범주형 변수 구분, 인코딩 및 모델별 입력 구성
- 상관계수/VIF 검토 및 변수 중요도 기반 축소 실험

## 모델 비교 및 최적화
- **Logistic Regression:** 해석 가능한 선형 기준 모델
- **Random Forest:** 비선형 상호작용을 포착하는 배깅 모델
- **XGBoost / LightGBM / CatBoost:** 부스팅 기반 분류기 비교
- **Optuna:** LightGBM·XGBoost·Random Forest 등의 후보 하이퍼파라미터 탐색
- 추가로 일부 모델의 **Voting 방식 결합**을 실험함.

## 평가 설계
- **분류 성능:** Accuracy·Precision·Recall·F1·ROC-AUC 및 혼동행렬 비교
- **오류 유형:** 예약 취소 고객의 미탐지(False Negative)와 정상 예약 고객의 오탐지(False Positive) 구분
- **모델 선택 기준:** F1과 ROC-AUC를 함께 확인하고 변수 축소·튜닝에 따른 성능 변화 점검

## 노트북 기록 성능

| 모델/실험 | F1 | ROC-AUC | 해석 |
|---|---:|---:|---|
| CatBoost 기본 실험 | **0.9357** | **0.9596** | 저장된 `y_test` 평가 출력 |
| Random Forest 후속 실험 | **0.9361** | **0.9598** | 특성·설정이 다른 별도 실험 출력 |

- **측정 조건:** 노트북 내 `y_test` 분할에서 기록한 수치. 각 모델은 별도 설정으로 실험.
- **오류 분석:** Accuracy 단독 평가 대신 Precision·Recall·F1과 혼동행렬을 병행. 취소 탐지의 누락과 오탐을 구분해 평가.
- **실험 결과:** 선형 모델에서 트리 앙상블로 모델 범위 확장. Optuna 하이퍼파라미터 탐색과 변수 축소 전후의 성능 차이 검토.

## File
- [`hotel_reservation_prediction.ipynb`](./hotel_reservation_prediction.ipynb): 데이터 분석·전처리·모델 비교 및 탐색

**Stack** `Python` `Pandas` `Scikit-learn` `XGBoost` `LightGBM` `CatBoost` `Optuna`
