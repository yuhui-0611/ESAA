# Bank Marketing Hackathon | 고객 세분화 및 정기예금 가입 예측

> **핵심:** PCA·K-Means 기반 고객 세분화 → 금융 행동 파생변수 설계 → 불균형 이진 분류 및 가중 하드 보팅으로 정기예금 가입 여부 예측

## 문제 정의 및 데이터

| 항목 | 내용 |
|---|---|
| **목표** | 은행 고객의 정기예금 가입 여부(`TermDepositSubscribed`) 예측 |
| **데이터** | 학습 70,000건(45개 컬럼), 테스트 30,000건(44개 컬럼) |
| **문제 유형** | 불균형 이진 분류 + 비지도학습 기반 고객 세분화 |
| **핵심 기법** | KNN 기반 결측치 대체, Feature Engineering, PCA, K-Means, Random Forest, LightGBM, XGBoost, CatBoost |
| **모델 최적화** | Stratified K-Fold, RandomizedSearchCV, Optuna, Early Stopping, F1 기반 임계값 탐색 |
| **앙상블** | 다중 모델 예측값의 가중 하드 보팅(Weighted Hard Voting) |

**분석 방향:** 정기예금 가입 예측에 고객 군집 특성과 금융 활동·디지털 접점·자산 수준을 반영한 파생변수 결합

## 데이터 전처리 및 변수 설계

### 결측치·이상치 처리

- **이웃 기반 결측치 대체:** `NearestNeighbors`를 이용해 유사 관측치의 중앙값·최빈값으로 수치형 및 범주형 결측치 대체 (`k=5`)
- **범주형 변수 정제:** 직업명 표기 통일, 비정상 지역 값(`Mars`, `Atlantis`, `Area-51`) 별도 범주 처리
- **변수 인코딩:** 고객 세그먼트·교육 수준·소득 구간 순서형 매핑 및 직업·지역·연락 채널 원-핫 인코딩
- **수치 분포 보정:** 금융 거래·투자 관련 변수 로그 변환, 거래 횟수 상위 1% 클리핑, 지점 방문 빈도 구간화
- **학습·테스트 특성 정렬:** 인코딩 이후 열 구조 일치 및 수치형 입력 구성

### 금융 행동 기반 파생변수

| 변수 영역 | 주요 파생변수 | 설계 내용 |
|---|---|---|
| 고객 활동성 | `Total_Activity_Score` | 전체 거래·모바일 앱 로그인·지점 방문 빈도 결합 |
| 디지털 이용 | `Digital_Usage_Ratio`, `Web_Engagement_Index` | 온라인 거래 및 앱·웹 접점의 상대적 활용도 구성 |
| 자산·신용 | `Young_High_Credit_Score`, `Wealth_Accumulation_Ratio` | 연령·소득·신용점수 및 자산 축적 수준 조합 |
| 은행 관계 | `Transaction_per_Tenure`, `Product_Density` | 이용 기간 대비 거래량·상품 보유 수준 산출 |
| 마케팅 반응 | `Marketing_Efficiency`, `Contact_Intensity`, `Contact_Gap_Risk` | 마케팅 점수·반응 성향·접촉 이력을 조합한 특성 생성 |
| 고객군 연계 | `C1_Pure_Dormant`, `C1_Asset_Holders`, `C3_Future_VIP` | 군집 라벨과 활동·자산·반응 지표를 결합한 이진 특성 생성 |

## PCA·K-Means 기반 고객 세분화

### 군집화 설계

1. 연령, 연소득, 신용점수, 은행 이용 기간, 거래량, 마케팅 반응 등 고객 특성 16개 선택
2. 연속형 변수 `StandardScaler` 표준화 및 범주형 변수 인코딩
3. **PCA로 2차원 축소** 후 `KMeans(n_clusters=4, random_state=42, n_init=10)` 적용
4. 학습 데이터에서 군집 학습 후 테스트 데이터에 동일한 PCA·K-Means 변환 적용
5. 군집별 평균 금융 지표를 정규화한 히트맵으로 고객군 특성 비교

### 군집별 주요 특징

| 군집 | 고객 유형 | 관찰된 특징 |
|---|---|---|
| **Cluster 0** | 고가치·고활동 고객 | 소득·순자산·신용점수 등 금융 지표 상위, 높은 거래 및 마케팅 반응 수준 |
| **Cluster 1** | 저활동·신규 고객 | 금융 거래·활동 지표 전반이 낮은 고객군 |
| **Cluster 2** | 장기 거래·고연령 고객 | 상대적으로 높은 연령과 은행 이용 기간, 활발한 거래·웹 활동 |
| **Cluster 3** | 디지털 잠재 고객 | 상대적으로 낮은 연령·짧은 거래 기간, 양호한 소득·신용 및 디지털 활동 수준 |

**분석 결과:** 고객의 자산 수준뿐 아니라 은행 이용 기간·채널 이용 패턴·마케팅 반응을 기준으로 서로 다른 고객군의 특징 확인. 군집 라벨과 고객 행동 조건을 조합해 예측용 추가 특성 구성

## 모델링 및 최적화 실험

### 모델별 실험

| 모델 | 주요 실험 |
|---|---|
| **Random Forest** | `class_weight='balanced'` 적용, 혼동행렬·분류 리포트·F1 평가, `RandomizedSearchCV`로 파라미터 탐색 |
| **LightGBM** | `scale_pos_weight` 포함한 Optuna 탐색, 3-Fold 검증 기반 목적함수 구성 및 5-Fold OOF 예측 코드 구성 |
| **XGBoost** | 8:2 계층화 홀드아웃, Early Stopping, F1·ROC-AUC 평가, 예측 확률 임계값 탐색 |
| **CatBoost** | 클래스 비율 기반 가중치 적용, 1,000회 학습 설정 및 예측 결과 생성 |

### 불균형 데이터와 분류 임계값 실험

- **불균형 대응:** Random Forest의 `class_weight`, LightGBM의 `scale_pos_weight`, CatBoost의 `class_weights` 활용
- **평가 지표:** 정기예금 가입 클래스에 대한 F1 중심 평가, XGBoost 실험에서 ROC-AUC·Classification Report 추가 확인
- **임계값 탐색:** 기본 분류 기준 `0.5` 외 여러 Threshold의 F1을 비교하여 분류 기준 탐색
- **하이퍼파라미터 탐색:** Random Forest의 무작위 탐색과 LightGBM의 Optuna 기반 최적화 실험 코드 구성

### 최종 예측 결합

- **다중 모델 결과 결합:** CatBoost, Random Forest, LightGBM, XGBoost의 이진 예측 결과를 `ClientID` 기준으로 병합
- **가중 하드 보팅:** 모델 가중치 `CatBoost : Random Forest : LightGBM : XGBoost = 1 : 1 : 1 : 2` 적용
- **최종 제출값 구성:** 가중 투표 합산값으로 가입 여부 결정 및 제출용 CSV 생성

## 주요 분석 결과

- **고객군 구분:** PCA·K-Means와 군집별 평균 지표 분석으로 고가치 고객, 저활동 고객, 장기 거래 고객, 디지털 잠재 고객의 특성 구분
- **고객 행동 특성 확장:** 단일 금융 변수 외에 이용 강도, 디지털 활동, 자산·신용 조합, 마케팅 접촉 강도를 반영한 파생변수 구성
- **불균형 분류 실험:** 클래스 가중치, F1 중심 검증, 예측 확률 임계값 조정을 통한 분류 모델 비교
- **앙상블 설계:** 서로 다른 학습 알고리즘의 이진 예측을 결합하는 가중 하드 보팅 구현

## Files

- [`bank_marketing_prediction.ipynb`](./bank_marketing_prediction.ipynb) — 전처리, PCA·K-Means 군집화, 파생변수 생성, 분류 모델링, 하이퍼파라미터 탐색 및 앙상블

**Stack**  
`Python` `Pandas` `NumPy` `Scikit-learn` `PCA` `K-Means` `Random Forest` `LightGBM` `XGBoost` `CatBoost` `Optuna` `Seaborn`
