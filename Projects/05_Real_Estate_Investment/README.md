# Real Estate Investment | 7개 연도 가구 조사 데이터 통합과 SHAP 해석

> **핵심:** 2019–2025년 가구 조사 데이터의 변수 체계 통합 → XGBoost 투자 의향 분류 → SHAP 기반 주요 예측 요인 분석

## 문제 정의 및 데이터
- 가구주의 인구·주거·재무 특성이 `investment_decision`과 어떤 관계에 있는지 탐색함
- 2019~2025년 연도별 데이터를 불러와, **서로 다른 변수명을 공통 변수 체계로 재매핑**한 뒤 분석용 데이터를 구축함

주요 변수: `head_age`, `own_house`, `year`, 자산·부채·소득·지출, 가구 구성, 대출 상환 관련 변수 등

## 분석 및 모델링 과정
1. 연도별 파일 로드 → 공통 변수 선정 → 컬럼명 표준화 → 연도 변수 추가 → 병합
2. 데이터형·결측 및 코드형 변수 확인, 더미변수 생성, 이상치·상관관계 검토
3. Train/Test 분리 후 **XGBClassifier** 학습
4. Accuracy·Precision·Recall·F1·ROC-AUC 및 혼동행렬 평가
5. **Feature Importance(gain)와 SHAP TreeExplainer**를 이용한 전역/방향성 해석

## 저장된 실험 성능

| Metric | Score |
|---|---:|
| Accuracy | 0.6830 |
| Precision | 0.6577 |
| Recall | 0.7074 |
| F1-score | 0.6816 |
| ROC-AUC | 0.7443 |

**평가 기준:** 전체 통합 데이터의 20%를 층화 무작위 홀드아웃으로 분리해 평가

## SHAP 및 주요 분석 결과

![Feature importance](./images/feature_importance.png)

![SHAP summary](./images/shap_summary.png)

- **Feature Importance:** XGBoost 분할에 활용된 변수의 상대적 중요도 분석
- **SHAP:** 변수별 예측 기여도와 방향 시각화. `head_age`, `own_house`, `year` 등의 중요 변수 확인
- **다년도 통합:** 연도마다 다른 변수명을 공통 스키마로 정규화해 일관된 입력 특징 구성
- **분석 종합:** 연도별로 분리된 가구 조사 자료를 통합하고, 투자 의향 예측값과 주요 특성의 기여 방향을 함께 확인

## Files
- [`01_Data_Analysis.ipynb`](./01_Data_Analysis.ipynb): 연도별 정제·통합 및 EDA
- [`02_Final_XGBoost_Model.ipynb`](./02_Final_XGBoost_Model.ipynb): XGBoost 성능 검증, 중요도/SHAP 시각화

**Stack** `Python` `Pandas` `XGBoost` `SHAP` `Scikit-learn` `Matplotlib` `Seaborn`
