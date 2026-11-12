# Real Estate Investment | 7개 연도 패널형 가구 데이터 통합과 XAI

> **프로젝트 핵심:** 2019–2025년 가구자료를 통합해 투자 의사결정을 분류하고, **XGBoost + SHAP**으로 예측 근거를 해석했습니다.

## 문제 정의 및 데이터
가구주의 인구·주거·재무 특성이 `investment_decision`과 어떤 관계에 있는지 탐색했습니다. 2019~2025년 연도별 데이터를 불러와, **서로 다른 변수명을 공통 변수 체계로 재매핑**한 뒤 분석용 데이터를 구축했습니다.

주요 변수: `head_age`, `own_house`, `year`, 자산·부채·소득·지출, 가구 구성, 대출 상환 관련 변수 등.

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

## Explainable AI

![Feature importance](./images/feature_importance.png)

![SHAP summary](./images/shap_summary.png)

- `head_age`, `own_house`, `year` 등이 중요한 변수로 나타났습니다.
- Feature Importance가 **무엇이 중요한지** 보여준다면, SHAP은 **개별 관측치의 예측값에 어떤 방향으로 작용하는지** 해석하도록 도와줍니다.
- 연도 변수가 중요하다고 해서 **금리 변화의 인과효과를 식별했다**고 주장할 수는 없습니다. 연도는 같은 기간의 다양한 외부 변화를 포괄할 수 있습니다.

## 분석의 가치와 한계
- **데이터 통합:** 다년도 원천 테이블의 변수 체계를 표준화
- **해석 가능성:** 예측 성능과 변수 기여도를 함께 제시
- **주의:** SHAP은 모델의 예측 기여도이며 인과 추정은 아닙니다. 시계열 외삽 성능은 별도 연도 홀드아웃 검증이 필요합니다.

## Files
- [`01_Data_Analysis.ipynb`](./01_Data_Analysis.ipynb): 연도별 정제·통합 및 EDA
- [`02_Final_XGBoost_Model.ipynb`](./02_Final_XGBoost_Model.ipynb): XGBoost 성능 검증, 중요도/SHAP 시각화

**Stack** `Python` `Pandas` `XGBoost` `SHAP` `Scikit-learn` `Matplotlib` `Seaborn`
