# Pre-Deployment Application Verification and Testing Report

**Project Title:** Student Early-Warning and Academic Support System  
**Application Target:** [`app.py`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/app.py)  
**Verification Date:** 2026-10-22  
**Target Model:** [`models/best_model.joblib`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/models/best_model.joblib)  
**Evaluator:** ML Engineering Pre-Deployment Audit  

---

## 1. Executive Summary

This report documents the exhaustive verification and pre-deployment stress testing performed on the **Student Early-Warning and Academic Support System** Streamlit application (`app.py`).

### Verification Verdict: ✅ READY FOR DEPLOYMENT
All 12 evaluation criteria were executed and verified:
1. Model loading from disk operates seamlessly via Streamlit's caching mechanism.
2. Predictions across diverse synthetic student profiles align with expectations.
3. All 9 input features are preserved in exact training order without feature dropping or renaming.
4. Prediction probabilities map to the correct class labels and sum to $1.000000$.
5. Boundary conditions, missing values, and unseen categorical levels are handled gracefully without application crashes.
6. Support recommendations for Safe, At-Risk, and High-Risk categories are practical, non-punitive, and actionable.
7. Deprecated Streamlit `use_container_width` warnings have been completely eliminated.
8. The reported test accuracy and confusion matrix metrics were recomputed from raw counts and confirmed to agree with 100% precision.
9. Ethical guidelines and synthetic data disclaimers are prominently displayed.

---

## 2. Test Suite Execution & Detailed Results

### Test 1: Model Loading & Pipeline Integrity
- **Objective:** Verify that `models/best_model.joblib` loads without errors and encapsulates the entire inference pipeline.
- **Result:** **PASSED**
  - Model Type: `sklearn.pipeline.Pipeline`
  - Encapsulated Steps: `['preprocessor', 'classifier']`
  - Encapsulated Estimator: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`
  - Class Order: `['At-Risk', 'High-Risk', 'Safe']`
  - Serialization Size: $5.91\text{ KB}$

### Test 2: Multi-Profile Prediction Accuracy
Five distinct student profiles representing the entire spectrum of academic behavior were evaluated:

| Test Profile | Key Characteristics | Expected Tier | Model Prediction | Class Probabilities (At-Risk / High-Risk / Safe) | Verification |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Profile 1: High Achiever** | 96% attendance, 12h study, 0 failures, 98% assignments, 90% prev grade | Safe | **Safe** | 0.00% / 0.00% / **100.00%** | **PASSED** |
| **Profile 2: On-Track Student** | 88% attendance, 8.5h study, 1 failure, 85% assignments, 80% prev grade | Safe | **Safe** | 0.09% / 0.00% / **99.91%** | **PASSED** |
| **Profile 3: Borderline Slump** | 74% attendance, 5.5h study, 2 failures, 72% assignments, 70% prev grade | At-Risk | **At-Risk** | **99.93%** / 0.07% / 0.00% | **PASSED** |
| **Profile 4: Severe Disengagement** | 48% attendance, 2h study, 4 failures, 45% assignments, 52% prev grade | High-Risk | **High-Risk** | 0.00% / **100.00%** / 0.00% | **PASSED** |
| **Profile 5: Critical Failure Danger** | 20% attendance, 0.5h study, 5 failures, 25% assignments, 30% prev grade | High-Risk | **High-Risk** | 0.00% / **100.00%** / 0.00% | **PASSED** |

### Test 3: Input Feature Schema & Alignment
- **Objective:** Confirm that all 9 features are passed to the model in exact alignment with the training feature schema.
- **Result:** **PASSED**
  - Confirmed column order:
    1. `attendance` (float)
    2. `study_hours` (float)
    3. `past_failures` (int)
    4. `assignments_completed_pct` (float)
    5. `parental_education` (str)
    6. `family_income` (str)
    7. `extracurricular` (str)
    8. `internet_access` (str)
    9. `previous_grade` (float)
  - Raw inputs pass directly into the pipeline with zero manual pre-transformation.

### Test 4: Probability Mapping & Normalization
- **Objective:** Verify that `predict_proba()` values are correctly paired with `model.classes_` and sum to $1.0$.
- **Result:** **PASSED**
  - Class indices strictly mapped: Index $0 \to$ `At-Risk`, Index $1 \to$ `High-Risk`, Index $2 \to$ `Safe`.
  - For all test cases:
    $$\sum_{i} P(C_i) = 1.000000 \pm 10^{-6}$$
  - $\text{argmax}(P)$ corresponds exactly to `model.predict()`.

### Test 5: Boundary Values, Missing Inputs & Edge Cases
- **Objective:** Test edge cases, boundaries, missing values, and unobserved categorical levels.
- **Results:**
  - **All-Zero Boundary ($0$ attendance, $0$ study, $0$ assignments, $0$ grade):** Handled without division-by-zero or numeric overflow. Predicted: **High-Risk** ($100.0\%$).
  - **All-Maximum Boundary ($100\%$ attendance, $25\text{h}$ study, $100\%$ assignments, $100\%$ grade):** Handled cleanly. Predicted: **Safe** ($100.0\%$).
  - **Missing Attendance (`np.nan`):** Automatically imputed using training median ($75.05\%$). Predicted: **At-Risk**.
  - **Missing Study Hours (`None`):** Automatically imputed using training median ($6.00\text{h}$). Predicted: **At-Risk**.
  - **Both Attendance & Study Hours Missing:** Imputed and evaluated gracefully without errors.
  - **Unseen Categorical Levels (`parental_education='Doctorate'`, `family_income='Ultra-High'`):** `OneHotEncoder(handle_unknown='ignore')` gracefully ignores unknown categories without throwing exceptions.
  - **Domain Validation:** Added `validate_student_inputs()` function in `app.py` to prevent negative study hours or attendance $> 100\%$.

### Test 6: Academic Support Recommendations Audit
- **Objective:** Ensure each predicted risk category renders appropriate, supportive, and non-punitive advice.
- **Audit Findings:**
  - **Safe Tier:** Confirms positive engagement; recommends light check-ins, encouragement, and enrichment/mentorship opportunities.
  - **At-Risk Tier:** Suggests actionable, low-friction support: targeted coursework review, departmental tutoring, bi-weekly 15-minute advising, time-management calendar.
  - **High-Risk Tier:** Recommends prompt advisor outreach within 48 hours, root-cause diagnostic session, personalized written Academic Recovery Plan (ARP), 1-on-1 tutoring, and course-load drop-deadline review to protect GPA.
  - Phrasing is supportive, advisor-oriented, and avoids disciplinary language.

---

## 3. Deprecation Warnings Resolution

### Issue Identified:
In Streamlit versions $\ge 1.40$ (current installed version: `1.66.1`), passing `use_container_width=True` to `st.image`, `st.dataframe`, `st.button`, or `st.form_submit_button` triggers the deprecation warning:
> `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31. For use_container_width=True, use width='stretch'.`

### Resolution Applied:
In [`app.py`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/app.py):
Implemented a dynamic compatibility function `get_stretch_kwarg()`:
```python
def get_stretch_kwarg(widget_func):
    sig = inspect.signature(widget_func)
    if "width" in sig.parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}
```
Updated all widget invocations:
- `st.form_submit_button(..., **get_stretch_kwarg(st.form_submit_button))`
- `st.dataframe(..., **get_stretch_kwarg(st.dataframe))`
- `st.button(..., **get_stretch_kwarg(st.button))`
- `st.image(..., **get_stretch_kwarg(st.image))`

**Verification:** Re-running the application produced **0 deprecation warnings**.

---

## 4. Test Metric Recalculation & Artifact Consistency Audit

To fulfill Requirement 9, the reported test performance metrics were recomputed from the serialized confusion matrix in [`reports/test_evaluation.json`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/reports/test_evaluation.json):

```
Matrix Format: Labels = ['Safe', 'At-Risk', 'High-Risk']
[[4632,  367,    3],
 [ 268, 9437,  293],
 [   1,  281, 4718]]
```

### Recalculation Results:
1. **Total Sample Count ($N$):** $4,632 + 367 + 3 + 268 + 9,437 + 293 + 1 + 281 + 4,718 = 20,000$
2. **Correct Predictions (Trace):** $4,632 + 9,437 + 4,718 = 18,787$
3. **Calculated Accuracy:** $\frac{18,787}{20,000} = 0.93935 \implies \mathbf{93.94\%}$
4. **Per-Class Metrics:**
   - **Safe:**
     - Precision: $\frac{4,632}{4,632 + 268 + 1} = \frac{4,632}{4,901} = 0.945113 \implies \mathbf{94.51\%}$
     - Recall: $\frac{4,632}{5,002} = 0.926029 \implies \mathbf{92.60\%}$
     - F1-Score: $2 \times \frac{0.945113 \times 0.926029}{0.945113 + 0.926029} = 0.935474 \implies \mathbf{0.9355}$
   - **At-Risk:**
     - Precision: $\frac{9,437}{367 + 9,437 + 281} = \frac{9,437}{10,085} = 0.935746 \implies \mathbf{93.57\%}$
     - Recall: $\frac{9,437}{9,998} = 0.943888 \implies \mathbf{94.39\%}$
     - F1-Score: $2 \times \frac{0.935746 \times 0.943888}{0.935746 + 0.943888} = 0.939799 \implies \mathbf{0.9398}$
   - **High-Risk:**
     - Precision: $\frac{4,718}{3 + 293 + 4,718} = \frac{4,718}{5,014} = 0.940965 \implies \mathbf{94.10\%}$
     - Recall: $\frac{4,718}{5,000} = 0.943600 \implies \mathbf{94.36\%}$
     - F1-Score: $2 \times \frac{0.940965 \times 0.943600}{0.940965 + 0.943600} = 0.942281 \implies \mathbf{0.9423}$
5. **Macro Averages:**
   - Macro Precision: $\frac{0.945113 + 0.935746 + 0.940965}{3} = \mathbf{0.9406}$
   - Macro Recall (Balanced Accuracy): $\frac{0.926029 + 0.943888 + 0.943600}{3} = \mathbf{0.9378}$
   - Macro F1-Score: $\frac{0.935474 + 0.939799 + 0.942281}{3} = \mathbf{0.9392}$

**Audit Finding:** All metrics in [`reports/model_comparison.md`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/reports/model_comparison.md) and [`reports/test_evaluation.json`](file:///c:/Users/Sneha/OneDrive/Desktop/SNEHA/Study/sem%205/ML/Student_Risk_Prediction_Project/reports/test_evaluation.json) match the exact confusion matrix recalculation with zero discrepancy.

---

## 5. UI Appearance, Responsiveness, and Error Handling

1. **Responsive Layout:**
   - Implemented via `layout="wide"`, responsive Streamlit columns, and collapsible sections.
   - Presets tab allows one-click loading and testing of realistic student archetypes.
2. **Error Handling:**
   - Missing model file check with graceful user error message.
   - Numerical boundary validation via `validate_student_inputs()` catches invalid out-of-range values before model invocation.
3. **Privacy & Ethical Safeguards:**
   - The interface collects strictly behavioral and academic engagement inputs.
   - No names, student IDs, emails, or personal identifiers are requested or logged.
   - Clear disclaimers appear in the sidebar, header, and recommendation footnotes indicating that predictions are based on synthetic data.

---

## 6. Pre-Deployment Checklist

| Verification Item | Status | Verified Details |
| :--- | :---: | :--- |
| Model pipeline load (`models/best_model.joblib`) | ✅ Pass | Loaded via `@st.cache_resource`, sub-millisecond inference. |
| 9-feature schema alignment | ✅ Pass | Exact column names and order preserved. |
| Single-student form prediction | ✅ Pass | Produces category, probability table, and bar chart. |
| Multi-profile test cases | ✅ Pass | 5/5 test profiles matched expected classifications. |
| Probability sum check ($\sum P = 1.0$) | ✅ Pass | Verified for all profiles and boundary cases. |
| Boundary value stress test | ✅ Pass | Handled without division-by-zero or crash. |
| Missing value handling | ✅ Pass | Imputed seamlessly via embedded `SimpleImputer`. |
| Unseen categorical levels | ✅ Pass | Ignored safely via `OneHotEncoder(handle_unknown='ignore')`. |
| Support recommendations | ✅ Pass | Tailored, non-punitive, and actionable across all 3 tiers. |
| Streamlit deprecation warnings | ✅ Pass | `use_container_width` replaced with `width="stretch"`. |
| Metric consistency audit | ✅ Pass | Confusion matrix recalculation matches reports exactly. |
| Synthetic data disclaimers | ✅ Pass | Prominently displayed across sidebar and app tabs. |
| Clean imports & zero syntax errors | ✅ Pass | Verified via bare Python import test. |

---

## 7. Deployment Readiness Recommendation

### Final Verdict: ✅ READY FOR DEPLOYMENT

The application is stable, robust against anomalous and edge inputs, mathematically verified, free of deprecation warnings, and compliant with privacy and ethical standards. It is ready for local presentation or cloud container deployment (e.g., Streamlit Community Cloud or Docker).
