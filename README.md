# Student Early-Warning and Academic Support System 🎓

An end-to-end Machine Learning decision-support application that identifies early indicators of student academic distress and recommends timely, non-punitive support interventions — built with scikit-learn and Streamlit.

**Live Demo:** *(add your Streamlit Community Cloud URL here after deployment)*

---

## 📌 Project Overview

This project demonstrates the full ML lifecycle for an academic early-warning system across four phases:

| Phase | Description |
|:------|:------------|
| **Phase 1 — Data Audit** | Dataset inspection, data-type validation, missingness analysis, and statistical proof of synthetic origin |
| **Phase 2 — Preprocessing & EDA** | Leakage prevention (removal of `final_score`), stratified 80/20 train-test split, scikit-learn `ColumnTransformer` pipeline, distribution and correlation visualisations |
| **Phase 3 — Training & Selection** | 5-fold Stratified Cross-Validation across Logistic Regression, Decision Tree, Random Forest, and XGBoost; selection on Macro-F1 and High-Risk Recall; single-pass evaluation on 20,000-row held-out test set |
| **Phase 4 — Web Application** | Interactive Streamlit dashboard with single-student risk assessment, preset archetype explorer, and model diagnostic panel |

---

## 🤖 Model Details

### Champion Model: Multinomial Logistic Regression

The deployed model is a scikit-learn `Pipeline` coupling a `ColumnTransformer` preprocessor with Logistic Regression:

1. **Median Imputation** — handles occasional missing values in attendance and study hours (MCAR mechanism)
2. **Z-score Standardisation** — `StandardScaler` on all five continuous features
3. **One-Hot Encoding** — `OneHotEncoder(handle_unknown='ignore')` expands the four categorical features to 16 binary columns
4. **Classifier** — `LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, multi_class='auto')`

### Why Logistic Regression outperformed tree ensembles

The synthetic target `risk_category` was derived by partitioning an underlying **linear function** of student engagement and academic history. Logistic Regression, which learns linear decision boundaries, represents this structure exactly — while Decision Tree, Random Forest, and XGBoost approximate it with recursive orthogonal splits, introducing discretisation error. All input features were statistically independent (Pearson *r* ≈ −0.003 to +0.005), matching the i.i.d. generation mechanism of the synthetic dataset.

### 5-Fold Stratified Cross-Validation Results (Training set, N = 80,000)

| Algorithm | Macro F1 | High-Risk Recall | Balanced Accuracy | Overall Accuracy |
|:----------|:--------:|:----------------:|:-----------------:|:----------------:|
| **Logistic Regression ✅** | **0.9361 ± 0.0020** | **0.9357 ± 0.0034** | **0.9347 ± 0.0021** | **0.9362 ± 0.0020** |
| XGBoost | 0.9239 ± 0.0033 | 0.8984 ± 0.0041 | 0.9157 ± 0.0038 | 0.9247 ± 0.0033 |
| Random Forest | 0.9048 ± 0.0030 | 0.8580 ± 0.0039 | 0.8902 ± 0.0034 | 0.9066 ± 0.0029 |
| Decision Tree | 0.8689 ± 0.0020 | 0.8547 ± 0.0042 | 0.8650 ± 0.0020 | 0.8695 ± 0.0020 |

### Final Test-Set Evaluation (Held-Out, N = 20,000)

| Metric | Value |
|:-------|------:|
| Overall Accuracy | **93.94%** (18,787 / 20,000) |
| Macro F1-Score | **93.92%** |
| Balanced Accuracy | **93.78%** |
| High-Risk Recall | **94.36%** (4,718 / 5,000) |
| Catastrophic Errors (High-Risk → Safe) | **0.02%** (1 / 5,000) |
| Vulnerability Capture (At-Risk or High-Risk) | **99.98%** (4,999 / 5,000) |

---

## ⚠️ Synthetic Data Limitations

> **This model was trained exclusively on synthetically generated data.**

Key limitations to understand before any real-world consideration:

- **Artificially clean distributions** — the synthetic generator created independent, near-perfectly normal features. Real student data contains multi-collinear, non-normal, and temporally autocorrelated signals.
- **No real-world validation** — performance metrics (93.94% accuracy) apply only to this synthetic test set and cannot be extrapolated to any actual student population.
- **Missing confounders** — variables such as mental health, disability status, institutional support quality, and socioeconomic complexity are absent.
- **No PII processed** — the application accepts no student names, IDs, or personally identifying information by design.

**This prototype must NOT be used for real administrative, disciplinary, or placement decisions without full institutional validation on real, ethically-collected student data.**

---

## 🏗️ Repository Structure

```
Student_Risk_Prediction_Project/
├── app.py                              # Streamlit application (entry point)
├── requirements.txt                    # Runtime dependencies
├── README.md                           # This file
├── .gitignore                          # Excludes datasets, large models, secrets, cache
├── models/
│   └── best_model.joblib               # Deployed champion pipeline (Logistic Regression, ~6 KB)
├── src/
│   ├── models_def.py                   # Pipeline builders & StringTargetXGBClassifier wrapper
│   └── preprocessing.py               # ColumnTransformer builder & standardize_categorical_casing
└── reports/
    └── figures/
        ├── model_comparison.png        # 5-fold CV bar chart (displayed in app Tab 3)
        └── confusion_matrix.png        # Test set confusion matrix (displayed in app Tab 3)
```

> **Not included in this repository** (excluded by `.gitignore`):
> - `data/student_data.csv` — 100,000-row raw synthetic dataset (~5.6 MB)
> - `data/processed/` — all intermediate CSV splits (~24 MB total)
> - `models/random_forest.joblib` — 43.5 MB (not deployed)
> - `models/decision_tree.joblib`, `xgboost.joblib`, `logistic_regression.joblib`, `preprocessor.joblib` — non-deployed artefacts
> - `.venv/`, `__pycache__/`, `.env` — local environment and cache files

---

## 💻 Local Run Instructions

### Prerequisites
- Python 3.10 – 3.14
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Student_Risk_Prediction_Project.git
cd Student_Risk_Prediction_Project
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

1. **Push this repository to GitHub** (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repository, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **"Deploy"**. Streamlit Cloud will automatically install `requirements.txt` and launch the app.

> **No secrets or environment variables are required** — the app reads only the bundled `models/best_model.joblib` file using a relative path.

---

## 🔒 Privacy & Security

- No passwords, API keys, or credentials exist in this project.
- No student names, IDs, or personally identifying information (PII) are requested, stored, or processed by the application.
- The `.gitignore` excludes `.env`, `secrets.toml`, and `.streamlit/secrets.toml` to prevent accidental credential commits.

---

## ⚖️ Academic Disclaimer

This project is an academic ML prototype submitted as coursework. The model achieves high statistical fidelity on the synthetic test set but **does not represent real-world clinical or institutional validity**. All predictions are demonstrations of early-warning system mechanics and must not be used for real student administrative decisions.
