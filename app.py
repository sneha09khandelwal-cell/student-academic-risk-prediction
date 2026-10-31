"""
Student Early-Warning and Academic Support System.

Interactive Streamlit Application for Academic Risk Prediction and Support Planning.
Loads the trained end-to-end model (models/best_model.joblib) and evaluates raw student
inputs without double-transformation.
"""

import inspect
import os
import sys

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Required for unpickling XGBoost or custom transformers if needed
from src.models_def import StringTargetXGBClassifier
from src.preprocessing import standardize_categorical_casing

MODEL_PATH = "models/best_model.joblib"

RAW_COLUMNS = [
    "attendance",
    "study_hours",
    "past_failures",
    "assignments_completed_pct",
    "parental_education",
    "family_income",
    "extracurricular",
    "internet_access",
    "previous_grade",
]

RISK_COLORS = {
    "Safe": "#2ecc71",       # Emerald Green
    "At-Risk": "#f39c12",    # Amber
    "High-Risk": "#e74c3c",  # Crimson
}

RISK_ICONS = {
    "Safe": "🟢",
    "At-Risk": "🟡",
    "High-Risk": "🔴",
}


def get_stretch_kwarg(widget_func):
    """
    Return width='stretch' for modern Streamlit (>=1.40/1.66) to avoid
    the deprecated use_container_width warning, while falling back gracefully.
    """
    sig = inspect.signature(widget_func)
    if "width" in sig.parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


@st.cache_resource
def load_trained_model(model_path: str = MODEL_PATH):
    """Load the serialized end-to-end best model pipeline."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please run `src/train_models.py` first.")
        st.stop()
    return joblib.load(model_path)


def validate_student_inputs(inputs: dict) -> list:
    """Validate numerical inputs against reasonable academic domain boundaries."""
    errors = []
    if inputs["attendance"] < 0.0 or inputs["attendance"] > 100.0:
        errors.append("Lecture Attendance must be between 0.0% and 100.0%.")
    if inputs["study_hours"] < 0.0 or inputs["study_hours"] > 60.0:
        errors.append("Weekly Study Hours must be between 0.0 and 60.0 hours.")
    if inputs["past_failures"] < 0 or inputs["past_failures"] > 20:
        errors.append("Past Course Failures cannot be negative or exceed 20.")
    if inputs["assignments_completed_pct"] < 0.0 or inputs["assignments_completed_pct"] > 100.0:
        errors.append("Coursework Assignments Completed must be between 0.0% and 100.0%.")
    if inputs["previous_grade"] < 0.0 or inputs["previous_grade"] > 100.0:
        errors.append("Previous Semester Grade must be between 0.0% and 100.0%.")
    return errors


def main():
    st.set_page_config(
        page_title="Student Early-Warning & Academic Support System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header section
    st.title("🎓 Student Early-Warning & Academic Support System")
    st.markdown(
        "A proactive decision-support tool designed for academic advisors to identify early signs "
        "of academic risk and coordinate timely, student-centered interventions."
    )

    # Model Load
    model = load_trained_model(MODEL_PATH)

    # Sidebar - Information & Disclaimers
    with st.sidebar:
        st.header("📌 System Overview")
        st.info(
            "**Primary Model:** Multinomial Logistic Regression\n\n"
            "**Verified Test Accuracy:** **93.94%** ($18,787 / 20,000$)\n\n"
            "**High-Risk Recall:** **94.36%** ($4,718 / 5,000$)\n\n"
            "**Intervention Capture:** **99.98%** (High-Risk or At-Risk)\n\n"
            "**Preprocessing:** Embedded ColumnTransformer (automatic median imputation, standard scaling, and one-hot encoding)."
        )

        st.markdown("---")
        st.subheader("⚠️ Prototype Disclaimer")
        st.warning(
            "This application is an academic ML prototype trained strictly on **synthetically generated data**. "
            "Predictions are demonstrations of machine learning early-warning principles and must **NOT** be used "
            "for real student administrative decisions, course placement, or disciplinary actions without institutional validation."
        )

        st.markdown("---")
        st.subheader("🔒 Privacy & Ethics")
        st.caption(
            "This system adheres to privacy-by-design principles: **no student names, IDs, or personally identifying "
            "information (PII)** are requested, stored, or processed."
        )

    # Main content tabs
    tab_predict, tab_batch, tab_about = st.tabs([
        "🔍 Single-Student Risk Assessment",
        "📂 Preset Profiles & Testing",
        "📊 Model Architecture & Inspection",
    ])

    # -------------------------------------------------------------
    # TAB 1: Single Student Prediction
    # -------------------------------------------------------------
    with tab_predict:
        st.subheader("Student Academic & Behavioral Profile")
        st.caption("Enter the student's early-semester engagement and prior academic record below:")

        with st.form("risk_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 📚 Engagement Indicators")
                attendance = st.slider(
                    "Lecture Attendance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=0.5,
                    help="Early attendance percentage observed across lectures and labs.",
                )
                study_hours = st.slider(
                    "Weekly Study Hours",
                    min_value=0.0,
                    max_value=25.0,
                    value=6.0,
                    step=0.5,
                    help="Self-reported or LMS-logged weekly study hours outside scheduled classes.",
                )
                assignments_completed_pct = st.slider(
                    "Coursework Assignments Completed (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    help="Percentage of submitted coursework, quizzes, and problem sets to date.",
                )

            with col2:
                st.markdown("##### 📈 Academic History")
                previous_grade = st.slider(
                    "Previous Semester Grade (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=70.0,
                    step=0.5,
                    help="Cumulative academic score from prior semester or qualifying baseline exam.",
                )
                past_failures = st.number_input(
                    "Past Course Failures",
                    min_value=0,
                    max_value=10,
                    value=1,
                    step=1,
                    help="Number of courses or credit units previously failed.",
                )
                extracurricular = st.selectbox(
                    "Extracurricular Participation",
                    options=["No", "Yes"],
                    index=0,
                    help="Involvement in sports, academic clubs, or campus organizations.",
                )

            with col3:
                st.markdown("##### 🌐 Context & Background")
                parental_education = st.selectbox(
                    "Parental Education Level",
                    options=["Secondary", "Higher", "Primary", "None"],
                    index=0,
                    help="Highest formal education level completed by parents/guardians.",
                )
                family_income = st.selectbox(
                    "Family Income Bracket",
                    options=["Low", "Medium", "High"],
                    index=0,
                    help="Socioeconomic household income bracket.",
                )
                internet_access = st.selectbox(
                    "Reliable Internet Access",
                    options=["Yes", "No"],
                    index=0,
                    help="Availability of reliable home internet access for study materials.",
                )

            submit_button = st.form_submit_button(
                "🚀 Predict Academic Risk Tier",
                type="primary",
                **get_stretch_kwarg(st.form_submit_button),
            )

        if submit_button:
            input_dict = {
                "attendance": float(attendance),
                "study_hours": float(study_hours),
                "past_failures": int(past_failures),
                "assignments_completed_pct": float(assignments_completed_pct),
                "parental_education": str(parental_education),
                "family_income": str(family_income),
                "extracurricular": str(extracurricular),
                "internet_access": str(internet_access),
                "previous_grade": float(previous_grade),
            }

            # Input validation check
            validation_errors = validate_student_inputs(input_dict)
            if validation_errors:
                for err in validation_errors:
                    st.error(f"⚠️ Validation Error: {err}")
                st.stop()

            # Construct raw DataFrame with exact schema
            raw_input_df = pd.DataFrame([input_dict])[RAW_COLUMNS]

            # Execute Model Prediction
            try:
                prediction = model.predict(raw_input_df)[0]
                probabilities = model.predict_proba(raw_input_df)[0]
                classes = list(model.classes_)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

            # Display Results
            st.markdown("---")
            st.subheader("🎯 Risk Assessment & Supportive Action Plan")

            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                icon = RISK_ICONS.get(prediction, "⚪")
                color = RISK_COLORS.get(prediction, "#333333")

                st.markdown(
                    f"""
                    <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
                        <h4 style="margin: 0; color: #555;">Predicted Risk Category</h4>
                        <h2 style="margin: 5px 0 0 0; color: {color};">{icon} {prediction}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probability table
                prob_df = pd.DataFrame({
                    "Risk Category": classes,
                    "Confidence Probability": [f"{p*100:.2f}%" for p in probabilities],
                })
                st.dataframe(prob_df, hide_index=True, **get_stretch_kwarg(st.dataframe))

            with res_col2:
                # Bar chart of class probabilities
                st.markdown("##### Prediction Probability Distribution")
                chart_data = pd.DataFrame({
                    "Risk Category": classes,
                    "Probability": probabilities,
                }).set_index("Risk Category")
                st.bar_chart(chart_data, color="#2980b9", height=220)

            # Recommendations section
            st.markdown("---")
            st.subheader("📋 Advisor Intervention & Academic Support Guidance")

            if prediction == "Safe":
                st.success(
                    "**Status: Student is on track.**\n\n"
                    "The student demonstrates sound attendance, consistent assignment submission, and adequate study hours.\n\n"
                    "**Recommended Support Actions:**\n"
                    "- **Positive Reinforcement:** Acknowledge consistent effort and encourage sustained engagement.\n"
                    "- **Periodic Monitoring:** Schedule light check-ins midway through the term to ensure pace is maintained.\n"
                    "- **Enrichment Opportunities:** Recommend advanced coursework, student mentoring roles, or undergraduate research."
                )
            elif prediction == "At-Risk":
                st.warning(
                    "**Status: Moderate Risk — Early Academic Intervention Recommended.**\n\n"
                    "Indicators reveal borderline attendance, gaps in coursework completion, or below-average study hours.\n\n"
                    "**Recommended Support Actions:**\n"
                    "- **Targeted Assignment Review:** Connect with the student to review outstanding or poorly scored assignments.\n"
                    "- **Tutoring & Peer Study:** Connect the student with departmental tutoring centers or study groups.\n"
                    "- **Advisor Check-In:** Schedule a bi-weekly 15-minute advising session to monitor attendance and study habits.\n"
                    "- **Time-Management Coaching:** Provide a structured weekly calendar for coursework and study blocks."
                )
            else:  # High-Risk
                st.error(
                    "**Status: High Risk — Urgent Proactive Outreach Required.**\n\n"
                    "Low attendance, past failure history, low assignment completion, and limited study hours strongly suggest "
                    "danger of course failure or semester withdrawal.\n\n"
                    "**Recommended Support Actions:**\n"
                    "- **Immediate Outreach:** Initiate proactive outreach from the academic dean or department advisor within 48 hours.\n"
                    "- **Comprehensive Diagnostic Session:** Discuss root causes (academic difficulty, health, personal challenges, access issues).\n"
                    "- **Personalized Academic Recovery Plan (ARP):** Formulate a written plan establishing minimum weekly study hours and attendance targets.\n"
                    "- **Dedicated Tutoring Support:** Assign a designated 1-on-1 peer tutor or faculty mentor.\n"
                    "- **Course Load Review:** Consider options such as course withdrawal or auditing before drop deadlines to protect GPA."
                )

            st.caption(
                "Note: Recommendations are advisory and non-punitive guidelines designed to empower advisors and students. "
                "Interventions should always be tailored through empathetic dialogue with the student."
            )

    # -------------------------------------------------------------
    # TAB 2: Preset Student Archetypes
    # -------------------------------------------------------------
    with tab_batch:
        st.subheader("Explore Preset Student Archetypes")
        st.caption("Test how the early-warning model evaluates diverse student profiles:")

        presets = {
            "Profile A: High-Performing Student (Expected: Safe)": {
                "attendance": 94.0,
                "study_hours": 10.5,
                "past_failures": 0,
                "assignments_completed_pct": 95.0,
                "parental_education": "Higher",
                "family_income": "Medium",
                "extracurricular": "Yes",
                "internet_access": "Yes",
                "previous_grade": 86.0,
            },
            "Profile B: Borderline Student with Attendance Slump (Expected: At-Risk)": {
                "attendance": 72.0,
                "study_hours": 5.5,
                "past_failures": 1,
                "assignments_completed_pct": 70.0,
                "parental_education": "Secondary",
                "family_income": "Low",
                "extracurricular": "No",
                "internet_access": "Yes",
                "previous_grade": 71.0,
            },
            "Profile C: Severe Disengagement & Past Failures (Expected: High-Risk)": {
                "attendance": 42.0,
                "study_hours": 2.0,
                "past_failures": 4,
                "assignments_completed_pct": 45.0,
                "parental_education": "None",
                "family_income": "Low",
                "extracurricular": "No",
                "internet_access": "No",
                "previous_grade": 50.0,
            },
        }

        selected_preset = st.selectbox("Select an Archetype to Inspect", list(presets.keys()))
        preset_values = presets[selected_preset]

        preset_df = pd.DataFrame([preset_values])[RAW_COLUMNS]
        st.dataframe(preset_df, **get_stretch_kwarg(st.dataframe))

        if st.button("Evaluate Selected Archetype", type="primary", **get_stretch_kwarg(st.button)):
            pred = model.predict(preset_df)[0]
            probs = model.predict_proba(preset_df)[0]
            classes = list(model.classes_)

            icon = RISK_ICONS.get(pred, "⚪")
            color = RISK_COLORS.get(pred, "#333333")

            st.markdown(
                f"""
                <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 15px; border-radius: 6px; margin-top: 15px;">
                    <h4 style="margin: 0; color: #555;">Result</h4>
                    <h3 style="margin: 5px 0 0 0; color: {color};">{icon} {pred}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            res_df = pd.DataFrame({
                "Risk Category": classes,
                "Probability": [f"{p*100:.2f}%" for p in probs],
            })
            st.table(res_df)

    # -------------------------------------------------------------
    # TAB 3: Model Architecture & Evaluation
    # -------------------------------------------------------------
    with tab_about:
        st.subheader("Model Architecture and Performance Artifacts")

        st.markdown(
            """
            ### 🏛️ Pipeline Design
            The deployed pipeline couples data preprocessing and predictive modeling in a unified scikit-learn `Pipeline`:
            1. **Imputation:** Median imputation for missing attendance and study hours (MCAR mechanism).
            2. **Scaling:** Z-score standardization (`StandardScaler`) on all continuous academic metrics.
            3. **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore')` expanding categories into 16 numeric features.
            4. **Classifier:** Multinomial Logistic Regression (`solver='lbfgs'`, `C=1.0`).
            """
        )

        st.markdown("### 📊 Test Set Performance Summary (N = 20,000)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Overall Accuracy", "93.94%")
        col_m2.metric("Macro F1-Score", "93.92%")
        col_m3.metric("High-Risk Recall", "94.36%")
        col_m4.metric("Balanced Accuracy", "93.78%")

        st.markdown("### 🔍 Diagnostic Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("##### 5-Fold Cross-Validation Comparison")
            if os.path.exists("reports/figures/model_comparison.png"):
                st.image("reports/figures/model_comparison.png", **get_stretch_kwarg(st.image))
            else:
                st.info("Model comparison chart not found in reports/figures/.")

        with viz_col2:
            st.markdown("##### Test Set Confusion Matrix")
            if os.path.exists("reports/figures/confusion_matrix.png"):
                st.image("reports/figures/confusion_matrix.png", **get_stretch_kwarg(st.image))
            else:
                st.info("Confusion matrix chart not found in reports/figures/.")

        st.markdown("---")
        st.markdown(
            """
            ### 🔬 Mathematical Findings from Phase 1 & Phase 2
            - **Linear Ground Truth:** In Phase 1 inspection, OLS and latent logit analysis revealed that the synthetic target `risk_category`
              was created by partitioning an underlying linear function of student engagement and historical ability.
            - **Tree Discretization Deficit:** Logistic Regression outperformed complex ensembles (Random Forest and XGBoost)
              because linear hyperplanes are represented naturally by logistic models, whereas tree models produce approximation errors
              from recursive orthogonal splits.
            - **Orthogonal Predictors:** Mutual correlation among input features was virtually zero ($-0.003$ to $+0.005$), confirming independent parametric generation.
            """
        )


if __name__ == "__main__":
    main()
