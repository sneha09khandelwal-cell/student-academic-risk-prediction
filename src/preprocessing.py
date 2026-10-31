"""
Preprocessing pipeline for the Student Early-Warning and Academic Support System.

This module provides reusable functions and a scikit-learn ColumnTransformer
pipeline to clean, impute, scale, and encode student features while strictly
preventing data leakage.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_COLS: List[str] = [
    "attendance",
    "study_hours",
    "past_failures",
    "assignments_completed_pct",
    "previous_grade",
]

CATEGORICAL_COLS: List[str] = [
    "parental_education",
    "family_income",
    "extracurricular",
    "internet_access",
]

TARGET_COL: str = "risk_category"
LEAKAGE_COLS: List[str] = ["final_score"]


def standardize_categorical_casing(
    X: Union[pd.DataFrame, pd.Series, np.ndarray]
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Standardize inconsistent categorical values, specifically 'none' to 'None'
    in parental_education or any other categorical text field.
    """
    if isinstance(X, pd.DataFrame):
        return X.replace({"none": "None"})
    if isinstance(X, pd.Series):
        return X.replace("none", "None")
    
    # Numpy array handling
    X_arr = np.array(X, dtype=object, copy=True)
    X_arr[X_arr == "none"] = "None"
    return X_arr


def get_feature_groups() -> Tuple[List[str], List[str]]:
    """Return lists of numerical and categorical column names."""
    return list(NUMERICAL_COLS), list(CATEGORICAL_COLS)


def build_preprocessor(
    numerical_cols: List[str] = None,
    categorical_cols: List[str] = None,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer pipeline for preprocessing.

    Numerical Pipeline:
      1. SimpleImputer(strategy='median')
      2. StandardScaler()

    Categorical Pipeline:
      1. FunctionTransformer(standardize_categorical_casing)
      2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    num_cols = numerical_cols or NUMERICAL_COLS
    cat_cols = categorical_cols or CATEGORICAL_COLS

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "standardizer",
                FunctionTransformer(
                    standardize_categorical_casing,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def load_raw_data(data_path: str = "data/student_data.csv") -> pd.DataFrame:
    """Load the raw student dataset without modifying the underlying file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    logger.info(f"Loading raw dataset from {data_path}")
    return pd.read_csv(data_path)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    leakage_cols: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate features and target, strictly removing data leakage columns,
    and split into stratified train and test partitions.
    """
    cols_to_drop = [target_col]
    if leakage_cols is None:
        leakage_cols = LEAKAGE_COLS

    for col in leakage_cols:
        if col in df.columns:
            cols_to_drop.append(col)
            logger.info(f"Excluding post-outcome leakage feature: '{col}'")

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    logger.info(
        f"Performing stratified split: test_size={test_size}, random_state={random_state}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(
    data_path: str = "data/student_data.csv",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    """
    Execute end-to-end preprocessing:
      1. Load raw data (unmodified).
      2. Exclude final_score to prevent data leakage.
      3. Stratified 80/20 train-test split.
      4. Fit preprocessor ONLY on X_train.
      5. Transform X_train and X_test.
      6. Persist raw splits, preprocessor, and configuration metadata.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = load_raw_data(data_path)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=TARGET_COL,
        leakage_cols=LEAKAGE_COLS,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info("Building ColumnTransformer preprocessor...")
    preprocessor = build_preprocessor()

    # Fit strictly on training features
    logger.info("Fitting preprocessor strictly on X_train (never on test data)...")
    preprocessor.fit(X_train)

    feature_names_out = list(preprocessor.get_feature_names_out())
    logger.info(f"Generated {len(feature_names_out)} preprocessed features: {feature_names_out}")

    # Transform partitions
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Save fitted preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved fitted preprocessor to {preprocessor_path}")

    # Save raw split DataFrames
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    # Save combined train and test splits (raw features + target)
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # Save transformed arrays as DataFrames for convenience
    df_train_trans = pd.DataFrame(X_train_transformed, columns=feature_names_out)
    df_train_trans[TARGET_COL] = y_train.values
    df_train_trans.to_csv(os.path.join(processed_dir, "train_preprocessed.csv"), index=False)

    df_test_trans = pd.DataFrame(X_test_transformed, columns=feature_names_out)
    df_test_trans[TARGET_COL] = y_test.values
    df_test_trans.to_csv(os.path.join(processed_dir, "test_preprocessed.csv"), index=False)

    # Compute and persist configuration metadata
    train_class_counts = y_train.value_counts().to_dict()
    test_class_counts = y_test.value_counts().to_dict()
    train_class_pct = (y_train.value_counts(normalize=True) * 100).to_dict()
    test_class_pct = (y_test.value_counts(normalize=True) * 100).to_dict()

    config = {
        "random_state": random_state,
        "test_size": test_size,
        "target_col": TARGET_COL,
        "excluded_leakage_cols": LEAKAGE_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "raw_features_count": X_train.shape[1],
        "transformed_features_count": len(feature_names_out),
        "transformed_feature_names": feature_names_out,
        "train_class_distribution": {
            "counts": train_class_counts,
            "percentages": train_class_pct,
        },
        "test_class_distribution": {
            "counts": test_class_counts,
            "percentages": test_class_pct,
        },
        "artifacts": {
            "preprocessor_joblib": preprocessor_path,
            "X_train_csv": os.path.join(processed_dir, "X_train.csv"),
            "X_test_csv": os.path.join(processed_dir, "X_test.csv"),
            "y_train_csv": os.path.join(processed_dir, "y_train.csv"),
            "y_test_csv": os.path.join(processed_dir, "y_test.csv"),
            "train_csv": os.path.join(processed_dir, "train.csv"),
            "test_csv": os.path.join(processed_dir, "test.csv"),
            "train_preprocessed_csv": os.path.join(processed_dir, "train_preprocessed.csv"),
            "test_preprocessed_csv": os.path.join(processed_dir, "test_preprocessed.csv"),
        },
    }

    config_path = os.path.join(processed_dir, "split_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved preprocessing configuration metadata to {config_path}")

    return config


if __name__ == "__main__":
    config = run_preprocessing_pipeline()
    print("\n--- PREPROCESSING COMPLETED SUCCESSFULLY ---")
    print(f"Train Rows: {config['train_rows']}, Test Rows: {config['test_rows']}")
    print(f"Features: {config['raw_features_count']} raw -> {config['transformed_features_count']} preprocessed")
    print("Train Class Distribution:", config['train_class_distribution']['percentages'])
    print("Test Class Distribution:", config['test_class_distribution']['percentages'])
