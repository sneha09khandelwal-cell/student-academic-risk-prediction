"""
Model definitions and pipeline builders for the Student Early-Warning System.

Encapsulates:
  - Logistic Regression pipeline
  - Decision Tree pipeline
  - Random Forest pipeline
  - XGBoost pipeline (with StringTargetXGBClassifier wrapper for string classes)
All pipelines include the scikit-learn ColumnTransformer preprocessor.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


class StringTargetXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost Classifier wrapper that natively handles multiclass string targets
    (e.g., 'At-Risk', 'Safe', 'High-Risk') for seamless integration with
    scikit-learn Pipelines, Cross-Validation, and serialization.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss",
            **self.kwargs,
        )
        self.classes_ = None
        self._class_to_idx = None
        self._idx_to_class = None

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y)
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        self._idx_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}
        y_encoded = np.array([self._class_to_idx[val] for val in y])
        self.model.fit(X, y_encoded, **fit_params)
        return self

    def predict(self, X):
        preds_encoded = self.model.predict(X)
        return np.array([self._idx_to_class[idx] for idx in preds_encoded])

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True):
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.kwargs[k] = v
        self.model.set_params(**params)
        return self


def get_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Construct end-to-end scikit-learn Pipelines for all four classifiers.
    Each pipeline couples the ColumnTransformer preprocessor with the estimator,
    guaranteeing leak-free cross-validation and raw-input inference.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_leaf=10,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    StringTargetXGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models
