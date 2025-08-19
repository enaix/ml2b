
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import json
import sys

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

import common

# Competitions database
# =====================

COMPETITIONS = {
    # -------------------------------------------------------------------------
    # Binary Classification (13)
    # -------------------------------------------------------------------------
    "wids-datathon-2020": {
        "metric": roc_auc_score,
        "target_col": "hospital_death",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "high-frequency-price-prediction": {
        "metric": roc_auc_score,
        "target_col": "y",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "explicit-content-detection": {
        "metric": f1_score,
        "target_col": "target",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "binary predictions (0 or 1)"
    },
    "2020-classification-data-challenge": {
        "metric": roc_auc_score,
        "target_col": "Buy",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "tabular-playground-series-mar-2021": {
        "metric": roc_auc_score,
        "target_col": "target",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "uwaterloo-stat441-jewelry": {
        "metric": roc_auc_score,
        "target_col": "Revenue",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "ai-cancer-predictions": {
        "metric": accuracy_score,
        "target_col": "diagnosis",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "binary predictions (0 or 1)"
    },
    "eco3119-yonsei-2021": {
        "metric": accuracy_score,
        "target_col": "y",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "binary predictions (0 or 1)"
    },
    "porto-seguro-challenge": {
        "metric": f1_score,
        "target_col": "y",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "binary predictions (0 or 1)"
    },
    "stroke-prediction-s3e2": {
        "metric": roc_auc_score,
        "target_col": "stroke",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "employee-attrition-s3e3": {
        "metric": roc_auc_score,
        "target_col": "Attrition",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },
    "credit-card-fraud-s3e4": {
        "metric": roc_auc_score,
        "target_col": "Class",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "probabilities for class 1 (floats in [0, 1])"
    },

    # -------------------------------------------------------------------------
    # Multi-Class Classification (4)
    # -------------------------------------------------------------------------
    "multi-class-classification": {
        "metric": accuracy_score,
        "target_col": "Class",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "class labels (integers 0-5)"
    },
    "megafon-accelerator": {
        "metric": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        "target_col": "TARGET",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "class labels (integers 0, 1, or 2)"
    },
    "syde-522-winter-2021": {
        "metric": accuracy_score,
        "target_col": "label",
        "direction": "maximize",
        "cv_folds": 5,
        "output_hint": "class labels (integers)"
    },
    "tabular-playground-may-2021": {
        "metric": log_loss,
        "target_col": "target",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "probabilities for all classes (N x K matrix, floats in [0, 1])"
    },

    # -------------------------------------------------------------------------
    # Regression (7)
    # -------------------------------------------------------------------------
    "ieor-242-nyc-taxi": {
        "metric": root_mean_squared_error,
        "target_col": "duration",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted durations (floats >= 0)"
    },
    "financial-engineering-1": {
        "metric": mean_squared_error,
        "target_col": "col_5",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted values for col_5 (floats)"
    },
    "actuarial-loss-prediction": {
        "metric": root_mean_squared_error,
        "target_col": "UltimateIncurredClaimCost",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted total claims payments (floats >= 0)"
    },
    "she-hacks-2021": {
        "metric": root_mean_squared_error,
        "target_col": "Unique Headcount",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "integer counts >= 0"
    },
    "google-brain-ventilator": {
        "metric": mean_absolute_error,
        "target_col": "pressure",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted pressure values (floats, in cmH2O)"
    },
    "crime-learn": {
        "metric": root_mean_squared_error,
        "target_col": "ViolentCrimesPerPop",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted crime rates (floats >= 0)"
    },
    "california-housing-s3e1": {
        "metric": root_mean_squared_error,
        "target_col": "MedHouseVal",
        "direction": "minimize",
        "cv_folds": 5,
        "output_hint": "predicted house values (floats >= 0)"
    }
}





# Submission grader code
# ======================

def autograde_cvfold(X: pd.DataFrame, y: pd.DataFrame, train_and_predict: object, comp: dict, scores: list, language: str) -> float:
    kf = KFold(n_splits=comp["cv_folds"])
    for train_idx, val_idx in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        try:
            # Execute submission code
            preds = train_and_predict(X_train, y_train, X_val) # Won't the distribution of X_train leak onto X_val?
            score = comp["metric"](y_val, preds)
            scores.append(score)
        except Exception as e:
            common.report_error(f"Submission code execution failed : {sys.exc_info()}")
            scores.append(np.nan)  # Mark failed folds

    # Aggregate results
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        common.report_error("Submission code failed for all CV folds")
        common.graceful_shutdown(1)
    return valid_scores.mean()



GRADERS = {
    "cvfold": autograde_cvfold
}



def grade_llm_code(train_and_predict: object, competition_id: str, language: str = "English") -> dict:
    """
    Executes LLM-generated code, computes CV scores, and returns metrics.
    """
    # Load competition config
    comp = COMPETITIONS.get(competition_id)
    if comp is None:
        common.report_error(f"Unknown competition: {competition_id}")
        common.graceful_exit(1)

    common.set_bench_info({"competition": competition_id, "language": language})

    # Load data
    try:
        train = pd.read_csv(f"data/{competition_id}/train.csv")  # Data should be mounted in this format
        X, y = train.drop(columns=[comp["target_col"]]), train[comp["target_col"]]
    except Exception as e:
        common.report_error(f"grade_llm_code() : internal error : data loading failed: {e=}")
        common.graceful_exit(1)

    scores = []

    grader = comp.get("grader")
    if grader is None:
        # Execute the default grader
        grader = "cvfold"
    common.set_bench_info({"grader": grader})

    # Execute the autograder
    score = GRADERS[grader](X, y, train_and_predict, comp, scores, language)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }
