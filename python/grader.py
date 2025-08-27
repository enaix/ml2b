import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import json
import sys
import os

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

import common


METRICS = {
    "roc_auc_score": roc_auc_score,
    "f1_score": f1_score,
    "accuracy_score": accuracy_score,
    "f1_score_avg_macro": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="macro"
    ),
    "root_mean_squared_error": root_mean_squared_error,
    "log_loss": log_loss,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
}


# Submission grader code
# ======================


def autograde_cvfold(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_and_predict: object,
    metric: object,
    comp: dict,
    scores: list,
    language: str,
) -> float:
    kf = KFold(n_splits=comp["cv_folds"])
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        try:
            # Execute submission code
            preds = train_and_predict(
                X_train, y_train, X_val
            )  # Won't the distribution of X_train leak onto X_val?
            score = metric(y_val, preds)
            scores.append(score)
            print(f"autograde_cvfold() : finished fold {i+1}/{comp['cv_folds']}")
        except Exception as e:
            common.report_error(f"Submission code execution failed : {sys.exc_info()}")
            scores.append(np.nan)  # Mark failed folds

    # Aggregate results
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        common.report_error("Submission code failed for all CV folds")
        common.graceful_shutdown(1)
    return np.mean(valid_scores)


GRADERS = {"cvfold": autograde_cvfold}


def grade_llm_code(
    train_and_predict: object, competition_id: str, language: str = "English"
) -> dict:
    """
    Executes LLM-generated code, computes CV scores, and returns metrics.
    """
    # Load competition config
    if not os.path.exists("data/competitions.json"):
        common.report_error("Could not find data/competitions.json")
        common.graceful_exit(1)

    with open("data/competitions.json", "r") as f:
        comp = json.load(f).get(competition_id)

    if comp is None:
        common.report_error(f"Unknown competition: {competition_id}")
        common.graceful_exit(1)

    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(
            f"grade_llm_code() : internal error : metric not found : {comp['METRIC']}"
        )
        common.graceful_exit(1)

    common.set_bench_info({"competition": competition_id, "language": language})

    # Load data
    try:
        train = pd.read_csv(
            f"data/{competition_id}/train.csv"
        )  # Data should be mounted in this format
        X, y = train.drop(columns=[comp["target_col"]]), train[comp["target_col"]]
    except Exception as e:
        common.report_error(
            f"grade_llm_code() : internal error : data loading failed: {e=} (file data/{competition_id}/train.csv)"
        )
        common.graceful_exit(1)

    scores = []

    grader = comp.get("grader")
    if grader is None:
        # Execute the default grader
        grader = "cvfold"
    elif GRADERS.get(grader) is None:
        common.report_error(
            f"grade_llm_code() : internal error : grader not found : {grader}"
        )
        common.graceful_exit(1)

    common.set_bench_info({"grader": grader})

    # Execute the autograder
    print("grade_llm_code() : executing...")
    score = GRADERS[grader](X, y, train_and_predict, metric, comp, scores, language)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }
