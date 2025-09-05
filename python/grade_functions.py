import common

import numpy as np
import pandas as pd
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


METRICS = {
    "roc_auc_score": roc_auc_score,
    "f1_score": f1_score,
    "accuracy_score": accuracy_score,
    "f1_score_avg_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    "root_mean_squared_error": root_mean_squared_error,
    "log_loss": log_loss,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
}

# Grader functions
# ================
def grader_default(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(f"grader_default() : internal error : metric not found : {comp['METRIC']}")
        common.graceful_exit(1)

    try:
        score = metric(val, pred)
        return score
    except Exception as e:
        common.report_error(f"Greader execution failed : {sys.exc_info()}")
        return np.nan


GRADERS = {
    "default": grader_default
}
