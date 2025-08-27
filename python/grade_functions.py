import common


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

def grader_default(pred: pd.DataFrame, val: pd.DataFrame, comp: dict) -> np.float64:
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


# Submission grader code
# ======================

def autograde_cvfold(X: pd.DataFrame, y: pd.DataFrame, train_and_predict: object, grader: object, competition_id: str, comp: dict, scores: list, language: str) -> float:
    def cvfold_run(X_train, y_train, X_val, y_val):
        try:
            # Execute submission code
            preds = train_and_predict(X_train, y_train, X_val) # Won't the distribution of X_train leak onto X_val?
            score = grader(preds, y_val, comp)
            scores.append(score)
            print(f"autograde_cvfold() : finished fold {i+1}/{comp['cv_folds']}")
        except Exception as e:
            common.report_error(f"Submission code execution failed : {sys.exc_info()}")
            scores.append(np.nan)  # Mark failed folds


    num_folds = len(glob(f"data/folds/{competition_id}/train_*.csv"))
    if num_folds == comp["cv_folds"]:
        # Use existing folds
        for i in range(num_folds):
            train_path = os.path.join("data", "folds", competition_id, f"train_{i}.csv")
            x_val_path = os.path.join("data", "folds", competition_id, f"X_val_{i}.csv")
            y_val_path = os.path.join("data", "private", competition_id, f"y_val_{i}.csv")

            df_train = pd.read_csv(train_path)
            X_val = pd.read_csv(x_val_path)
            y_val = pd.read_csv(y_val_path)

            X_train, y_train = df_train.drop(columns=[comp["target_col"]]), df_train[comp["target_col"]]

            cvfold_run(X_train, y_train, X_val, y_val)

    else:
        # Split them manually
        kf = KFold(n_splits=comp["cv_folds"])
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            cvfold_run(X_train, y_train, X_val, y_val)

    # Aggregate results
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        common.report_error("Submission code failed for all CV folds")
        common.graceful_shutdown(1)
    return np.mean(valid_scores)



