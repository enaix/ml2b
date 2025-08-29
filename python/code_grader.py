
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import json
import sys
import os
from glob import glob


from grade_functions import *


# Submission grader code
# ======================

def autograde_cvfold(X: pd.DataFrame, y: pd.DataFrame, train_code: object, grader: object, competition_id: str, comp: dict, scores: list, language: str, mono_predict: bool) -> float:
    def cvfold_run(X_train, y_train, X_val, y_val):
        def mono_predict():
            return train_code["train_and_predict"](X_train, y_train, X_val) # Won't the distribution of X_train leak onto X_val?

        def modular_predict():
            train_output = train_code["train"](X_train, y_train)
            prepare_val_output = train_code["prepare_val"](X_val, train_output)

            predict = train_code["predict"](train_output, prepare_val_output)
            return predict

        try:
            # Execute submission code
            if mono_predict:
                preds = mono_predict()
            else:
                preds = modular_predict()
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

            X_train, y_train = df_train.drop(columns=[comp["target_col"]]), df_train[[comp["target_col"]]]

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




def grade_llm_code(train_code: dict, competition_id: str, language: str, mono_predict: bool) -> dict:
    """
    Executes LLM-generated code, computes CV scores, and returns metrics.
    """
    # Load competition config
    if not os.path.exists("data/competitions.json"):
        common.report_error("Could not find data/competitions.json")
        common.graceful_exit(1)

    with open("data/competitions.json", 'r') as f:
        comp = json.load(f).get(competition_id)

    if comp is None:
        common.report_error(f"Unknown competition: {competition_id}")
        common.graceful_exit(1)

    common.set_bench_info({"grader_competition": competition_id, "grader_language": language})

    # Load data
    try:
        train = pd.read_csv(f"data/{competition_id}/train.csv")  # Data should be mounted in this format
        X, y = train.drop(columns=[comp["target_col"]]), train[comp["target_col"]]
    except Exception as e:
        common.report_error(f"grade_llm_code() : internal error : data loading failed: {e=} (file data/{competition_id}/train.csv)")
        common.graceful_exit(1)

    scores = []

    grader = comp.get("grader")
    if grader is None:
        # Execute the default grader
        grader = "cvfold"
    elif GRADERS.get(grader) is None:
        common.report_error(f"grade_llm_code() : internal error : grader not found : {grader}")
        common.graceful_exit(1)

    common.set_bench_info({"grader": grader})

    # Execute the autograder
    print("grade_llm_code() : executing...")
    score = autograde_cvfold(X, y, train_code, GRADERS[grader], comp, scores, language, mono_predict)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }
