import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import json
import sys
import os
from glob import glob

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

    common.set_bench_info({"grader_competition": competition_id, "grader_language": language})

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
    score = autograde_cvfold(X, y, train_and_predict, GRADERS[grader], comp, scores, language)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }
