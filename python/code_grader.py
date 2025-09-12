import numpy as np
import pandas as pd
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import traceback
import shutil

# Import from our architecture
from .grade_functions import GRADERS
import common
from .competition import *



def grade_llm_code(train_code: dict, competition_id: str, language: str, mono_predict: bool, folds: int | None) -> dict:
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

    # Set folds override
    if folds is not None:
        comp["cv_folds"] = folds

    common.set_bench_info({"grader_competition": competition_id, "grader_language": language})

    log_error = lambda x: common.report_error(x)
    do_shutdown = lambda x: common.graceful_exit(x)

    # Create competition object for grading stage
    comp = Competition(competition_id, comp, {}, "data", log_error, do_shutdown)

    grader = comp.get("grader")
    if grader is None:
        # Execute the default grader
        grader = "default"
    elif GRADERS.get(grader) is None:
        common.report_error(f"grade_llm_code() : internal error : grader not found : {grader}")
        common.graceful_exit(1)

    common.set_bench_info({"grader": grader})

    print("grade_llm_code() : executing...")

    # Load data
    try:
        # DATA LOADING GOES HERE
        loader_name = comp.metadata.get("data_loader", "default")
        loader_class = DATA_LOADERS.get(loader_name)
        if loader_class is None:
            raise ValueError(f"Data loader '{loader_name}' not found")

        loader = loader_class()

        if folds is None:
            common.report_error("Folds are unset")
            common.graceful_exit(1)

        scores = []

        for fold_idx in range(folds):
            # Load training data
            train_dataset = loader.load_train_data(comp, fold_idx, self.base_path)

            # Load validation features
            val_features_dataset = loader.load_validation_features(comp, fold_idx, self.base_path)

            # Load validation labels
            val_labels = loader.load_validation_labels(comp, fold_idx, self.base_path)


            if mono_predict:
                if not callable(train_code.get("train_and_predict")):
                    raise ValueError("train_and_predict is not a callable function")
            else:
                for func_name in ["train", "prepare_val", "predict"]:
                    if not callable(train_code.get(func_name)):
                        raise ValueError(f"{func_name} is not a callable function")

            # Execute the appropriate prediction function
            try:
                if mono_predict:
                    predictions = train_code["train_and_predict"](train_dataset, val_features_dataset)
                else:
                    # Train phase
                    train_output = train_code["train"](train_dataset)

                    # Prepare validation phase
                    val_prepared = train_code["prepare_val"](val_features_dataset, train_output)

                    # Predict phase
                    predictions = train_code["predict"](train_output, val_prepared)

                    # Grade the predictions against true labels
                    score = GRADERS[grader](predictions, val_labels, comp.metadata)
                    scores.append(fold_score)
                    print(f"grade_llm_code() : finished fold {i+1}/{folds}")

            except Exception as e:
                common.log_error(f"Error during fold {fold_idx} execution: {str(e)}")
                common.do_shutdown(1)

    except Exception as e:
        common.report_error(f"grade_llm_code() : grading failed : {e=} ({competition_id})")
        common.graceful_exit(1)

    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        common.report_error("Submission code failed for all CV folds")
        common.graceful_shutdown(1)
    score = np.mean(valid_scores)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }

