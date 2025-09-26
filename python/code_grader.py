import numpy as np
import json
import os
import traceback

# Import from our architecture
from loaders import DATA_LOADERS
from python.grade_functions import GRADERS
import python.common as common
from python.competition import *



def grade_llm_code(train_code: dict, competition_id: str, language: str, mono_predict: bool, folds: int | None, extended_schema: bool) -> dict:
    """
    Executes LLM-generated code, computes CV scores, and returns metrics.
    """
    # Load competition config
    if not os.path.exists("competitions/competitions.json"):
        common.report_error("Could not find competitions/competitions.json")
        common.graceful_exit(1)

    with open("competitions/competitions.json", 'r') as f:
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
    comp = Competition(competition_id, comp, {}, "competitions", log_error, do_shutdown)

    grader = comp.metadata.get("grader")
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
        loader_name = comp.metadata.get("load_strategy", "default")
        loader_class = DATA_LOADERS.get(loader_name)
        if loader_class is None:
            raise ValueError(f"Data loader '{loader_name}' not found")

        loader = loader_class()

        if folds is None:
            common.report_error("grade_llm_code() : folds variable is unset")
            common.graceful_exit(1)

        base_path = "/home/bench/competitions"

        scores = []

        for fold_idx in range(folds):
            # Load training data
            loader_train_dataset = loader.load_train_data(comp, fold_idx, base_path)

            # Load validation features
            loader_val_features_dataset = loader.load_validation_features(comp, fold_idx, base_path)

            # Load validation labels
            val_labels = loader.load_validation_labels(comp, fold_idx, base_path)

            # if not isinstance(loader_train_dataset, dict) and not isinstance(loader_val_features_dataset, dict):
                # common.report_error(f"Data loader shapes : train {str(loader_train_dataset.shape)}; val {str(loader_val_features_dataset.shape)}; val_y {val_labels.shape}")

            if mono_predict:
                if not callable(train_code.get("train_and_predict")):
                    raise ValueError("train_and_predict is not a callable function")
            else:
                for func_name in ["train", "prepare_val", "predict"]:
                    if not callable(train_code.get(func_name)):
                        raise ValueError(f"{func_name} is not a callable function")

            if extended_schema:
                # Ensure that the arguments are sorted
                # schema = list(loader.schema_dict().items())
                train_dataset = loader_train_dataset
                val_features_dataset = loader_val_features_dataset
            else:
                train_dataset = loader_train_dataset
                val_features_dataset = loader_val_features_dataset

            # Execute the appropriate prediction function
            try:
                if mono_predict:
                    if extended_schema and isinstance(train_dataset, dict):
                        predictions = train_code["train_and_predict"](*train_dataset.values(), *val_features_dataset.values())
                    else:
                        predictions = train_code["train_and_predict"](train_dataset, val_features_dataset)
                else:
                    # Train phase
                    train_output = (train_code["train"](*train_dataset.values()) if extended_schema and isinstance(train_dataset, dict) else train_code["train"](train_dataset))

                    # Prepare validation phase
                    val_prepared = (train_code["prepare_val"](train_output, *val_features_dataset.values()) if extended_schema and isinstance(val_features_dataset, dict) else
                                    train_code["prepare_val"](train_output, val_features_dataset))

                    # Predict phase
                    predictions = train_code["predict"](train_output, val_prepared)

                # Grade the predictions against true labels
                # common.report_error(f"Grader shapes : pred {predictions.shape}; val_prepared {val_prepared.shape}; val_labels {val_labels.shape}")
                score = GRADERS[grader](predictions, val_labels, comp.metadata)
                scores.append(score)
                print(f"grade_llm_code() : finished fold {fold_idx+1}/{folds}")

            except Exception:
                common.report_error(f"Error during fold {fold_idx} execution: {traceback.format_exc()}")
                common.graceful_exit(1)

    except Exception as e:
        common.report_error(f"grade_llm_code() : grading failed : {e=} ({competition_id})")
        common.graceful_exit(1)

    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        common.report_error("Submission code failed for all CV folds")
        common.graceful_exit(1)
    score = np.mean(valid_scores)

    return {
        "score": score,
        "cv_scores": scores,  # For debugging
    }

