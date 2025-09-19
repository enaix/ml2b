# This is the entrypoint file

import os
import importlib
from typing import Any
from enum import StrEnum

import python.common as common
from python.code_grader import grade_llm_code



class BenchMode(StrEnum):
    MonolithicPredict = "MONO_PREDICT"
    ModularPredict = "MODULAR_PREDICT"



# Getting the benchmark name
# ==========================
def get_bench_params() -> dict:
    comp_id = os.environ.get("COMPETITION_ID")
    if not comp_id:
        common.report_error("Environment variable COMPETITION_ID is unset")
        common.graceful_exit(1)
    
    bench_lang = os.environ.get("BENCH_LANG")
    if not bench_lang:
        bench_lang = "English"
    
    bench_mode = os.environ.get("BENCH_MODE")
    if not bench_mode:
        common.report_error("Environment variable BENCH_MODE is unset")
        common.graceful_exit(1)

    try:
        mode = BenchMode(bench_mode)
    except ValueError:
        common.report_error(f"Bad BENCH_MODE value: {bench_mode}. BENCH_MODE should be either MONO_PREDICT or MODULAR_PREDICT")
        common.graceful_exit(1)

    bench_folds_ov = os.environ.get("BENCH_FOLDS_OVERRIDE")
    if bench_folds_ov is not None:
        try:
            bench_folds = int(bench_folds_ov)
        except ValueError:
            common.report_error(f"Bad BENCH_FOLDS_OVERRIDE value: {bench_folds_ov} is not an integer")
            common.graceful_exit(1)
    else:
        bench_folds = None

    extended_schema_val = os.environ.get("EXTENDED_SCHEMA")
    if extended_schema_val is None:
        common.report_error("Environment variable EXTENDED_SCHEMA is unset")
        common.graceful_exit(1)
    if extended_schema_val.lower() in ["1", "y", "yes", "true"]:
        extended_schema = True
    elif extended_schema_val.lower() in ["0", "n", "no", "false"]:
        extended_schema = False
    else:
        common.report_error("Bad EXTENDED_SCHEMA value: must be one either 1/0 OR y/n OR yes/no OR true/false")
        common.graceful_exit(1)

    submission_name = os.environ.get("SUBMISSION_NAME")

    return {"comp_id": comp_id, "bench_lang": bench_lang, "bench_mode": mode, "bench_folds": bench_folds, "extended_schema": extended_schema, "submission_name": submission_name}


# Loading the submission code
# ===========================

def _load_submission_code(codepath: os.PathLike) -> Any:
    spec = importlib.util.spec_from_file_location("submission", codepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_mono_submission(submission_name: str) -> dict:
    try:
        mod = importlib.import_module("submission.code")  # loads the file 'submission.py'
        if not hasattr(mod, "train_and_predict"):
            common.report_error(
                "Submission code does not have the train_and_predict function"
            )
            common.graceful_exit(1)
        return {"train_and_predict": mod.train_and_predict}
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)


def load_modular_submission(submission_name: str) -> dict:
    try:
        mod = importlib.import_module("submission.code") # loads the file 'submission.py'
        funcs = {}
        for func in ["train", "prepare_val", "predict"]:
            if not hasattr(mod, func):
                common.report_error(f"Submission code does not have the {func} function")
                common.graceful_exit(1)
            funcs[func] = getattr(mod, func)
        return funcs
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)


def main():
    # Initialize logging
    common.bench_results.is_in_container = True
    submission_name = os.environ.get("SUBMISSION_NAME")

    if submission_name is None:
        common.report_error("Environment variable SUBMISSION_NAME is unset")
        common.graceful_exit(1)
    common.bench_results.submission_name = submission_name
    # Init complete

    params = get_bench_params()
    params["submission_name"] = submission_name
    if params["bench_mode"] == BenchMode.MonolithicPredict:
        train_code = load_mono_submission(common.submission_name)
    else:
        train_code = load_modular_submission(common.submission_name)

    results = grade_llm_code(train_code, params["comp_id"], params["bench_lang"], params["bench_mode"] == BenchMode.MonolithicPredict, params.get("bench_folds"), params.get("extended_schema"))

    common.log_results_and_exit(results)


if __name__ == "__main__":
    main()
