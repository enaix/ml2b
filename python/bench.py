# This is the entrypoint file

import os
import importlib
from typing import Any
from enum import StrEnum

import common
from code_grader import grade_llm_code



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

    return {"comp_id": comp_id, "bench_lang": bench_lang, "bench_mode": mode}


# Loading the submission code
# ===========================
def load_mono_submission(params) -> dict:
    try:
        mod = importlib.import_module("submission.code")  # loads the file 'code.py'
        if not hasattr(mod, "train_and_predict"):
            common.report_error(
                "Submission code does not have the train_and_predict function"
            )
            common.graceful_exit(1)
        return {"train_and_predict": mod.train_and_predict}
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)


def load_modular_submission(params) -> dict:
    try:
        mod = importlib.import_module("submission.code")  # loads the file 'code.py'
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
    common.bench_results.is_in_container = True

    params = get_bench_params()
    if params.mode == BenchMode.MonolithicPredict:
        train_code = load_mono_submission()
    else:
        train_code = load_modular_submission()

    results = grade_llm_code(train_code, params["comp_id"], params["bench_lang"], params.mode == BenchMode.MonolithicPredict)

    common.log_results_and_exit(results)


if __name__ == "__main__":
    main()
