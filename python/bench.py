# This is the entrypoint file

import os
import importlib

import common
from grader import grade_llm_code


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
    return {"comp_id": comp_id, "bench_lang": bench_lang, "bench_mode": bench_mode}


# Loading the submission code
# ===========================
def load_submission() -> object:
    try:
        mod = importlib.import_module("submission.code")  # loads the file 'code.py'
        if not hasattr(mod, "train_and_predict"):
            common.report_error(
                "Submission code does not have the train_and_predict function"
            )
            common.graceful_exit(1)
        return mod.train_and_predict
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)


def main():
    common.bench_results.is_in_container = True

    params = get_bench_params()
    train_and_predict = load_submission()

    results = grade_llm_code(train_and_predict, params["comp_id"], params["bench_lang"])

    common.log_results_and_exit(results)


if __name__ == "__main__":
    main()
