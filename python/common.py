# Common definitions
# ==================

import os
import sys
import json
import traceback


class Results:
    """
    Global class which stores and saves results to file.
    """

    res = {"errors": [], "tracebacks": [], "success": False}
    is_in_container = False

    def write(self):
        with open(os.path.join("submission", "results.json"), "w") as f:
            json.dump(self.res, f)


bench_results = Results()


def report_error(err: str):
    print(err)  # log to stdout
    bench_results.res["errors"].append(err)  # set result flag to the output file
    bench_results.res["tracebacks"].append(str(traceback.format_exc()))


def graceful_exit(status: int):
    if not bench_results.is_in_container:
        raise BaseException  # Allow top-level code to catch this

    bench_results.res["success"] = status == 0
    bench_results.write()  # write results to file
    sys.exit(status)


def set_bench_info(info: dict):
    bench_results.res = {**bench_results.res, **info}


def log_results_and_exit(results: dict):
    bench_results.res = {**bench_results.res, **results}
    graceful_exit(0)
