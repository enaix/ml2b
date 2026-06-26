import python.common as common

import os
import pandas as pd
import numpy as np
from importlib.metadata import version
from packaging.version import Version as SemVer

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri, default_converter
from rpy2.rinterface import SexpVector
if SemVer(version("rpy2")) < SemVer("2.6.1"):
    # legacy name
    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
else:
    from rpy2.robjects.packages import STAP



# =====================
# Foreignlang (R/Julia) module, handles code loading and data conversion
# TODO implement Python code execution prevention
# =====================


# Data conversion
# ===============

def pandas_to_r(df: pd.DataFrame) -> ro.vectors.DataFrame:
    with (default_converter + pandas2ri.converter + numpy2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
    return r_df

def r_to_numpy(r_vec: SexpVector) -> np.ndarray:
    return np.asarray(r_vec)

def r_to_pandas(r_df: ro.vectors.DataFrame) -> pd.DataFrame:
    with (default_converter + pandas2ri.converter + numpy2ri.converter).context():
        df = ro.conversion.get_conversion().rpy2py(r_df)
    return df


# Code execution
# ==============

def _load_r_submission() -> str:
    code_path = "submission/code.r"
    if not os.path.exists(code_path):
        raise FileNotFoundError(f"File not found: {code_path}")

    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()
    return code

def load_r_submission_mono() -> STAP:
    code = _load_r_submission()
    try:
        submission = STAP(code, "submission")
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)

    if "train_and_predict" not in submission:
        common.report_error("Submission code does not have the train_and_predict function")
        common.graceful_exit(1)

    return submission  # should be OK to be called as a dict

def load_r_submission_modular() -> STAP:
    code = _load_r_submission()
    try:
        submission = STAP(code, "submission")
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)

    for func in ["train", "prepare_val", "predict"]:
        if func not in submission:
            common.report_error(f"Submission code does not have the {func} function")
            common.graceful_exit(1)

    return submission  # should be OK to be called as a dict
