import python.common as common

import os
import gc
import pandas as pd
import numpy as np
import pyarrow as pa
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

from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import AnyValue, VectorValue
from juliacall import convert as jl_convert



# =====================
# Foreignlang (R/Julia) module, handles code loading and data conversion
# Should always be imported as "from python.foreignlang import *"
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

# ---

def pandas_to_julia(df: pd.DataFrame) -> AnyValue:
    # Basic translation (erron-prone):
    # return jl.DataFrame(**{col: df[col].to_numpy() for col in df.columns})

    # pd -> pyarrow -> arrow.jl -> DataFrame.jl bridge (high memory usage):
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    # deallocate pyarrow table
    del table
    arrow_bytes = sink.getvalue().to_pybytes()
    del sink
    jl_bytes = jl_convert(jl.Vector[jl.UInt8], arrow_bytes)
    del arrow_bytes
    arrow_table = jl.Arrow.Table(jl.IOBuffer(jl_bytes))
    # TODO check that this conversion is correct for NaN, datetime, etc
    jl_df = jl.DataFrame(arrow_table)
    del arrow_table
    del jl_bytes
    return jl_df  # probably need to call gc.collect()

# def julia_to_numpy(jl_vec: VectorValue) -> np.ndarray:
#     return np.asarray(jl_vec)

def julia_to_pandas(jl_df: AnyValue) -> pd.DataFrame:
    # Basic translation (erron-prone):
    # col_names = [str(n) for n in jl.names(jl_df)]
    # return pd.DataFrame({col: np.asarray(getattr(jl_df, col)) for col in col_names})

    # DataFrame.jl -> arrow.jl -> pyarrow -> pd bridge (high memory usage):
    arrow_bytes = bytes(jl._df_to_arrow_bytes(jl_df))
    buf = pa.py_buffer(arrow_bytes)
    return pa.ipc.open_stream(buf).read_pandas()

def julia_dealloc_data() -> None:
    """Deallocate the input dataframe. Dataframes must be explicitly deallocated using del ..."""
    gc.collect()
    jl.seval("GC.gc(false)")

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

# ---

def _load_julia_submission() -> None:
    if not os.path.exists("submission/code.jl"):
        raise FileNotFoundError(f"No Julia submission module code at submission/code.jl")
    if not os.path.exists("submission/Project.toml"):
        raise FileNotFoundError(f"No Julia submission module config at submission/Project.toml")

    # We need to use the global jl namespace in order to overload pythoncall
    try:
        jlPkg.activate("submission")

        # Julia packages config begin
        jl.seval("using DataFrames")
        jl.seval("using Arrow")
        jl.seval("""
function _df_to_arrow_bytes(df::DataFrame)::Vector{UInt8}
    buf = IOBuffer()
    Arrow.write(buf, df)
    take!(buf)
end
""")
        # Julia packages config end

        jl.seval("using submission")
    except BaseException as e:
        common.report_error(f"Could not load submission code: {e=}")
        common.graceful_exit(1)

def load_julia_submission_mono() -> dict:
    _load_julia_submission()  # loaded in the global namespace

    if not hasattr(jl, "train_and_predict"):
        common.report_error("Submission code does not have the train_and_predict function")
        common.graceful_exit(1)

    return {"train_and_predict": jl.train_and_predict}

def load_julia_submission_modular() -> dict:
    _load_julia_submission()  # loaded in the global namespace

    funcs = {}
    for func in ["train", "prepare_val", "predict"]:
        if not hasattr(jl, func):
            common.report_error(f"Submission code does not have the {func} function")
            common.graceful_exit(1)
        funcs[func] = getattr(jl, func)
    return funcs
