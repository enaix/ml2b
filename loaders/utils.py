import pandas as pd
import chardet

def read_csv_smart(path: str, **kwargs) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read(100_000)
        result = chardet.detect(raw)
        encoding = result["encoding"]
    try:
        return pd.read_csv(path, encoding=encoding, **kwargs)
    except BaseException:
        return pd.read_csv(path, encoding=encoding, sep=";", **kwargs)