import pandas as pd
from charset_normalizer import from_path

def read_csv_smart(path: str, **kwargs) -> pd.DataFrame:
    if "encoding" not in kwargs:
        result = from_path(path).best()
        encoding = result.encoding if result else "utf-8"
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep=";", **kwargs)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding, **kwargs)