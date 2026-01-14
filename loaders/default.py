from typing import Any, Dict, TypedDict, Annotated
import pandas as pd

from python.competition import *
from .data_loader import DataLoader
from .utils import read_csv_smart


class Dataset(TypedDict):
    data: Annotated[pd.DataFrame, "Training data with target"]
    X_val: Annotated[pd.DataFrame, "Validation data without target"]


class DefaultDataLoader(DataLoader):
    """Default data loader with hardcoded paths"""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded path"""
        train_path = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv")

        if os.path.exists(train_path):
            dataset = read_csv_smart(train_path)
        else:
            raise ValueError(f"Train file not found: {train_path}")

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded path"""
        val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv")

        if os.path.exists(val_path):
            dataset = read_csv_smart(val_path)
        else:
            raise ValueError(f"Validation features file not found: {val_path}")

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded path"""
        y_val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv")

        if os.path.exists(y_val_path):
            return read_csv_smart(y_val_path)
        else:
            raise ValueError(f"Validation labels file not found: {y_val_path}")

    def load_grader_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load additional data (excluded columns) which is passed to the grader, returns an empty dict by default"""
        extra_train = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "excluded_cols_train.csv")
        extra_val = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "excluded_cols_val.csv")

        train_exists, val_exists = os.path.exists(extra_train), os.path.exists(extra_val)
        if not train_exists and not val_exists:
            return {}  # No extra files
        elif train_exists != val_exists:
            raise ValueError(f"Extra columns file exists for {'train' if train_exists else 'validation'}, but not for {'validation' if train_exists else 'train'}")
        else:
            return {"extra_train": read_csv_smart(extra_train), "extra_val": read_csv_smart(extra_val)}

