from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, TypedDict, Annotated
import numpy as np
import pandas as pd

from python.competition import *
from loaders.data_loaders import DataLoader


class Dataset(TypedDict):
    data: Annotated[pd.DataFrame, "Training data with target"]
    X_val: Annotated[pd.DataFrame, "Validation data without target"]


class DefaultDataLoader(DataLoader):
    """Default data loader with hardcoded paths"""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded path"""
        dataset = {}
        train_path = os.path.join(base_path, "data", "folds", comp.comp_id, f"train_{fold_idx}.csv")

        if os.path.exists(train_path):
            dataset['data'] = pd.read_csv(train_path)
        else:
            raise ValueError(f"Train file not found: {train_path}")

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded path"""
        dataset = {}
        val_path = os.path.join(base_path, "data", "folds", comp.comp_id, f"X_val_{fold_idx}.csv")

        if os.path.exists(val_path):
            dataset['X_val'] = pd.read_csv(val_path)
        else:
            raise ValueError(f"Validation features file not found: {val_path}")

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded path"""
        y_val_path = os.path.join(base_path, "data", "validation", comp.comp_id, f"y_val_{fold_idx}.csv")

        if os.path.exists(y_val_path):
            return pd.read_csv(y_val_path)
        else:
            raise ValueError(f"Validation labels file not found: {y_val_path}")
