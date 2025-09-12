from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd

from .competition import *



class DataLoader(ABC):
    """Abstract base class for competition data loading strategies"""

    @abstractmethod
    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded fold directory"""
        pass

    @abstractmethod
    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded fold directory"""
        pass

    @abstractmethod
    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded private directory"""
        pass

    @abstractmethod
    def get_data_structure(self) -> Dict[str, str]:
        """Return the expected data structure for this loader"""
        pass


class DefaultDataLoader(DataLoader):
    """Default data loader with hardcoded paths"""

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

    def get_data_structure(self) -> Dict[str, str]:
        return {
            'data': 'Training data with features and target',
            'X_val': 'Validation features without target',
        }


# Registry of data loaders
DATA_LOADERS = {
    "default": DefaultDataLoader,
}
