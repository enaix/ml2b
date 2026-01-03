from typing import TypedDict, Annotated
import os
import numpy as np
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader


class SheepClassificationTrain(TypedDict):
    filename: Annotated[pd.Series, 'Training image filenames']
    label: Annotated[pd.Series, 'Training labels']


class SheepClassificationVal(TypedDict):
    filename: Annotated[pd.Series, 'Validation image filenames']


class Dataset(TypedDict):
    data: Annotated[SheepClassificationTrain, "Training features"]
    X_val: Annotated[SheepClassificationVal, "Validation features"]


class SheepClassificationDataLoader(DataLoader):
    """Data loader for sheep classification challenge dataset in .csv format."""
    DEFAULT_SCHEMA = Dataset

    def _build_image_paths(self, filenames: pd.Series, comp: Competition) -> pd.Series:
        """Construct full image paths from filenames."""
        image_dir = comp.metadata.get("image_dir", "train")
        if not os.path.isabs(image_dir):
            image_dir = os.path.join(comp.comp_path, image_dir)
        return filenames.apply(lambda x: os.path.join(image_dir, x))

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Sheep Classification training data from .csv file."""
        train_path = os.path.join(
            base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")

        data = pd.read_csv(train_path)
        image_col = comp.metadata.get("image_col", "filename")
        target_col = comp.metadata.get("target_col", "label")

        if image_col not in data.columns:
            raise ValueError(f"Column '{image_col}' not found in training data. Available columns: {data.columns.tolist()}")
        if target_col not in data.columns:
            raise ValueError(f"Column '{target_col}' not found in training data. Available columns: {data.columns.tolist()}")

        dataset = {
            "filename": self._build_image_paths(data[image_col], comp),
            "label": data[target_col],
        }

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Sheep Classification validation features."""
        val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv"
        )

        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        data = pd.read_csv(val_path)
        image_col = comp.metadata.get("image_col", "filename")

        dataset = {
            "filename": self._build_image_paths(data[image_col], comp),
        }

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.Series:
        """Load Sheep Classification validation labels."""
        y_val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv"
        )

        if not os.path.exists(y_val_path):
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

        data = pd.read_csv(y_val_path)
        target_col = comp.metadata.get("target_col", "label")
        return data[target_col]

    @staticmethod
    def get_data_structure() -> Dict[str, str]:
        return {
            "filename": "Image filenames (n_samples,)",
            "label": "Training labels (n_samples,)"
        }