from typing import TypedDict, Annotated
import numpy as np
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader


class ClassifyLeavesTrain(TypedDict):
    image: Annotated[np.ndarray, 'Training images']
    label: Annotated[np.ndarray, 'Training labels']


class ClassifyLeavesVal(TypedDict):
    image: Annotated[np.ndarray, 'Validation images with labels']


class Dataset(TypedDict):
    data: Annotated[ClassifyLeavesTrain, "Training features"]
    X_val: Annotated[ClassifyLeavesVal, "Validation features"]


class ClassifyLeavesDataLoader(DataLoader):
    """Data loader for Classify Leaves dataset in .csv format."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Classify Leaves training data from .csv file."""
        dataset = {}
        train_path = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv")

        if os.path.exists(train_path):
            # Load the .csv file
            data = pd.read_csv(train_path)
            dataset['image'] = data['image']  # Shape: (n_samples, 28, 28)
            dataset['label'] = data['label']  # Shape: (n_samples,)
        else:
            raise FileNotFoundError(f"Train file not found: {train_path}")

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Classify Leaves validation features."""
        dataset = {}
        val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv")

        if os.path.exists(val_path):
            data = pd.read_csv(val_path)
            dataset['image'] = data['image']  # Shape: (n_samples, 28, 28)
        else:
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> np.ndarray:
        """Load Classify Leaves validation labels."""
        y_val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv")

        if os.path.exists(y_val_path):
            data = pd.read_csv(y_val_path)
            return data['label']
        else:
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

    @staticmethod
    def get_data_structure() -> Dict[str, str]:
        return {
            'image': 'Training images (n_samples, 28, 28)',
            'label': 'Training labels (n_samples,)'
        }