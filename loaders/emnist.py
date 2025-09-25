from typing import Any, Dict, TypedDict, Annotated
import numpy as np

from python.competition import *
from loaders.data_loader import DataLoader

class EMNISTTrain(TypedDict):
    images: Annotated[np.ndarray, 'Training images']
    labels: Annotated[np.ndarray, 'Training labels']

class EMNISTVal(TypedDict):
    images: Annotated[np.ndarray, 'Validation images with labels']

class Dataset(TypedDict):
    data: Annotated[EMNISTTrain, "Training features"]
    X_val: Annotated[EMNISTVal, "Validation features"]


class EMNISTDataLoader(DataLoader):
    """Data loader for EMNIST dataset in .npz format."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load EMNIST training data from .npz file."""
        dataset = {}
        train_path = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.npz")

        if os.path.exists(train_path):
            # Load the .npz file
            with np.load(train_path) as data:
                dataset['images'] = data['images']  # Shape: (n_samples, 28, 28)
                dataset['labels'] = data['labels']  # Shape: (n_samples,)
        else:
            raise FileNotFoundError(f"Train file not found: {train_path}")

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load EMNIST validation features."""
        dataset = {}
        val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.npz")

        if os.path.exists(val_path):
            with np.load(val_path) as data:
                dataset['images'] = data['images']  # Shape: (n_samples, 28, 28)
        else:
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> np.ndarray:
        """Load EMNIST validation labels."""
        y_val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.npz")

        if os.path.exists(y_val_path):
            with np.load(y_val_path) as data:
                return data['labels']
        else:
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")
