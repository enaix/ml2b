from typing import TypedDict, Annotated, List
import os
import numpy as np
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader


class PhotoClassificationTrain(TypedDict):
    image: Annotated[pd.Series, 'Training image paths']
    labels: Annotated[List[List[int]], 'Training multi-label targets']
    caption: Annotated[pd.Series, 'Training captions']


class PhotoClassificationVal(TypedDict):
    image: Annotated[pd.Series, 'Validation image paths']
    caption: Annotated[pd.Series, 'Validation captions']


class Dataset(TypedDict):
    data: Annotated[PhotoClassificationTrain, "Training features"]
    X_val: Annotated[PhotoClassificationVal, "Validation features"]


class PhotoClassificationDataLoader(DataLoader):
    """Data loader for multi-label photo classification dataset in .csv format."""
    DEFAULT_SCHEMA = Dataset

    def _parse_labels(self, labels: pd.Series) -> List[List[int]]:
        """Parse whitespace-separated label strings into lists of ints."""
        return labels.fillna("").apply(
            lambda x: list(map(int, x.split())) if x else []
        ).tolist()

    def _build_image_paths(self, image_ids: pd.Series, comp: Competition) -> pd.Series:
        """Construct full image paths from ImageID."""
        image_dir = comp.metadata.get("image_dir", "data")
        return image_ids.apply(lambda x: os.path.join(image_dir, x))

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Photo Classification training data."""
        train_path = os.path.join(
            base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")

        data = pd.read_csv(train_path)

        image_col = comp.metadata.get("image_col", "ImageID")
        labels_col = comp.metadata.get("labels_col", "Labels")
        caption_col = comp.metadata.get("caption_col", "Caption")

        dataset = {
            "image": self._build_image_paths(data[image_col], comp),
            "labels": self._parse_labels(data[labels_col]),
            "caption": data[caption_col] if caption_col in data.columns else None,
        }

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load Photo Classification validation features (no labels)."""
        val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv"
        )

        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        data = pd.read_csv(val_path)

        image_col = comp.metadata.get("image_col", "ImageID")
        caption_col = comp.metadata.get("caption_col", "Caption")

        dataset = {
            "image": self._build_image_paths(data[image_col], comp),
            "caption": data[caption_col] if caption_col in data.columns else None,
        }

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> List[List[int]]:
        """Load Photo Classification validation labels (multi-label)."""
        y_val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv"
        )

        if not os.path.exists(y_val_path):
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

        data = pd.read_csv(y_val_path)

        labels_col = comp.metadata.get("labels_col", "Labels")
        return self._parse_labels(data[labels_col])

    @staticmethod
    def get_data_structure() -> Dict[str, str]:
        return {
            "image": "Image paths (n_samples,)",
            "labels": "Multi-label targets (List[List[int]])",
            "caption": "Optional text captions (n_samples,)"
        }