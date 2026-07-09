from typing import Any, Dict, TypedDict, Annotated
import os
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader
from loaders.utils import read_csv_smart


class HealthQATrain(TypedDict):
    input: Annotated[pd.Series, "Health questions"]
    output: Annotated[pd.Series, "Reference answers"]


class HealthQAVal(TypedDict):
    ID: Annotated[pd.Series, "Question IDs"]
    input: Annotated[pd.Series, "Health questions"]
    subset: Annotated[pd.Series, "Language-country subset"]


class Dataset(TypedDict):
    data: Annotated[HealthQATrain, "Training question-answer pairs"]
    X_val: Annotated[HealthQAVal, "Validation questions without answers"]


class HealthQADataLoader(DataLoader):
    """Data loader for multilingual health QA (seq2seq text generation)."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        train_path = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")

        data = read_csv_smart(train_path)
        input_col = comp.metadata.get("input_col", "input")
        target_col = comp.metadata.get("target_col", "output")

        for col in (input_col, target_col):
            if col not in data.columns:
                raise ValueError(
                    f"Column '{col}' not found in training data. "
                    f"Available columns: {data.columns.tolist()}"
                )

        return {"input": data[input_col], "output": data[target_col]}

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv"
        )
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        data = read_csv_smart(val_path)
        input_col = comp.metadata.get("input_col", "input")
        id_col = comp.metadata.get("id_col", "ID")
        subset_col = comp.metadata.get("subset_col", "subset")

        if input_col not in data.columns:
            raise ValueError(
                f"Column '{input_col}' not found in validation data. "
                f"Available columns: {data.columns.tolist()}"
            )

        dataset: Dict[str, Any] = {"input": data[input_col]}
        if id_col in data.columns:
            dataset["ID"] = data[id_col]
        if subset_col in data.columns:
            dataset["subset"] = data[subset_col]
        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.Series:
        y_val_path = os.path.join(
            base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv"
        )
        if not os.path.exists(y_val_path):
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

        data = read_csv_smart(y_val_path)
        target_col = comp.metadata.get("target_col", "output")
        if target_col not in data.columns:
            raise ValueError(
                f"Column '{target_col}' not found in validation labels. "
                f"Available columns: {data.columns.tolist()}"
            )
        return data[target_col]
