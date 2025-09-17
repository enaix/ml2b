from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, TypedDict, Annotated
import numpy as np
import pandas as pd

from python.competition import *
from loaders.data_loaders import DataLoader


class BikersData(TypedDict):
    bikers: Annotated[pd.DataFrame, 'Biker demographic information (training-filtered)']
    tours: Annotated[pd.DataFrame, 'Tour features and word counts (training-filtered)']
    tour_convoy: Annotated[pd.DataFrame, 'Tour participation lists (training-filtered)']
    bikers_network: Annotated[pd.DataFrame, 'Social network connections (training-filtered)']

class BikersTrain(BikersData):
    train: Annotated[pd.DataFrame, 'Training interactions with like/dislike labels']

class BikersVal(BikersData):
    X_val: Annotated[pd.DataFrame, 'Validation features without labels']

class Dataset(TypedDict):
    data: Annotated[BikersTrain, 'Training features']
    X_val: Annotated[BikersVal, 'Validation features']


class BikerRecommenderDataLoader(DataLoader):
    """Data loader for biker tour recommendation system with multiple tables."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> BikersTrain:
        """Load training data with training-filtered meta tables."""
        dataset = {}
        fold_dir = os.path.join(base_path, "data", "folds", comp.comp_id, f"fold_{fold_idx}")

        # Load main training data
        train_path = os.path.join(fold_dir, f"train.csv")
        if os.path.exists(train_path):
            dataset['train'] = pd.read_csv(train_path)

        # Load training-filtered meta tables
        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            meta_path = os.path.join(fold_dir, f"{table}_train.csv")
            if os.path.exists(meta_path):
                dataset[table] = pd.read_csv(meta_path)
                dataset[table] = self._parse_table_specific_columns(dataset[table], table)

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> BikersVal:
        """Load validation features with validation-filtered meta tables."""
        dataset = {}
        fold_dir = os.path.join(base_path, "data", "folds", comp.comp_id, f"fold_{fold_idx}")

        # Load validation features
        x_val_path = os.path.join(fold_dir, f"X_val.csv")
        if os.path.exists(x_val_path):
            dataset['X_val'] = pd.read_csv(x_val_path)

        # Load validation-filtered meta tables
        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            meta_path = os.path.join(fold_dir, f"{table}_val.csv")
            if os.path.exists(meta_path):
                dataset[table] = pd.read_csv(meta_path)
                dataset[table] = self._parse_table_specific_columns(dataset[table], table)

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from private directory."""
        y_val_path = os.path.join(base_path, "data", "validation", comp.comp_id, f"fold_{fold_idx}", f"y_val.csv")

        if os.path.exists(y_val_path):
            return pd.read_csv(y_val_path)
        else:
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

    def _parse_table_specific_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Parse space-delimited columns for specific tables."""
        if table_name == 'tour_convoy':
            for col in ['going', 'maybe', 'invited', 'not_going']:
                if col in df.columns:
                    df[col] = df[col].apply(self._parse_space_delimited)
        elif table_name == 'bikers_network':
            if 'friends' in df.columns:
                df['friends'] = df['friends'].apply(self._parse_space_delimited)
        return df

    def _parse_space_delimited(self, text):
        """Parse space-delimited strings into lists of integers."""
        if pd.isna(text) or text == '':
            return []
        return [int(x) for x in str(text).split() if x.strip().isdigit()]
