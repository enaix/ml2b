from typing import TypedDict, Annotated
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader


class BikersData(TypedDict):
    bikers: Annotated[pd.DataFrame, 'Biker demographic information']
    tours: Annotated[pd.DataFrame, 'Tour features and word counts']
    tour_convoy: Annotated[pd.DataFrame, 'Tour participation lists']
    bikers_network: Annotated[pd.DataFrame, 'Social network connections']

class BikersTrain(BikersData):
    data: Annotated[pd.DataFrame, 'Train data']

class BikersVal(BikersData):
    data: Annotated[pd.DataFrame, 'Validation data']

class Dataset(TypedDict):
    X_train: Annotated[BikersTrain, 'Training features']
    X_val: Annotated[BikersVal, 'Validation features']


class BikerRecommenderDataLoader(DataLoader):
    """Data loader for biker tour recommendation system with multiple tables."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> BikersData:
        """Load training data with training-filtered meta tables."""
        dataset = {}
        fold_dir = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}")

        # Load main training data
        train_path = os.path.join(fold_dir, "train.csv")

        # Load training-filtered meta tables
        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            meta_path = os.path.join(fold_dir, f"{table}_train.csv")
            if os.path.exists(meta_path):
                dataset[table] = pd.read_csv(meta_path)
                # dataset[table] = self._parse_table_specific_columns(dataset[table], table)
        if os.path.exists(train_path):
            dataset['data'] = pd.read_csv(train_path)

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> BikersData:
        """Load validation features with validation-filtered meta tables."""
        dataset = {}
        fold_dir = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}")

        # Load validation features
        x_val_path = os.path.join(fold_dir, "X_val.csv")

        # Load validation-filtered meta tables
        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            meta_path = os.path.join(fold_dir, f"{table}_val.csv")
            if os.path.exists(meta_path):
                dataset[table] = pd.read_csv(meta_path)
                # dataset[table] = self._parse_table_specific_columns(dataset[table], table)
        if os.path.exists(x_val_path):
            dataset['data'] = pd.read_csv(x_val_path)

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from private directory."""
        y_val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv")

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
