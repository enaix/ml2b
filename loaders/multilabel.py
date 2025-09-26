from typing import Any, Dict, List, TypedDict, Annotated
import pandas as pd

from python.competition import *
from loaders.data_loader import DataLoader


class Dataset(TypedDict):
    data: Annotated[pd.DataFrame, 'Training data with parsed into lists multi-label genres']
    X_val: Annotated[pd.DataFrame, 'Validation features']


class MultiLabelDataLoader(DataLoader):
    """Data loader for multi-label CSV competitions where labels are stored as strings."""
    DEFAULT_SCHEMA = Dataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data and parse multi-label strings into actual lists."""
        dataset = {}
        train_path = os.path.join(base_path, "folds", comp.comp_id, f"fold_{fold_idx}", "train.csv")

        if os.path.exists(train_path):
            data = pd.read_csv(train_path)

            # Parse multi-label strings for the target column
            target_col = comp.metadata.get("target_col")
            if target_col and target_col in data.columns:
                data[target_col] = data[target_col].apply(self._parse_multi_label_string)

            dataset = data
        else:
            raise FileNotFoundError(f"Train file not found: {train_path}")

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features - no parsing needed for features."""
        dataset = {}
        val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "X_val.csv")

        if os.path.exists(val_path):
            dataset = pd.read_csv(val_path)
        else:
            raise FileNotFoundError(f"Validation features file not found: {val_path}")

        return dataset

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> List[List[str]]:
        """Load validation labels and parse multi-label strings."""
        y_val_path = os.path.join(base_path, "validation", comp.comp_id, f"fold_{fold_idx}", "y_val.csv")

        if os.path.exists(y_val_path):
            y_val_df = pd.read_csv(y_val_path)
            target_col = comp.metadata.get("target_col", "genres")

            if target_col in y_val_df.columns:
                # Parse the multi-label strings
                parsed_labels = y_val_df[target_col].apply(self._parse_multi_label_string)
                return parsed_labels
            else:
                # Assume the first column contains the labels
                parsed_labels = y_val_df.iloc[:, 0].apply(self._parse_multi_label_string).tolist()
                return parsed_labels
        else:
            raise FileNotFoundError(f"Validation labels file not found: {y_val_path}")

    def _parse_multi_label_string(self, label_str):
        """Convert string representation like "[u'drama', u'comedy']" to actual list of strings."""
        if isinstance(label_str, list):
            return [str(item) for item in label_str]  # Ensure all items are strings

        if not isinstance(label_str, str):
            return [str(label_str)]  # Single value converted to list

        if label_str.startswith('[') and label_str.endswith(']'):
            try:
                # Try to parse as Python list using ast.literal_eval
                import ast
                parsed = ast.literal_eval(label_str)
                return parsed
            except (ValueError, SyntaxError):
                # Fallback: simple string parsing for malformed representations
                cleaned = label_str.strip('[]')
                # Handle various quote styles: u'genre', "genre", 'genre'
                cleaned = cleaned.replace("u'", "").replace("'", "").replace('"', '')
                items = [item.strip() for item in cleaned.split(',') if item.strip()]
                return items if items else [label_str.strip()]

        # Single label as string - return as list with one element
        return [label_str.strip()]
