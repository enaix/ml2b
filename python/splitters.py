from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split

from python.competition import *
from loaders import read_csv_smart


class DataSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        self.log_error = log_error
        self.do_shutdown = do_shutdown
        self.grading_stage = grading_stage

    @abstractmethod
    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[Any, Any]]:
        """Return list of (train_indices, val_indices) tuples"""
        pass

    @abstractmethod
    def prepare_fold_data(self, comp: Competition, train_indices: Any, val_indices: Any,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare data for a specific fold. Returns (train_path, val_path, additional_files)"""
        pass

    def prepare_competition_files(self, comp: Competition) -> None:
        if comp.files is not None:
            # Already populated
            return

        comp.set_files(self._initialize_files(comp))
        if not self.grading_stage:
            self._validate_files(comp)

    def _initialize_files(self, comp: Competition) -> Dict[str, CompetitionFile]:
        """Initialize competition files from metadata or by discovery"""
        files = {}

        # Get file configuration from metadata
        file_config = comp.metadata.get("files", {
            "train": {"type": "data", "required": True, "extensions": [".csv"]},
        })

        # First, try to use explicit file mapping from metadata
        explicit_files = comp.metadata.get("file_mapping", {})
        for file_key, file_info in explicit_files.items():
            file_path = os.path.join(comp.comp_path, file_info["filename"])
            files[file_key] = CompetitionFile(
                name=file_key,
                path=file_path,
                file_type=file_info.get("type", "data"),
                required=file_info.get("required", True)
            )

        # If no explicit mapping, discover files based on configuration
        if not explicit_files:
            for file_key, config in file_config.items():
                file_found = False
                for ext in config.get("extensions", [".csv"]):
                    potential_path = os.path.join(comp.comp_path, f"{file_key}{ext}")

                    # Only check existence if we're not in grading stage
                    if not self.grading_stage:
                        file_exists = os.path.exists(potential_path)
                        file_exists = True

                    if file_exists:
                        files[file_key] = CompetitionFile(
                            name=file_key,
                            path=potential_path,
                            file_type=config.get("type", "data"),
                            required=config.get("required", True)
                        )
                        file_found = True
                        break

                # If required file not found and we're not in grading stage, create entry for validation
                if not self.grading_stage and not file_found and config.get("required", True):
                    potential_path = os.path.join(comp.comp_path, f"{file_key}.csv")
                    files[file_key] = CompetitionFile(
                        name=file_key,
                        path=potential_path,
                        file_type=config.get("type", "data"),
                        required=True
                    )

        # Discover additional files
        if os.path.exists(comp.comp_path):
            for item in os.listdir(comp.comp_path):
                item_path = os.path.join(comp.comp_path, item)

                # Skip if we're not in grading stage and path doesn't exist
                if not os.path.exists(item_path):
                    continue

                if os.path.isfile(item_path):
                    filename = item
                    if self._is_submission_file(filename):
                        continue

                    file_key = os.path.splitext(filename)[0]
                    if file_key not in files:
                        file_type = self._infer_file_type(filename)
                        if file_type in ["data", "metadata"]:
                            files[file_key] = CompetitionFile(
                                name=file_key,
                                path=item_path,
                                file_type=file_type,
                                required=False
                            )

                elif os.path.isdir(item_path):
                    dir_name = item
                    if not self._is_submission_dir(dir_name):
                        files[dir_name] = CompetitionFile(
                            name=dir_name,
                            path=item_path,
                            file_type="data",
                            required=False
                        )

        return files

    def _is_submission_dir(self, dirname: str) -> bool:
        """Check if a directory is submission-related"""
        dirname_lower = dirname.lower()
        submission_keywords = ["submission", "sample", "baseline", "example"]
        return any(keyword in dirname_lower for keyword in submission_keywords)

    def _is_submission_file(self, filename: str) -> bool:
        """Check if a file is submission-related"""
        filename_lower = filename.lower()
        submission_keywords = [
            "sample_submission", "samplesubmission", "submission",
            "submit", "example_submission", "baseline"
        ]
        return any(keyword in filename_lower for keyword in submission_keywords)

    def _infer_file_type(self, filename: str) -> str:
        """Infer file type from filename"""
        filename_lower = filename.lower()

        if filename_lower.endswith(('.csv', '.json', '.parquet', '.pkl', '.pickle', '.h5', '.hdf5')):
            return "data"
        elif filename_lower.endswith(('.txt', '.md', '.json', '.xml', '.yml', '.yaml')):
            if any(word in filename_lower for word in ["description", "readme", "info", "meta"]):
                return "metadata"
            else:
                return "data"
        elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif')):
            return "data"
        elif filename_lower.endswith(('.npy', '.npz', '.mat')):
            return "data"
        else:
            return "other"

    def _validate_files(self, comp: Competition) -> None:
        """Validate that all required files exist"""
        missing_files = []
        for file_key, comp_file in comp.files.items():
            if comp_file.required and not comp_file.exists():
                missing_files.append(f"{file_key} ({comp_file.path})")

        if missing_files:
            self.log_error(f"Competition {comp.comp_id}: missing required files: {', '.join(missing_files)}")
            self.do_shutdown(1)



class CSVDataSplitter(DataSplitter):
    """Standard CSV data splitter using train_test_split with 80:20 ratio"""

    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data using train_test_split with 80:20 ratio"""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        # TODO implement multiple target_col

        train_df = read_csv_smart(train_file.path)
        target_col = comp.metadata.get("target_col")
        if not target_col:
            raise ValueError(f"No target_col specified in metadata for competition {comp.comp_id}")

        X = train_df.drop(columns=target_col)
        y = train_df[target_col]

        # Use train_test_split for a single 80:20 split
        if n_splits != 1:
            self.log_error(f"Warning: train_test_split only supports 1 split (80:20). Using n_splits=1 instead of {n_splits}")

        # Perform the 80:20 split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=y if comp.metadata.get("stratified_split", False) else None
        )

        # Convert to indices for compatibility
        train_indices = X_train.index.values
        val_indices = X_val.index.values

        return [(train_indices, val_indices)]

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare fold data and save to appropriate directories"""
        train_file = comp.get_file("train")
        train_df = read_csv_smart(train_file.path)
        target_col = comp.metadata["target_col"]

        # Split data using the provided indices
        X, y = train_df.drop(columns=target_col), train_df[target_col]
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]

        # Save fold data
        train_fold = pd.concat([X_train, y_train], axis=1)
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(exist_ok=True, parents=True)

        train_fold.to_csv(train_path / "train.csv", index=False)
        X_val.to_csv(val_path / "X_val.csv", index=False)
        y_val.to_csv(val_path / "y_val.csv", index=False)

        return train_path, val_path, {}


class MultilabelDataSplitter(DataSplitter):
    """Standard multilabel CSV data splitter using train_test_split with 80:20 ratio"""

    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data using train_test_split with 80:20 ratio"""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        # TODO implement multiple target_col

        train_df = read_csv_smart(train_file.path)
        target_col = comp.metadata.get("target_col")
        if not target_col:
            raise ValueError(f"No target_col specified in metadata for competition {comp.comp_id}")

        X = train_df.drop(columns=[target_col])
        y = train_df[target_col]

        # Use train_test_split for a single 80:20 split
        if n_splits != 1:
            self.log_error(f"Warning: train_test_split only supports 1 split (80:20). Using n_splits=1 instead of {n_splits}")

        # Perform the 80:20 split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=y if comp.metadata.get("stratified_split", False) else None
        )

        # Convert to indices for compatibility
        train_indices = X_train.index.values
        val_indices = X_val.index.values

        return [(train_indices, val_indices)]

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare fold data and save to appropriate directories"""
        train_file = comp.get_file("train")
        train_df = read_csv_smart(train_file.path)
        target_col = comp.metadata["target_col"]

        # Split data using the provided indices
        X, y = train_df.drop(columns=[target_col]), train_df[target_col]
        y = y.apply(self._parse_multi_label_string)
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]

        # Save fold data
        train_fold = pd.concat([X_train, y_train], axis=1)
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(exist_ok=True, parents=True)

        train_fold.to_csv(train_path / "train.csv", index=False)
        X_val.to_csv(val_path / "X_val.csv", index=False)
        y_val.to_csv(val_path / "y_val.csv", index=False)

        return train_path, val_path, {}

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


class ImageFolderDataSplitter(DataSplitter):
    """Data splitter for image classification with folder structure"""

    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[List[str], List[str]]]:
        """Split image data using folder structure"""
        train_images_dir = None
        train_csv_path = None


        self.prepare_competition_files(comp)

        # Look for train images directory or CSV
        for file_key, comp_file in comp.get_all_files().items():
            if "train" in file_key.lower() and os.path.isdir(comp_file.path):
                train_images_dir = comp_file.path
            elif file_key == "train" and comp_file.path.endswith('.csv'):
                train_csv_path = comp_file.path

        if not train_images_dir and not train_csv_path:
            raise FileNotFoundError(f"No train images directory or CSV found for competition {comp.comp_id}")

        # Strategy 1: CSV with image paths and labels
        if train_csv_path and os.path.exists(train_csv_path):
            df = pd.read_csv(train_csv_path)
            image_col = comp.metadata.get("image_col", "image")
            target_col = comp.metadata.get("target_col", "label")

            if image_col not in df.columns or target_col not in df.columns:
                image_col, target_col = df.columns[0], df.columns[1]

            images = df[image_col].tolist()
            labels = df[target_col].tolist()

            # Use stratified split for classification
            if comp.metadata.get("stratified_split", True):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                return [(
                    [images[i] for i in train_idx],
                    [images[i] for i in val_idx]
                ) for train_idx, val_idx in skf.split(images, labels)]
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                return [(
                    [images[i] for i in train_idx],
                    [images[i] for i in val_idx]
                ) for train_idx, val_idx in kf.split(images)]

        # Strategy 2: Folder structure (class folders)
        elif train_images_dir:
            all_images = []
            all_labels = []

            # Walk through class directories
            for class_name in os.listdir(train_images_dir):
                class_dir = os.path.join(train_images_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                            all_images.append(os.path.join(class_dir, img_file))
                            all_labels.append(class_name)

            # Stratified split by default for image classification
            if comp.metadata.get("stratified_split", True):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                return list(skf.split(all_images, all_labels))
            else:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                return list(kf.split(all_images))

        raise ValueError(f"Could not determine image data structure for competition {comp.comp_id}")

    def prepare_fold_data(self, comp: Competition, train_indices: List[str], val_indices: List[str],
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare image fold data with directory structure"""
        # Create fold-specific directories
        fold_train_dir = Path(fold_dir) / f"fold_{fold_idx}" / "train_images"
        fold_val_dir = Path(private_dir) / f"fold_{fold_idx}" / "val_images"
        fold_train_dir.mkdir(exist_ok=True, parents=True)
        fold_val_dir.mkdir(exist_ok=True, parents=True)

        # Prepare training data
        train_data = []
        for idx in train_indices:
            if isinstance(idx, str):
                img_path = idx
                label = self._extract_label_from_path(img_path)

                # Copy image to fold directory
                img_name = os.path.basename(img_path)
                fold_img_path = os.path.join(fold_train_dir, img_name)
                shutil.copy2(img_path, fold_img_path)

                train_data.append({
                    'image': os.path.relpath(fold_img_path, fold_dir),
                    'label': label
                })

        # Prepare validation data
        val_data = []
        val_labels = []
        for idx in val_indices:
            if isinstance(idx, str):
                img_path = idx
                label = self._extract_label_from_path(img_path)

                img_name = os.path.basename(img_path)
                fold_img_path = os.path.join(fold_val_dir, img_name)
                shutil.copy2(img_path, fold_img_path)

                val_data.append({
                    'image': os.path.relpath(fold_img_path, fold_dir)
                })
                val_labels.append({'label': label})

        # Save CSV files
        pd.DataFrame(train_data).to_csv(fold_train_dir.parent / "train.csv", index=False)
        pd.DataFrame(val_data).to_csv(fold_val_dir.parent / "X_val.csv", index=False)
        pd.DataFrame(val_labels).to_csv(fold_val_dir.parent / "y_val.csv", index=False)

        additional_files = {
            'train_images': fold_train_dir,
            'val_images': fold_val_dir
        }

        return fold_train_dir.parent, fold_val_dir.parent, additional_files

    def _extract_label_from_path(self, img_path: str) -> str:
        """Extract label from image path (assumes parent directory is class name)"""
        return os.path.basename(os.path.dirname(img_path))


class RecommendationDataSplitter(DataSplitter):
    """Custom data splitter for recommendation systems"""

    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data by users to avoid data leakage"""

        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        train_df = pd.read_csv(train_file.path)

        # Check if this is a recommendation system
        required_cols = ['biker_id', 'tour_id']
        if not all(col in train_df.columns for col in required_cols):
            self.log_error(f"Warning: Expected columns {required_cols} for recommendation splitting")
            self.log_error("Falling back to regular KFold splitting")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return list(kf.split(train_df))

        # Use GroupKFold to ensure each user is in only one fold
        group_kf = GroupKFold(n_splits=n_splits)
        dummy_X = np.arange(len(train_df))
        groups = train_df['biker_id'].values

        return list(group_kf.split(dummy_X, groups=groups))

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare recommendation data for a specific fold"""
        train_file = comp.get_file("train")
        train_df = pd.read_csv(train_file.path)

        # Split by indices
        train_fold = train_df.iloc[train_indices].copy()
        val_fold = train_df.iloc(val_indices).copy()

        # Save training data
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        train_fold.to_csv(train_path / "train.csv", index=False)

        # For validation: create X_val and y_val
        target_col = comp.metadata.get("target_col", "like")

        if target_col in val_fold.columns:
            target_cols_to_drop = [target_col]
            if target_col == "like" and "dislike" in val_fold.columns:
                target_cols_to_drop.append("dislike")
            elif target_col == "dislike" and "like" in val_fold.columns:
                target_cols_to_drop.append("like")

            val_features = val_fold.drop(columns=target_cols_to_drop).copy()
            y_val_cols = ['biker_id', 'tour_id'] + [col for col in target_cols_to_drop if col in val_fold.columns]
            val_labels = val_fold[y_val_cols].copy()
        else:
            val_features = val_fold.copy()
            val_labels = pd.DataFrame()

        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(exist_ok=True, parents=True)

        val_features.to_csv(val_path / "X_val.csv", index=False)
        val_labels.to_csv(val_path / "y_val.csv", index=False)

        return train_path, val_path, {}


class TimeSeriesDataSplitter(DataSplitter):
    """Time series data splitter that respects temporal order"""

    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split time series data maintaining temporal order"""

        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        train_df = pd.read_csv(train_file.path)

        # Get time column from metadata
        time_col = comp.metadata.get("time_col", "timestamp")
        if time_col not in train_df.columns:
            self.log_error(f"Warning: Time column '{time_col}' not found, falling back to regular split")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return list(kf.split(train_df))

        # Sort by time
        train_df = train_df.sort_values(time_col).reset_index(drop=True)

        # Time-based splitting
        splits = []
        total_len = len(train_df)

        # Use expanding window approach
        for i in range(n_splits):
            train_end = int(total_len * (i + 1) / (n_splits + 1))
            val_start = train_end
            val_end = min(int(total_len * (i + 2) / (n_splits + 1)), total_len)

            if val_start >= val_end:
                break

            train_indices = np.arange(train_end)
            val_indices = np.arange(val_start, val_end)
            splits.append((train_indices, val_indices))

        return splits

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare time series data maintaining temporal order"""
        train_file = comp.get_file("train")
        train_df = pd.read_csv(train_file.path)

        # Sort by time to ensure proper order
        time_col = comp.metadata.get("time_col", "timestamp")
        if time_col in train_df.columns:
            train_df = train_df.sort_values(time_col).reset_index(drop=True)

        # Get data for this fold
        train_data = train_df.iloc[train_indices].copy()
        val_data = train_df.iloc[val_indices].copy()

        # Split features and target
        target_col = comp.metadata.get("target_col")
        if target_col and target_col in val_data.columns:
            X_val = val_data.drop(columns=[target_col])
            y_val = val_data[[target_col]]
        else:
            X_val = val_data.copy()
            y_val = pd.DataFrame()

        # Save files
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)

        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(parents=True, exist_ok=True)

        train_data.to_csv(train_path / "train.csv", index=False)
        X_val.to_csv(val_path / "X_val.csv", index=False)
        y_val.to_csv(val_path / "y_val.csv", index=False)

        return train_path, val_path, {}


class CustomDataSplitter(DataSplitter):
    """Custom data splitter for specific competitions"""

    def __init__(self, split_function: Callable, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        self.split_function = split_function
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[Any, Any]]:

        self.prepare_competition_files(comp)
        return self.split_function(comp, n_splits)

    def prepare_fold_data(self, comp: Competition, train_indices: Any, val_indices: Any,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        return self.split_function.prepare_fold_data(comp, train_indices, val_indices, fold_idx, fold_dir, private_dir)


class EMNISTDataSplitter(DataSplitter):
    """Data splitter for EMNIST dataset using fixed 80:20 split with seed 42."""
    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split EMNIST data using fixed 80:20 split with stratification."""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        # Load the original .npz file
        with np.load(train_file.path) as data:
            images = data['images']
            labels = data['labels']

        # Use fixed 80:20 split with seed 42
        # For image classification, we should use stratified split to maintain class balance
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=labels
        )

        # Return as a list with one split (to maintain interface compatibility)
        return [(train_idx, val_idx)]

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare fold data for EMNIST dataset and save as .npz files."""
        train_file = comp.get_file("train")

        # Load the original data
        with np.load(train_file.path) as data:
            images = np.squeeze(data['images'], axis=-1)
            labels = data['labels']

        # Split the data using the provided indices
        train_images = images[train_indices]
        train_labels = labels[train_indices]
        val_images = images[val_indices]
        val_labels = labels[val_indices]

        # Save training data for this fold (always fold 0 since we only have one split)
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        np.savez(train_path / "train.npz", images=train_images, labels=train_labels)

        # Save validation features
        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(exist_ok=True, parents=True)
        np.savez(val_path / "X_val.npz", images=val_images)

        # Save validation labels (private)
        np.savez(val_path / "y_val.npz", labels=val_labels)

        return train_path, val_path, {}


class ClassifyLeavesDataSplitter(DataSplitter):
    """Data splitter for image classification dataset using fixed 80:20 split with seed 42."""
    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split Classify Leaves data using fixed 80:20 split with stratification."""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        # CSV with image paths and labels
        df = pd.read_csv(train_file.path)
        image_col = comp.metadata.get("image_col", "image")
        target_col = comp.metadata.get("target_col", "label")

        if image_col not in df.columns or target_col not in df.columns:
            image_col, target_col = df.columns[0], df.columns[1]

        images = df[image_col].tolist()
        labels = df[target_col].tolist()

        train_images, val_images, _, _ = train_test_split(images,
                                                          labels,
                                                          test_size=0.2,
                                                          stratify=labels if comp.metadata.get("stratified_split", True) else None,
                                                          random_state=42
        )
        return [(train_images, val_images)]

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare fold data for Classify Leaves dataset and save as .npz files."""
        train_file = comp.get_file("train")

        # Load the original data
        data = pd.read_csv(train_file.path)
        train_data = data.loc[data['image'].isin(train_indices)].copy()
        val_data = data.loc[data['image'].isin(val_indices)].copy()

        # Save training data for this fold (always fold 0 since we only have one split)
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        train_csv_path = os.path.join(train_path, "train.csv")
        pd.DataFrame(train_data).to_csv(train_csv_path, index=False)

        # Save validation features
        val_path = Path(private_dir) / f"fold_{fold_idx}"
        val_path.mkdir(exist_ok=True, parents=True)
        val_csv_path = os.path.join(val_path, "X_val.csv")
        pd.DataFrame(val_data.drop(columns=['label'])).to_csv(val_csv_path, index=False)

        # Save validation labels (private)
        val_labels_path = os.path.join(val_path, "y_val.csv")
        pd.DataFrame(val_data.drop(columns=['image'])).to_csv(val_labels_path, index=False)

        return train_path, val_path, {}

    def _extract_label_from_path(self, img_path: str) -> str:
        """Extract label from image path (assumes parent directory is class name)"""
        return os.path.basename(os.path.dirname(img_path))


class BikerRecommenderDataSplitter(DataSplitter):
    """Data splitter for biker tour recommendation system with multiple tables."""
    def __init__(self, log_error: Any, do_shutdown: Any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split recommender data using random splitting."""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError("Train file not found")

        train_df = pd.read_csv(train_file.path)
        train_indices, val_indices = train_test_split(
            train_df.index,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=None
        )
        return [(train_indices, val_indices)]

    def prepare_fold_data(self, comp: Competition, train_indices: np.ndarray, val_indices: np.ndarray,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare fold data with separate filtering for train and validation meta tables."""
        additional_files = {}

        # Load all original tables
        original_tables = {}
        tables = ['train', 'tour_convoy', 'bikers', 'tours', 'bikers_network']
        for table in tables:
            table_file = comp.get_file(table)
            if table_file and table_file.exists():
                original_tables[table] = pd.read_csv(table_file.path)

        # Get the main splits
        train_main = original_tables['train'].iloc[train_indices]
        val_main = original_tables['train'].iloc[val_indices]

        # Save main splits
        train_path = Path(fold_dir) / f"fold_{fold_idx}"
        train_path.mkdir(exist_ok=True, parents=True)
        val_path = Path(private_dir) / f"fold_{fold_idx}" 
        val_path.mkdir(exist_ok=True, parents=True)


        train_main.to_csv(train_path / "train.csv", index=False)
        val_main.drop(columns=['like', 'dislike']).to_csv(val_path / "X_val.csv", index=False)
        val_main[['biker_id', 'tour_id', 'like', 'dislike']].to_csv(val_path / "y_val.csv", index=False)

        # === SEPARATE FILTERING FOR TRAIN AND VALIDATION META TABLES ===

        # 1. FILTER FOR TRAINING META TABLES (only training entities)
        train_bikers = set(train_main['biker_id'].unique())
        train_tours = set(train_main['tour_id'].unique())

        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            if table in original_tables:
                train_meta = self._filter_meta_table(original_tables[table], table, train_bikers, train_tours)
                train_meta_path = train_path / f"{table}_train.csv"
                train_meta.to_csv(train_meta_path, index=False)
                additional_files[f"{table}_train"] = train_meta_path

        # 2. FILTER FOR VALIDATION META TABLES (only validation entities)
        val_bikers = set(val_main['biker_id'].unique())  # ONLY validation bikers
        val_tours = set(val_main['tour_id'].unique())    # ONLY validation tours

        for table in ['bikers', 'tours', 'tour_convoy', 'bikers_network']:
            if table in original_tables:
                val_meta = self._filter_meta_table(original_tables[table], table, val_bikers, val_tours)
                val_meta_path = val_path / f"{table}_val.csv"
                val_meta.to_csv(val_meta_path, index=False)
                additional_files[f"{table}_val"] = val_meta_path

        return train_path, val_path, additional_files

    def _filter_meta_table(self, table_df: pd.DataFrame, table_name: str,
                          valid_bikers: set, valid_tours: set) -> pd.DataFrame:
        """Filter meta tables based on valid bikers and tours."""
        if table_name == 'bikers':
            return table_df[table_df['biker_id'].isin(valid_bikers)].copy()
        elif table_name == 'tours':
            return table_df[table_df['tour_id'].isin(valid_tours)].copy()
        elif table_name == 'tour_convoy':
            filtered = table_df[table_df['tour_id'].isin(valid_tours)].copy()
            # Also filter the participant lists to only include valid bikers
            for col in ['going', 'maybe', 'invited', 'not_going']:
                if col in filtered.columns:
                    filtered.loc[:, col] = filtered[col].apply(
                        lambda x: [bid for bid in x if bid in valid_bikers]
                        if isinstance(x, list) else x
                    )
            return filtered
        elif table_name == 'bikers_network':
            filtered = table_df[table_df['biker_id'].isin(valid_bikers)].copy()
            if 'friends' in filtered.columns:
                filtered.loc[:, 'friends'] = filtered['friends'].apply(
                    lambda x: [friend for friend in x if friend in valid_bikers]
                    if isinstance(x, list) else x
                )
            return filtered
        return table_df.copy()


# Registry of data splitters
DATA_SPLITTERS = {
    "csv": CSVDataSplitter,
    "image_folder": ImageFolderDataSplitter,
    "image_classification": ImageFolderDataSplitter,
    "recommendation": RecommendationDataSplitter,
    "time_series": TimeSeriesDataSplitter,
    "custom": CustomDataSplitter,
    "emnist": EMNISTDataSplitter,
    "biker_recommender": BikerRecommenderDataSplitter,
    "multilabel": MultilabelDataSplitter,
    "classify_leaves": ClassifyLeavesDataSplitter
}
