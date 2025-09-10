from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

from competition import *


class DataSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    def __init__(self, log_error: any, do_shutdown: any, grading_stage: bool = False):
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

    def __init__(self, log_error: any, do_shutdown: any, grading_stage: bool = False):
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data using train_test_split with 80:20 ratio"""
        self.prepare_competition_files(comp)

        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")

        train_df = pd.read_csv(train_file.path)
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
        train_df = pd.read_csv(train_file.path)
        target_col = comp.metadata["target_col"]

        # Split data using the provided indices
        X, y = train_df.drop(columns=[target_col]), train_df[target_col]
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]

        # Save fold data
        train_fold = pd.concat([X_train, y_train], axis=1)
        train_path = os.path.join(fold_dir, f"train_{fold_idx}.csv")
        x_val_path = os.path.join(fold_dir, f"X_val_{fold_idx}.csv")
        y_val_path = os.path.join(private_dir, f"y_val_{fold_idx}.csv")

        train_fold.to_csv(train_path, index=False)
        X_val.to_csv(x_val_path, index=False)
        y_val.to_csv(y_val_path, index=False)

        return train_path, x_val_path, {}


class ImageFolderDataSplitter(DataSplitter):
    """Data splitter for image classification with folder structure"""

    def __init__(self, log_error: any, do_shutdown: any, grading_stage: bool = False):
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
        fold_train_dir = os.path.join(fold_dir, f"train_images_{fold_idx}")
        fold_val_dir = os.path.join(fold_dir, f"val_images_{fold_idx}")
        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_val_dir, exist_ok=True)

        # Create CSV files with image paths and labels
        train_csv_path = os.path.join(fold_dir, f"train_{fold_idx}.csv")
        val_csv_path = os.path.join(fold_dir, f"X_val_{fold_idx}.csv")
        val_labels_path = os.path.join(private_dir, f"y_val_{fold_idx}.csv")

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
        pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
        pd.DataFrame(val_data).to_csv(val_csv_path, index=False)
        pd.DataFrame(val_labels).to_csv(val_labels_path, index=False)

        additional_files = {
            'train_images': fold_train_dir,
            'val_images': fold_val_dir
        }

        return train_csv_path, val_csv_path, additional_files

    def _extract_label_from_path(self, img_path: str) -> str:
        """Extract label from image path (assumes parent directory is class name)"""
        return os.path.basename(os.path.dirname(img_path))


class RecommendationDataSplitter(DataSplitter):
    """Custom data splitter for recommendation systems"""

    def __init__(self, log_error: any, do_shutdown: any, grading_stage: bool = False):
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
        train_path = os.path.join(fold_dir, f"train_{fold_idx}.csv")
        train_fold.to_csv(train_path, index=False)

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

        x_val_path = os.path.join(fold_dir, f"X_val_{fold_idx}.csv")
        y_val_path = os.path.join(private_dir, f"y_val_{fold_idx}.csv")

        val_features.to_csv(x_val_path, index=False)
        val_labels.to_csv(y_val_path, index=False)

        return train_path, x_val_path, {}


class TimeSeriesDataSplitter(DataSplitter):
    """Time series data splitter that respects temporal order"""

    def __init__(self, log_error: any, do_shutdown: any, grading_stage: bool = False):
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
        train_path = os.path.join(fold_dir, f"train_{fold_idx}.csv")
        val_path = os.path.join(fold_dir, f"X_val_{fold_idx}.csv")
        val_labels_path = os.path.join(private_dir, f"y_val_{fold_idx}.csv")

        train_data.to_csv(train_path, index=False)
        X_val.to_csv(val_path, index=False)
        y_val.to_csv(val_labels_path, index=False)

        return train_path, val_path, {}


class CustomDataSplitter(DataSplitter):
    """Custom data splitter for specific competitions"""

    def __init__(self, split_function: Callable, log_error: any, do_shutdown: any, grading_stage: bool = False):
        self.split_function = split_function
        super().__init__(log_error, do_shutdown, grading_stage)

    def split_data(self, comp: Competition, n_splits: int) -> List[Tuple[Any, Any]]:

        self.prepare_competition_files(comp)
        return self.split_function(comp, n_splits)

    def prepare_fold_data(self, comp: Competition, train_indices: Any, val_indices: Any,
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        return self.split_function.prepare_fold_data(comp, train_indices, val_indices, fold_idx, fold_dir, private_dir)


# Registry of data splitters
DATA_SPLITTERS = {
    "csv": CSVDataSplitter,
    "image_folder": ImageFolderDataSplitter,
    "image_classification": ImageFolderDataSplitter,
    "recommendation": RecommendationDataSplitter,
    "time_series": TimeSeriesDataSplitter,
    "custom": CustomDataSplitter,
}
