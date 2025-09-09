import os
import sys
from enum import Enum, StrEnum
import weakref
import traceback
import json
import subprocess
from pathlib import Path
import shutil
import docker
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, train_test_split
import importlib
from docker import DockerClient
from loguru import logger

class Language(StrEnum):
    English = "English"
    Arab = "Arab"
    Chinese = "Chinese"
    Italian = "Italian"
    Kazakh = "Kazakh"
    Polish = "Polish"
    Romanian = "Romanian"
    Spanish = "Spanish"
    Turkish = "Turkish"
    Belarus = "Belarus"
    Japanese = "Japanese"


class CodeLanguage(StrEnum):
    """Supported programming languages for code generation"""
    Python = "python"
    #R = "rlang"
    #Julia = "julia"


CODEPATHS = {CodeLanguage.Python: "code.py",} #CodeLanguage.R: None, CodeLanguage.Julia: None}
CODE_EXT = {CodeLanguage.Python: ".py"}


class RunnerInput(StrEnum):
    """Types of input that runners can accept"""
    DescOnly = "DescOnly"
    DescAndData = "DescAndData"


class RunnerOutput(StrEnum):
    """Types of output that runners can produce"""
    CodeOnly = "CodeOnly"
    CodeAndData = "CodeAndData"
    DataOnly = "DataOnly"


class BenchMode(StrEnum):
    """Benchmark operation modes"""
    MonolithicPredict = "MONO_PREDICT"
    ModularPredict = "MODULAR_PREDICT"


class DataSplitter(ABC):
    """Abstract base class for data splitting strategies"""
    
    @abstractmethod
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[Any, Any]]:
        """Return list of (train_indices, val_indices) tuples"""
        pass
    
    @abstractmethod
    def prepare_fold_data(self, comp: 'Competition', train_indices: Any, val_indices: Any, 
                         fold_idx: int, fold_dir: str, private_dir: str) -> Tuple[str, str, Dict[str, str]]:
        """Prepare data for a specific fold. Returns (train_path, val_path, additional_files)"""
        pass


class CSVDataSplitter(DataSplitter):
    """Standard CSV data splitter using train_test_split with 80:20 ratio"""
    
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data using train_test_split with 80:20 ratio"""
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
            print(f"Warning: train_test_split only supports 1 split (80:20). Using n_splits=1 instead of {n_splits}")
        
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
    
    def prepare_fold_data(self, comp: 'Competition', train_indices: np.ndarray, val_indices: np.ndarray,
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
    
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[List[str], List[str]]]:
        """Split image data using folder structure"""
        train_images_dir = None
        train_csv_path = None
        
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
    
    def prepare_fold_data(self, comp: 'Competition', train_indices: List[str], val_indices: List[str],
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
    
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data by users to avoid data leakage"""
        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")
        
        train_df = pd.read_csv(train_file.path)
        
        # Check if this is a recommendation system
        required_cols = ['biker_id', 'tour_id']
        if not all(col in train_df.columns for col in required_cols):
            print(f"Warning: Expected columns {required_cols} for recommendation splitting")
            print("Falling back to regular KFold splitting")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return list(kf.split(train_df))
        
        # Use GroupKFold to ensure each user is in only one fold
        group_kf = GroupKFold(n_splits=n_splits)
        dummy_X = np.arange(len(train_df))
        groups = train_df['biker_id'].values
        
        return list(group_kf.split(dummy_X, groups=groups))
    
    def prepare_fold_data(self, comp: 'Competition', train_indices: np.ndarray, val_indices: np.ndarray,
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
    
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split time series data maintaining temporal order"""
        train_file = comp.get_file("train")
        if not train_file or not train_file.exists():
            raise FileNotFoundError(f"Train file not found for competition {comp.comp_id}")
        
        train_df = pd.read_csv(train_file.path)
        
        # Get time column from metadata
        time_col = comp.metadata.get("time_col", "timestamp")
        if time_col not in train_df.columns:
            print(f"Warning: Time column '{time_col}' not found, falling back to regular split")
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
    
    def prepare_fold_data(self, comp: 'Competition', train_indices: np.ndarray, val_indices: np.ndarray,
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
    
    def __init__(self, split_function: Callable):
        self.split_function = split_function
    
    def split_data(self, comp: 'Competition', n_splits: int) -> List[Tuple[Any, Any]]:
        return self.split_function(comp, n_splits)
    
    def prepare_fold_data(self, comp: 'Competition', train_indices: Any, val_indices: Any,
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


class DataLoader(ABC):
    """Abstract base class for competition data loading strategies"""
    
    @abstractmethod
    def load_train_data(self, comp: 'Competition', fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded fold directory"""
        pass
    
    @abstractmethod
    def load_validation_features(self, comp: 'Competition', fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded fold directory"""
        pass
    
    @abstractmethod
    def load_validation_labels(self, comp: 'Competition', fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded private directory"""
        pass
    
    @abstractmethod
    def get_data_structure(self) -> Dict[str, str]:
        """Return the expected data structure for this loader"""
        pass


class DefaultDataLoader(DataLoader):
    """Default data loader with hardcoded paths"""
    
    def load_train_data(self, comp: 'Competition', fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded path"""
        dataset = {}
        train_path = os.path.join(base_path, "data", "folds", comp.comp_id, f"train_{fold_idx}.csv")
        
        if os.path.exists(train_path):
            dataset['data'] = pd.read_csv(train_path)
        else:
            raise ValueError(f"Train file not found: {train_path}")
        
        return dataset
    
    def load_validation_features(self, comp: 'Competition', fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded path"""
        dataset = {}
        val_path = os.path.join(base_path, "data", "folds", comp.comp_id, f"X_val_{fold_idx}.csv")
        
        if os.path.exists(val_path):
            dataset['X_val'] = pd.read_csv(val_path)
        else:
            raise ValueError(f"Validation features file not found: {val_path}")
        
        return dataset
    
    def load_validation_labels(self, comp: 'Competition', fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded path"""
        y_val_path = os.path.join(base_path, "data", "validation", comp.comp_id, f"y_val_{fold_idx}.csv")
        
        if os.path.exists(y_val_path):
            return pd.read_csv(y_val_path)
        else:
            raise ValueError(f"Validation labels file not found: {y_val_path}")
    
    def get_data_structure(self) -> Dict[str, str]:
        return {
            'data': 'Training data with features and target',
            'X_val': 'Validation features without target',
        }


# Registry of data loaders
DATA_LOADERS = {
    "default": DefaultDataLoader,
}


class CompetitionFile:
    """Represents a single file in a competition"""
    def __init__(self, name: str, path: str, file_type: str = "data", required: bool = True):
        self.name = name
        self.path = path
        self.file_type = file_type
        self.required = required
    
    def exists(self) -> bool:
        return os.path.exists(self.path)


class Competition:
    def __init__(self, comp_id: str, bench: weakref.ReferenceType, metadata: dict, tasks: dict, 
                 grading_stage: bool = False):
        """
        Represents a machine learning competition.
        
        Args:
            comp_id: Unique competition identifier
            bench: Weak reference to the benchmark pipeline
            metadata: Competition configuration and metadata
            tasks: Task descriptions for different languages
            grading_stage: If True, skip file validation (for Docker grading stage)
        """
        self.comp_id = comp_id
        self.metadata = metadata
        self.bench = bench
        self.tasks = tasks
        self.grading_stage = grading_stage
        
        # Set competition path based on stage
        if grading_stage:
            self.comp_path = os.path.join('/bench/data/', "data", comp_id)
        else:
            self.comp_path = os.path.join(bench().base_path(), "competitions", "data", comp_id)
        
        # Initialize files based on metadata or discovery
        self.files = self._initialize_files()
        
        # Validate files unless we're in grading stage
        if not grading_stage:
            self._validate_files()

    def _initialize_files(self) -> Dict[str, CompetitionFile]:
        """Initialize competition files from metadata or by discovery"""
        files = {}
        
        # Get file configuration from metadata
        file_config = self.metadata.get("files", {
            "train": {"type": "data", "required": True, "extensions": [".csv"]},
        })
        
        # First, try to use explicit file mapping from metadata
        explicit_files = self.metadata.get("file_mapping", {})
        for file_key, file_info in explicit_files.items():
            file_path = os.path.join(self.comp_path, file_info["filename"])
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
                    potential_path = os.path.join(self.comp_path, f"{file_key}{ext}")
                    
                    # Only check existence if we're not in grading stage
                    if not self.grading_stage:
                        file_exists = os.path.exists(potential_path)
                    else:
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
                if not file_found and config.get("required", True) and not self.grading_stage:
                    potential_path = os.path.join(self.comp_path, f"{file_key}.csv")
                    files[file_key] = CompetitionFile(
                        name=file_key,
                        path=potential_path,
                        file_type=config.get("type", "data"),
                        required=True
                    )
        
        # Discover additional files
        if os.path.exists(self.comp_path) or self.grading_stage:
            for item in os.listdir(self.comp_path):
                item_path = os.path.join(self.comp_path, item)
                
                # Skip if we're not in grading stage and path doesn't exist
                if not self.grading_stage and not os.path.exists(item_path):
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
    
    def _validate_files(self) -> None:
        """Validate that all required files exist"""
        missing_files = []
        for file_key, comp_file in self.files.items():
            if comp_file.required and not comp_file.exists():
                missing_files.append(f"{file_key} ({comp_file.path})")
        
        if missing_files:
            error_msg = f"Competition {self.comp_id}: missing required files: {', '.join(missing_files)}"
            if self.bench() is not None:
                self.bench().shutdown(1)
            else:
                raise FileNotFoundError(error_msg)
    
    def get_file(self, file_key: str) -> Optional[CompetitionFile]:
        """Get a specific file by key"""
        return self.files.get(file_key)
    
    def get_files_by_type(self, file_type: str) -> List[CompetitionFile]:
        """Get all files of a specific type"""
        return [f for f in self.files.values() if f.file_type == file_type]
    
    def get_all_files(self) -> Dict[str, CompetitionFile]:
        """Get all files in the competition"""
        return self.files.copy()
    
    def get_data_files(self) -> Dict[str, str]:
        """Get all data files as a dict of {name: path}"""
        data_files = {}
        for file_key, comp_file in self.files.items():
            if comp_file.file_type in ["data", "metadata"] and comp_file.exists():
                data_files[file_key] = comp_file.path
        return data_files

    def get_data_loader(self, loader_name: Optional[str] = None) -> DataLoader:
        """Get the appropriate data loader for this competition"""
        if loader_name is None:
            loader_name = self.metadata.get('data_loader', 'default')
        
        loader_class = DATA_LOADERS.get(loader_name, DefaultDataLoader)
        return loader_class()

    def get_available_languages(self) -> list[Language]:
        """Get available languages for this competition"""
        return list(self.tasks.keys())

    def _get_meta_for_lang(self, lang: Language) -> dict:
        """Get metadata for a specific language"""
        values = self.tasks.get(lang)
        if values is None:
            print(f"Competition: could not find metadata for language {lang}")
            self.bench().shutdown(1)
        return values

    def get_description(self, lang: Language) -> dict:
        """Get description for a specific language"""
        return self._get_meta_for_lang(lang).get("description")

    def get_data_card(self, lang: Language) -> dict:
        """Get data card for a specific language"""
        return self._get_meta_for_lang(lang).get("data_card")

    def get_domain(self, lang: Language) -> dict:
        """Get domain information for a specific language"""
        return self._get_meta_for_lang(lang).get("domain")

    def get_metric(self, lang: Language) -> dict:
        return self._get_meta_for_lang(lang).get("metric")

    def get_code_ext(self, code_lang: CodeLanguage) -> str:
        return CODE_EXT[code_lang]

    # Legacy properties for backward compatibility
    @property
    def train_data(self) -> str:
        """Get train data path (backward compatibility)"""
        train_file = self.get_file("train")
        return train_file.path if train_file else os.path.join(self.comp_path, "train.csv")
    
    @property
    def test_data(self) -> str:
        """Get test data path (backward compatibility)"""
        test_file = self.get_file("test")
        return test_file.path if test_file else os.path.join(self.comp_path, "test.csv")


class CompetitionData:
    """Represents data for a specific competition fold"""
    def __init__(self, train_path: os.PathLike, val_path: os.PathLike, fold_idx: int = 0, 
                 additional_files: Dict[str, str] = None):
        self.train_path = train_path
        self.val_path = val_path
        self.fold_idx = fold_idx
        self.additional_files = additional_files or {}

    def get_train(self) -> os.PathLike:
        return self.train_path

    def get_val(self) -> os.PathLike:
        return self.val_path
    
    def get_additional_file(self, file_key: str) -> Optional[str]:
        """Get path to additional file"""
        return self.additional_files.get(file_key)
    
    def get_all_files(self) -> Dict[str, str]:
        """Get all files including train/val and additional files"""
        all_files = {
            "train": str(self.train_path),
            "val": str(self.val_path)
        }
        all_files.update(self.additional_files)
        return all_files


class BenchPipeline:
    """Main benchmark pipeline for managing competitions and data"""
    def __init__(self, basepath: os.PathLike, max_folds: int = 5, prepare_data: bool = False):
        self.basepath = basepath
        self.max_folds = max_folds
        self.current_comp = 0
        self.current_fold = 0
        self.competitions: list[Competition] = []
        self.folds: dict[str, list[CompetitionData]] = {}
        self._languages: list[Language] = []
        self.grader_module = None
        self.prepare_data = prepare_data

        self._initialize_folders()
        self._load_competitions()
        self._load_graders()

    def _initialize_folders(self) -> None:
        """Initialize folder structure for folds and validation"""
        folds_dir = os.path.join("competitions", "folds")
        private_dir = os.path.join("competitions", "validation")

        if os.path.exists(folds_dir):
            shutil.rmtree(folds_dir)
        if os.path.exists(private_dir):
            shutil.rmtree(private_dir)

    def _load_competitions(self) -> None:
        """Load competitions from competitions.json"""
        comp_json = os.path.join(self.base_path(), "competitions", "competitions.json")
        if not os.path.exists(comp_json):
            print(f"Missing {comp_json} file")
            self.shutdown(1)

        with open(comp_json, "r") as f:
            comp_data = json.load(f)

        tasks_dir = os.path.join(self.base_path(), "competitions", "tasks")
        
        # Handle tasks directory
        self._languages = [Language.English]
        tasks = {Language.English: []}
        
        if os.path.exists(tasks_dir):
            language_files = os.listdir(tasks_dir)
            self._languages = []
            tasks = {}
            
            for file in language_files:
                try:
                    lang = Language(file.split('.')[0])
                    self._languages.append(lang)
                except ValueError:
                    print(f"Bad task file {file}: no such language")
                    self.shutdown(1)

                file_path = os.path.join(tasks_dir, file)
                df = pd.read_csv(file_path)
                tasks[lang] = df.to_dict('records')
        else:
            print(f"Note: Tasks directory not found at {tasks_dir}. Using default English setup.")
        
        # Process each competition
        for key, value in comp_data.items():
            if key.startswith("_"):
                continue
            
            comp_tasks = {}
            for lang in self._languages:
                lang_tasks = tasks.get(lang, [])
                comp_task = None
                
                for task in lang_tasks:
                    if task.get("comp-id") == key:
                        comp_task = task
                        break
                
                if comp_task is None:
                    print(f"! warning: competition id {key} : missing in {str(lang)} task descriptions")
                    continue
                
                comp_tasks[lang] = comp_task

            if not comp_tasks:
                print(f"!  warning: competition id {key} : missing in all task descriptions")
                continue

            self.competitions.append(Competition(key, weakref.ref(self), value, comp_tasks))
            self.folds[key] = []

    def _load_graders(self) -> None:
        """Load grading functions module"""
        grader_path = os.path.join(self.base_path(), "python", "grade_functions.py")
        spec = importlib.util.spec_from_file_location("grade_functions", grader_path)
        self.grader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.grader_module)

    def base_path(self) -> os.PathLike:
        return self.basepath

    def shutdown(self, exit_code: int):
        """Shutdown the benchmark with exit code"""
        if exit_code != 0:
            print(f"Benchmark stopped with abnormal exit code {exit_code}")
            traceback.print_exc()
        sys.exit(exit_code)

    def languages(self) -> list[Language]:
        return self._languages

    def total(self) -> int:
        return len(self.competitions)

    def total_folds(self, comp: Competition) -> int:
        return min(self.max_folds, comp.metadata.get("cv_folds", 1))

    def next_competition(self) -> Optional[Competition]:
        """Get next competition"""
        if self.current_comp >= len(self.competitions):
            self.current_comp = 0
            return None
        comp = self.competitions[self.current_comp]
        self.current_comp += 1
        return comp

    def next_fold(self, comp: Competition) -> Optional[CompetitionData]:
        """Get next fold for a competition"""
        if not self.prepare_data or self.current_fold >= self.total_folds(comp):
            self.current_fold = 0
            return None

        fold = self.folds[comp.comp_id][self.current_fold]
        self.current_fold += 1
        return fold

    def test_submission_code(self, comp: Competition, lang: Language, codelang: CodeLanguage, code: str) -> dict:
        """
        Submit the code and return metric
        """

        # Prepare submission dir
        # ======================
        submission_dir = (Path(self.base_path()) / str(codelang) / f"submission_{uniq_suf}").resolve()
        if os.path.exists(submission_dir):
            shutil.rmtree(submission_dir)
        os.mkdir(submission_dir)

        if codelang == CodeLanguage.Python:
            with open(os.path.join(submission_dir, "__init__.py"), 'w') as f:
                f.write("")

        with open(os.path.join(submission_dir, CODEPATHS[codelang]), 'w') as f:
            f.write(code)

        env_vars = {
            "COMPETITION_ID": comp.comp_id,
            "BENCH_LANG": str(lang),
            "BENCH_MODE": str(BenchMode.ModularPredict),
            "BENCH_FOLDS_OVERRIDE": "1",
            "PYTHONDONTWRITEBYTECODE": "1"
        }
        network_name = "python_no_inet"

        # Проверяем, есть ли сеть
        networks = [n.name for n in client.networks.list()]
        if network_name not in networks:
            client.networks.create(network_name, driver="bridge", internal=True)
        container = client.containers.run(
            image=image_name,
            detach=True,
            environment=env_vars,
            volumes={
                submission_dir.as_posix(): {'bind': '/home/bench/submission', 'mode': 'rw'},
                (Path(self.base_path()) / "competitions").resolve().as_posix(): {'bind': '/home/bench/competitions', 'mode': 'ro'}
            },
            network="python_no_inet",
            entrypoint=["mamba", "run", "-n", "agent", "python", "./bench.py"],
            **runtime_config,
            working_dir="/home/bench"
        )

        exit_code = container.wait(timeout=60*60)
        logs = container.logs().decode('utf-8')

        logger.info("Evaluation container results:\n exit_code: {}\n logs: {}", exit_code, logs)
        container.remove() 
        results_path = submission_dir / "results.json"
        if not os.path.exists(results_path):
            logger.info("{} container failed to generate output", str(codelang))
            return {"errors": [f"Failed to obtain output for {str(codelang)}"], "success": False}
        with open(results_path, 'r') as f:
            results = json.load(f)
        shutil.rmtree(submission_dir)
        return results

    def test_submission_data(self, comp: Competition, fold: CompetitionData, lang: Language, 
                            codelang: CodeLanguage, data: Any) -> dict:
        """Test submission data with grader"""
        val_dir = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        grader = comp.metadata.get("grader", "default")

        try:
            score = self.grader_module.GRADERS[grader](data, val_dir, comp.metadata)
        except Exception as e:
            err_msg = f"Grader {grader} failed for competition {comp.comp_id} : {e=}"
            print(err_msg)
            return {"manual_submission_score": None, "manual_submission_error": err_msg}

        return {"manual_submission_score": score, "manual_submission_error": None}

    def test_submission_data_path(self, comp: Competition, fold: CompetitionData, lang: Language, 
                                 codelang: CodeLanguage, path_to_data: os.PathLike) -> dict:
        """Test submission data from file path"""
        df = pd.read_csv(path_to_data)
        return self.test_submission_data(comp, fold, lang, codelang, df)

    def prepare_train_data(self, comp: Competition) -> None:
        """Prepare training data for all folds"""
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds")
        comp_fold_dir = os.path.join(fold_dir, comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation")
        comp_private_dir = os.path.join(private_dir, comp.comp_id)

        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(private_dir, exist_ok=True)
        os.makedirs(comp_fold_dir, exist_ok=True)
        os.makedirs(comp_private_dir, exist_ok=True)

        # Get splitting strategy
        split_strategy = comp.metadata.get("split_strategy", "csv")
        splitter_class = DATA_SPLITTERS.get(split_strategy)
        
        if not splitter_class:
            print(f"Unknown split strategy '{split_strategy}' for competition {comp.comp_id}")
            print(f"Available strategies: {list(DATA_SPLITTERS.keys())}")
            self.shutdown(1)
        
        try:
            splitter = splitter_class()
            splits = splitter.split_data(comp, self.total_folds(comp))
        except Exception as e:
            print(f"prepare_train_data(): splitting failed: {e=} for competition {comp.comp_id}")
            self.shutdown(1)

        # Prepare additional files
        additional_files = {}
        for file_key, comp_file in comp.get_all_files().items():
            if (file_key not in ["train"] and comp_file.exists() and 
                comp_file.file_type in ["data", "metadata"]):
                additional_files[file_key] = comp_file.path

        # Create folds
        for i, (train_indices, val_indices) in enumerate(splits):
            try:
                train_path, val_path, fold_additional_files = splitter.prepare_fold_data(
                    comp, train_indices, val_indices, i, comp_fold_dir, comp_private_dir
                )
            except Exception as e:
                print(f"prepare_train_data(): fold {i} preparation failed: {e=} for competition {comp.comp_id}")
                self.shutdown(1)

            # Copy additional files
            for file_key, file_path in additional_files.items():
                if os.path.isfile(file_path):
                    fold_file_path = os.path.join(comp_fold_dir, f"{file_key}_{i}{os.path.splitext(file_path)[1]}")
                    shutil.copy2(file_path, fold_file_path)
                    fold_additional_files[file_key] = fold_file_path
                elif os.path.isdir(file_path):
                    fold_dir_path = os.path.join(comp_fold_dir, f"{file_key}_{i}")
                    if not os.path.exists(fold_dir_path):
                        shutil.copytree(file_path, fold_dir_path)
                    fold_additional_files[file_key] = fold_dir_path

            self.folds[comp.comp_id].append(CompetitionData(
                train_path, val_path, i, fold_additional_files
            ))

    def erase_train_data(self, comp: Competition) -> None:
        """Erase training data"""
        if not self.prepare_data:
            return None

        fold_dir = os.path.join(self.base_path(), "competitions", "folds", comp.comp_id)
        private_dir = os.path.join(self.base_path(), "competitions", "validation", comp.comp_id)
        
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        if os.path.exists(private_dir):
            shutil.rmtree(private_dir)

    def register_custom_splitter(self, name: str, splitter_class: type):
        """Register a custom data splitter"""
        if not issubclass(splitter_class, DataSplitter):
            raise ValueError("Custom splitter must inherit from DataSplitter")
        DATA_SPLITTERS[name] = splitter_class

    def get_available_splitters(self) -> List[str]:
        """Get available splitting strategies"""
        return list(DATA_SPLITTERS.keys())

    def register_custom_loader(self, name: str, loader_class: type):
        """Register a custom data loader"""
        if not issubclass(loader_class, DataLoader):
            raise ValueError("Custom loader must inherit from DataLoader")
        DATA_LOADERS[name] = loader_class

    def get_available_loaders(self) -> List[str]:
        """Get available data loaders"""
        return list(DATA_LOADERS.keys())
