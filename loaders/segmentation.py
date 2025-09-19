from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, TypedDict, Annotated
import numpy as np
import pandas as pd
import os
from python.competition import *


class SegmentationData(TypedDict):
    train: Annotated[pd.DataFrame, 'Training data with RLE annotations']
    train_images_path: Annotated[str, 'Path to training images directory']
    sample_submission: Annotated[pd.DataFrame, 'Sample submission format']
    train_semi_supervised_path: Annotated[Optional[str], 'Path to semi-supervised images']
    livecell_dataset_path: Annotated[Optional[str], 'Path to LIVECell dataset']

class SegmentationDataset(TypedDict):
    data: Annotated[SegmentationData, 'Training features and metadata']
    X_val: Annotated[pd.DataFrame, 'Validation features (image metadata)']


class SegmentationDataLoader(DataLoader):
    """Data loader for neuronal cell segmentation with RLE masks."""
    DEFAULT_SCHEMA = SegmentationDataset

    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> SegmentationData:
        """Load training data with image paths and metadata."""
        dataset = {}
        fold_dir = os.path.join(base_path, "data", "folds", comp.comp_id)

        train_path = os.path.join(fold_dir, f"train_{fold_idx}.csv")
        if os.path.exists(train_path):
            dataset['train'] = pd.read_csv(train_path)
            dataset['train'] = self._parse_segmentation_annotations(dataset['train'])
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")

        dataset['train_images_path'] = os.path.join(base_path, "data", comp.comp_id, "train")
        dataset['train_semi_supervised_path'] = os.path.join(base_path, "data", comp.comp_id, "train_semi_supervised")
        dataset['livecell_dataset_path'] = os.path.join(base_path, "data", comp.comp_id, "LIVECell_dataset_2021")

        sample_submission_path = os.path.join(base_path, "data", comp.comp_id, "sample_submission.csv")
        if os.path.exists(sample_submission_path):
            dataset['sample_submission'] = pd.read_csv(sample_submission_path)

        return dataset

    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        fold_dir = os.path.join(base_path, "data", "folds", comp.comp_id)
        x_val_path = os.path.join(fold_dir, f"X_val_{fold_idx}.csv")

        if os.path.exists(x_val_path):
            x_val = pd.read_csv(x_val_path)
            return self._parse_validation_features(x_val)
        else:
            raise FileNotFoundError(f"Validation features not found: {x_val_path}")

    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        y_val_path = os.path.join(base_path, "data", "validation", comp.comp_id, f"y_val_{fold_idx}.csv")

        if os.path.exists(y_val_path):
            y_val = pd.read_csv(y_val_path)
            return self._parse_segmentation_annotations(y_val)
        else:
            raise FileNotFoundError(f"Validation labels not found: {y_val_path}")

    def _parse_segmentation_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['id', 'annotation', 'width', 'height']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df['image_id'] = df['id'].apply(self._extract_image_id)
        
        df['annotation'] = df['annotation'].apply(self._validate_rle_annotation)

        optional_columns = ['cell_type', 'plate_time', 'sample_date', 'sample_id', 'elapsed_timedelta']
        for col in optional_columns:
            if col not in df.columns:
                df[col] = None

        return df

    def _parse_validation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'id' in df.columns:
            df['image_id'] = df['id'].apply(self._extract_image_id)
        
        return df

    def _extract_image_id(self, object_id: str) -> str:
        parts = str(object_id).split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:-1])
        return str(object_id)

    def _validate_rle_annotation(self, annotation: str) -> str:
        if pd.isna(annotation) or annotation == '':
            return ''
        
        try:
            parts = str(annotation).split()
            if len(parts) % 2 != 0:
                raise ValueError(f"RLE annotation must have even number of values: {annotation}")
            
            for part in parts:
                int(part)
            
            return str(annotation)
            
        except ValueError as e:
            print(f"Warning: Invalid RLE annotation format: {annotation}. Error: {e}")
            return ''

    def get_image_path(self, image_id: str, dataset: SegmentationData, split: str = 'train') -> str:
        if split == 'train':
            base_path = dataset['train_images_path']
        elif split == 'semi_supervised':
            base_path = dataset.get('train_semi_supervised_path', '')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        image_filename = f"{image_id}.png"
        return os.path.join(base_path, image_filename)

    def get_objects_for_image(self, image_id: str, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['image_id'] == image_id].copy()

    def create_submission_format(self, predictions_by_image: Dict[str, List[str]], 
                                image_dimensions: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
        submission_rows = []
        
        for image_id, masks in predictions_by_image.items():
            width, height = image_dimensions.get(image_id, (0, 0))
            masks_str = ' '.join(masks) if masks else ''
            submission_row = f"{image_id},{width} {height},{masks_str}"
            submission_rows.append(submission_row)
        
        return pd.DataFrame({'ImageId_ClassId': submission_rows})
