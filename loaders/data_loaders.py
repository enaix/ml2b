from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from python.competition import *



class DataLoader(ABC):
    """Abstract base class for competition data loading strategies"""

    @abstractmethod
    def load_train_data(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load training data from hardcoded fold directory"""
        pass

    @abstractmethod
    def load_validation_features(self, comp: Competition, fold_idx: int, base_path: str) -> Dict[str, Any]:
        """Load validation features from hardcoded fold directory"""
        pass

    @abstractmethod
    def load_validation_labels(self, comp: Competition, fold_idx: int, base_path: str) -> pd.DataFrame:
        """Load validation labels from hardcoded private directory"""
        pass

    @abstractmethod
    def get_data_structure(self) -> Dict[str, str]:
        """Return the expected data structure for this loader"""
        pass
