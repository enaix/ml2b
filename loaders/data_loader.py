from abc import ABC, abstractmethod
from typing import Any, Dict, get_type_hints, get_args
#from python.competition import Competition
from python.competition import *
import pandas as pd



class DataLoader(ABC):
    """Abstract base class for competition data loading strategies"""
    DEFAULT_SCHEMA = {}

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


    @classmethod
    def schema_dict(cls, schema_type: type | None = None, expose: bool = False, parent_key: str | None = None) -> dict[str, Any]:
        """
        Returns a dictionary with types. Recursively unpacks the TypedDict if expose=True.
        If parent_key contains 'val', adds '_val' suffix to the nested fields.
        """
        if schema_type is None:
            schema_type = cls.DEFAULT_SCHEMA

        result = {}
        annotations = get_type_hints(schema_type, include_extras=True)

        for k, v in annotations.items():
            # if Annotated
            if get_args(v):
                base_type, *_ = get_args(v)
            else:
                base_type = v

            key_name = k
            if expose and parent_key and "val" in parent_key.lower():
                key_name = f"{k}_val"

            if isinstance(base_type, type) and issubclass(base_type, dict) and hasattr(base_type, "__annotations__"):
                nested = cls.schema_dict(base_type, expose=expose, parent_key=k)
                if expose:
                    result.update(nested)
                else:
                    result[key_name] = nested
            else:
                result[key_name] = cls._normalize_type(base_type)

        return result

    @classmethod
    def schema(cls, schema_type: type | None = None, expose: bool = False, parent_key: str | None = None) -> dict[str, Any]:
        """Returns a dictionary of types without comments"""
        if schema_type is None:
            schema_type = cls.DEFAULT_SCHEMA

        result = {}
        annotations = get_type_hints(schema_type, include_extras=True)

        for k, v in annotations.items():
            if get_args(v):
                base_type, *meta = get_args(v)
                comment = meta[0] if meta else ""
            else:
                base_type, comment = v, ""

            key_name = k
            if expose and parent_key and "val" in parent_key.lower():
                key_name = f"{k}_val"

            if isinstance(base_type, type) and issubclass(base_type, dict) and hasattr(base_type, "__annotations__"):
                nested = cls.schema(base_type, expose=expose, parent_key=k)
                if expose:
                    result.update(nested)
                else:
                    result[key_name] = nested
            else:
                result[key_name] = {
                    "type": cls._normalize_type(base_type),
                    "comment": comment
                }

        return result

    @classmethod
    def get_ordered_result(cls, result: dict[str, Any], flat_schema: dict[str, Any]) -> dict[str, Any]:
        """Ensure that the loader result follows the schema dictionary order"""
        result_sorted = {}

        for key in flat_schema.keys():
            # Python 3.7+ ensures dictionary order being the same as the initialization order
            if key in result:
                result_sorted[key] = result[key]
        return result_sorted

    @staticmethod
    def _normalize_type(tp: type) -> str:
        """Simplify the string representation of a type"""
        module = getattr(tp, "__module__", "")
        name = getattr(tp, "__name__", str(tp))
        if module == "pandas.core.frame":
            return "pd.DataFrame"
        elif module == "numpy":
            return "np.ndarray"
        elif module in ("builtins", ""):
            return name
        return f"{module}.{name}"
