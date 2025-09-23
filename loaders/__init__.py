from .default import DefaultDataLoader
from .emnist import EMNISTDataLoader
from .multilabel import MultiLabelDataLoader
from .biker import BikerRecommenderDataLoader
from .classify_leaves import ClassifyLeavesDataLoader
from .data_loader import DataLoader
from .utils import read_csv_smart


# Registry of data loaders
DATA_LOADERS: dict[str, DataLoader] = {
    "default": DefaultDataLoader,
    "emnist": EMNISTDataLoader,
    "multilabel": MultiLabelDataLoader,
    "biker_recommender": BikerRecommenderDataLoader,
    "classify_leaves": ClassifyLeavesDataLoader
}
