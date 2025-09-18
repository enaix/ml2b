from .default import DefaultDataLoader
from .emnist import EMNISTDataLoader
from .multilabel import MultiLabelDataLoader
from .biker import BikerRecommenderDataLoader
from .data_loader import DataLoader


# Registry of data loaders
DATA_LOADERS: dict[str, DataLoader] = {
    "default": DefaultDataLoader,
    "emnist": EMNISTDataLoader,
    "multilabel": MultiLabelDataLoader,
    "biker_recommender": BikerRecommenderDataLoader
}
