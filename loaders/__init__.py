from .default import DefaultDataLoader
from .emnist import EMNISTDataLoader
from .multilabel import MultiLabelDataLoader
from .biker import BikerRecommenderDataLoader


# Registry of data loaders
DATA_LOADERS = {
    "default": DefaultDataLoader,
    "emnist": EMNISTDataLoader,
    "multilabel": MultiLabelDataLoader,
    "biker_recommender": BikerRecommenderDataLoader,
    "segmentation": SegmentationDataLoader,
}
