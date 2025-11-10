from .extractor import FeatureExtractor
from .features import DINOFeatureExtractor
from .gem import vit_dino_embedder, GeMFeatureExtractor
from .mobile import RMACFeatureExtractor
from .reid import ReIDLikeExtractor
from .fewshot import DinoV2Extractor
from .matching import SimilarityResult, rank_similarities, select_top_detection_indices
from .reference_loader import (
    ReferenceFeature,
    list_reference_image_paths,
    load_reference_features,
)
from .utils import crop_bbox

__all__ = [
    "FeatureExtractor",
    "DINOFeatureExtractor",
    "GeMFeatureExtractor",
    "RMACFeatureExtractor",
    "ReIDLikeExtractor",
    "DinoV2Extractor",
    "vit_dino_embedder",
    "SimilarityResult",
    "rank_similarities",
    "select_top_detection_indices",
    "ReferenceFeature",
    "list_reference_image_paths",
    "load_reference_features",
    "crop_bbox",
]
