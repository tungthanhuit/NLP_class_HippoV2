from .base import EmbeddingConfig, BaseEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Transformers import TransformersEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "OpenAIEmbeddingModel",
    "TransformersEmbeddingModel",
    "_get_embedding_model_class",
]


def _get_embedding_model_class(embedding_model_name: str = "text-embedding-3-small"):
    if "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
