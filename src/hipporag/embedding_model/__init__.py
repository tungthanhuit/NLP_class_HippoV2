from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Transformers import TransformersEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "ContrieverModel",
    "NVEmbedV2EmbeddingModel",
    "OpenAIEmbeddingModel",
    "TransformersEmbeddingModel",
    "_get_embedding_model_class",
]


def _get_embedding_model_class(embedding_model_name: str = "text-embedding-3-small"):
    if "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"