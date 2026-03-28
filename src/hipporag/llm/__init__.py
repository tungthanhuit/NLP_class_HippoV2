import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if (
        config.llm_base_url is not None
        and "localhost" in config.llm_base_url
        and os.getenv("OPENAI_API_KEY") is None
    ):
        os.environ["OPENAI_API_KEY"] = "sk-"

    if config.llm_name.startswith("Transformers/"):
        raise ValueError(
            "Local HuggingFace Transformers LLMs (llm_name starting with 'Transformers/') are not supported in this build. "
            "Use an OpenAI-compatible API by setting llm_base_url and llm_name."
        )

    return CacheOpenAI.from_experiment_config(config)
