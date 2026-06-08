import logging

from .EnhancedHippoRAG import EnhancedHippoRAG
from .HippoRAG import HippoRAG
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)


def _needs_enhanced(config: BaseConfig) -> bool:
    return config.use_enhancements or config.max_qa_steps > 1


class HippoRAGGate:
    """
    Single entry point that returns the correct HippoRAG variant.

    Routes to EnhancedHippoRAG when:
    - use_enhancements=True  (explicit opt-in)
    - max_qa_steps > 1       (IRCoT requires the enhanced pipeline)

    Routes to HippoRAG (baseline) otherwise.

    Because __new__ returns a concrete subclass instance, the result is a
    genuine HippoRAG or EnhancedHippoRAG — not a wrapper — so all existing
    method calls, isinstance checks, and type annotations work unchanged.

    Example
    -------
    config = BaseConfig(use_enhancements=True, max_qa_steps=3)
    rag = HippoRAGGate(global_config=config)
    # rag is an EnhancedHippoRAG instance
    """

    def __new__(
        cls,
        global_config: BaseConfig = None,
        save_dir: str = None,
        llm_model_name: str = None,
        llm_base_url: str = None,
        embedding_model_name: str = None,
        embedding_base_url: str = None,
        embedding_trust_remote_code: bool = None,
    ):
        if global_config is None:
            global_config = BaseConfig()

        kwargs = dict(
            global_config=global_config,
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            embedding_base_url=embedding_base_url,
            embedding_trust_remote_code=embedding_trust_remote_code,
        )

        if _needs_enhanced(global_config):
            logger.info(
                "[HippoRAGGate] → EnhancedHippoRAG "
                f"(use_enhancements={global_config.use_enhancements}, "
                f"max_qa_steps={global_config.max_qa_steps})"
            )
            return EnhancedHippoRAG(**kwargs)

        logger.info("[HippoRAGGate] → HippoRAG (baseline)")
        return HippoRAG(**kwargs)
