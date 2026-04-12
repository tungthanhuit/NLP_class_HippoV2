import os
from dataclasses import dataclass, field
from typing import Literal, Union, Optional, Any

from .logging_utils import get_logger


def _load_dotenv_if_present() -> None:
    """Load environment variables from a local .env file (if available).

    This is a convenience for local development/tests so users don't need to
    export a long list of variables.

    Opt-out by setting env `HIPPORAG_DISABLE_DOTENV=1`.
    Override the path via env `HIPPORAG_DOTENV_PATH=/path/to/.env`.
    """

    if (os.getenv("HIPPORAG_DISABLE_DOTENV") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    dotenv_path = (os.getenv("HIPPORAG_DOTENV_PATH") or "").strip() or None

    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    try:
        if dotenv_path is not None:
            load_dotenv(dotenv_path, override=False)
            # Normalize common misconfigurations (e.g., OPENAI_API_KEY="")
            # so downstream SDKs don't construct invalid headers.
            _key = os.getenv("OPENAI_API_KEY")
            if _key is not None and _key.strip() == "":
                os.environ.pop("OPENAI_API_KEY", None)
            return

        # First try from the current working directory (common when running scripts).
        # If that fails, fall back to searching relative to the calling file/module.
        discovered = find_dotenv(usecwd=True) or find_dotenv(usecwd=False)
        if discovered:
            load_dotenv(discovered, override=False)
            _key = os.getenv("OPENAI_API_KEY")
            if _key is not None and _key.strip() == "":
                os.environ.pop("OPENAI_API_KEY", None)
    except Exception:
        # Never fail import just because dotenv loading failed.
        return


_load_dotenv_if_present()

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""

    # LLM specific attributes
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."},
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the LLM model, if none, means using OPENAI service."
        },
    )
    embedding_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."
        },
    )
    max_new_tokens: Union[None, int] = field(
        default=2048, metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={
            "help": "How many chat completion choices to generate for each input message."
        },
    )
    seed: Union[None, int] = field(default=None, metadata={"help": "Random seed."})
    temperature: float = field(
        default=0, metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: {"type": "json_object"},
        metadata={"help": "Specifying the format that the model must output."},
    )

    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={
            "help": "Max number of retry attempts for an asynchronous API calling."
        },
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If set to True, will ignore all existing openie files and rebuild them from scratch."
        },
    )

    # KB backend (persistence)
    kb_backend: Literal["parquet", "neo4j"] = field(
        default="parquet",
        metadata={
            "help": "Knowledge base backend for persistence. 'parquet' uses local parquet+pickle+json; 'neo4j' stores KB in Neo4j."
        },
    )

    # Chunk vector backend (where passage vectors are stored/searched)
    # - default behavior: vectors are stored alongside the KB backend (parquet or Neo4j)
    # - optional: use Milvus for scalable similarity search of chunk/passage vectors
    chunk_vector_backend: Literal["default", "milvus"] = field(
        default="default",
        metadata={
            "help": "Backend for chunk/passage vectors. 'default' uses the KB backend (parquet/neo4j). 'milvus' stores/searches chunk vectors in Milvus (typically with kb_backend='neo4j')."
        },
    )

    # Neo4j KB backend specific attributes
    neo4j_uri: str = field(
        default="bolt://localhost:7687",
        metadata={"help": "Neo4j Bolt URI (e.g., bolt://localhost:7687)."},
    )
    neo4j_user: str = field(
        default="neo4j",
        metadata={"help": "Neo4j username."},
    )
    neo4j_password: Optional[str] = field(
        default=None,
        metadata={
            "help": "Neo4j password. If None, reads from env NEO4J_PASSWORD."
        },
    )
    neo4j_database: str = field(
        default="neo4j",
        metadata={"help": "Neo4j database name."},
    )
    neo4j_namespace: str = field(
        default="hipporag",
        metadata={
            "help": "Namespace prefix used to isolate KB data within a shared Neo4j instance."
        },
    )
    neo4j_namespace_include_llm: bool = field(
        default=False,
        metadata={
            "help": "If True, the effective Neo4j namespace is suffixed with llm_name and embedding_model_name. "
            "Set to False (default) to reuse the same KB across different llm_name values (embedding model still remains part of the namespace)."
        },
    )
    neo4j_batch_size: int = field(
        default=500,
        metadata={"help": "Batch size for Neo4j UNWIND writes/reads."},
    )

    # Milvus (vector DB) settings (used when chunk_vector_backend='milvus')
    milvus_uri: Optional[str] = field(
        default=None,
        metadata={
            "help": "Milvus URI for pymilvus connections, e.g. 'http://localhost:19530' or 'tcp://localhost:19530'."
        },
    )
    milvus_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Milvus/Zilliz token (if required). Leave None for local Milvus."
        },
    )
    milvus_collection_prefix: str = field(
        default="hipporag",
        metadata={
            "help": "Prefix for Milvus collection names. Actual collection names also include Neo4j namespace + llm + embedding labels."
        },
    )
    milvus_metric_type: Literal["IP", "COSINE", "L2"] = field(
        default="IP",
        metadata={
            "help": "Milvus metric type. If using IP, embeddings should be normalized to approximate cosine similarity."
        },
    )
    milvus_index_type: Literal["HNSW", "IVF_FLAT", "AUTOINDEX"] = field(
        default="HNSW",
        metadata={"help": "Milvus index type for chunk vectors."},
    )
    milvus_index_params: Optional[dict[str, Any]] = field(
        default_factory=lambda: {"M": 16, "efConstruction": 200},
        metadata={"help": "Index build params for Milvus (depends on index type)."},
    )
    milvus_search_params: Optional[dict[str, Any]] = field(
        default_factory=lambda: {"ef": 128},
        metadata={"help": "Search params for Milvus (depends on index type)."},
    )
    milvus_dense_top_k: int = field(
        default=200,
        metadata={
            "help": "Top-k passages to fetch from Milvus for dense retrieval and PPR passage weighting (should be >= retrieval_top_k)."
        },
    )

    # Storage specific attributes
    force_index_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."
        },
    )
    rerank_dspy_file_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={
            "help": "Multiplicative factor that modified the passage node weights in PPR."
        },
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."},
    )

    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={
            "help": "Name of the text-based preprocessor to use in preprocessing."
        },
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={
            "help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."
        },
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."},
    )
    preprocess_chunk_max_token_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."
        },
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default="by_token")

    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt",] = field(
        default="openie_openai_gpt",
        metadata={
            "help": "Class name indicating which information extraction model to use."
        },
    )
    openie_mode: Literal["online"] = field(
        default="online",
        metadata={
            "help": "Mode of the OpenIE model to use. Only 'online' is supported; it uses the configured LLM API."
        },
    )
    skip_graph: bool = field(
        default=False, metadata={"help": "Whether to skip graph construction or not."}
    )

    # Embedding specific attributes
    embedding_model_name: str = field(
        default="text-embedding-3-small",
        metadata={
            "help": "Embedding model to use. Supported patterns: OpenAI embeddings like 'text-embedding-3-small' (API, via embedding_base_url), or local Sentence-Transformers models via 'Transformers/<hf-id>' (e.g., 'Transformers/BAAI/bge-m3')."
        },
    )
    embedding_trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow Hugging Face remote code when loading local Sentence-Transformers models (Transformers/<hf-id>). Enable only if required by the model (e.g., Transformers/BAAI/bge-m3)."
        },
    )
    embedding_batch_size: int = field(
        default=16, metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True, metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048, metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto", metadata={"help": "Data type for local embedding model."}
    )

    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."},
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."
        },
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={
            "help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."
        },
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."},
    )
    is_directed_graph: bool = field(
        default=False, metadata={"help": "Whether the graph is directed or not."}
    )

    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"},
    )
    retrieval_top_k: int = field(
        default=200, metadata={"help": "Retrieving k documents at each step"}
    )
    retrieval_recall_k_list: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "Optional list of k values to compute Recall@k during retrieval evaluation. "
            "Example: [1, 2, 5, 10, 20, 50]. If None, uses the library default."
        },
    )
    damping: float = field(
        default=0.5, metadata={"help": "Damping factor for ppr algorithm."}
    )

    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={
            "help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."
        },
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."},
    )

    # Save dir (highest level directory)
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."
        },
    )

    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[
        Literal["hotpotqa", "hotpotqa_train", "musique", "2wikimultihopqa"]
    ] = field(
        default=None,
        metadata={
            "help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."
        },
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        "dpr_only",
        "entity",
        "passage_entity",
        "relation_aware_passage_entity",
        "passage_entity_relation",
        "facts_and_sim_passage_node_unidirectional",
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."},
    )
    corpus_len: Optional[int] = field(
        default=None, metadata={"help": "Length of the corpus to use."}
    )

    def __post_init__(self):
        if self.save_dir is None:  # If save_dir not given
            if self.dataset is None:
                self.save_dir = "outputs"  # running freely
            else:
                self.save_dir = os.path.join(
                    "outputs", self.dataset
                )  # customize your dataset's output dir here
        logger.debug(
            f"Initializing the highest level of save_dir to be {self.save_dir}"
        )
