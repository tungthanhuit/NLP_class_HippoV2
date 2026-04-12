import os

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool


def main():

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County.",
    ]

    save_dir = "outputs/demo_llama"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_base_url = os.getenv("HIPPORAG_LLM_BASE_URL", "http://localhost:6578/v1")
    embedding_trust_remote_code = string_to_bool(
        os.getenv("HIPPORAG_EMBEDDING_TRUST_REMOTE_CODE", "false")
    )

    cfg = BaseConfig(
        save_dir=save_dir,
        llm_name=os.getenv(
            "HIPPORAG_LLM_NAME", "meta-llama/Llama-3.1-8B-Instruct"
        ),
        llm_base_url=llm_base_url,
        embedding_model_name=os.getenv(
            "HIPPORAG_EMBEDDING_MODEL_NAME",
            "Transformers/sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_trust_remote_code=embedding_trust_remote_code,
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config=cfg)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?",
    ]

    # For Evaluation
    answers = [["Politician"], ["By going to the ball."], ["Rockland County"]]

    gold_docs = [
        ["George Rankin is a politician."],
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        ],
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County.",
        ],
    ]

    print(hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers))


if __name__ == "__main__":
    main()
