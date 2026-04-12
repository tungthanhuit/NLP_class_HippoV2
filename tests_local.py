import os

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.misc_utils import QuerySolution


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
        "Marina is born in Minsk.",
        "Montebello is a part of Rockland County.",
    ]
    save_dir = "outputs/local_test"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_base_url = os.getenv("HIPPORAG_LLM_BASE_URL", "http://localhost:4000/v1")
    embedding_base_url = (
        os.getenv("HIPPORAG_EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or llm_base_url
    )
    embedding_trust_remote_code = string_to_bool(
        os.getenv("HIPPORAG_EMBEDDING_TRUST_REMOTE_CODE", "false")
    )

    cfg = BaseConfig(
        save_dir=save_dir,
        llm_name=os.getenv("HIPPORAG_LLM_NAME", "gpt-4o-mini"),
        llm_base_url=llm_base_url,
        embedding_model_name=os.getenv(
            "HIPPORAG_EMBEDDING_MODEL_NAME",
            "Transformers/sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_base_url=embedding_base_url,
        embedding_trust_remote_code=embedding_trust_remote_code,
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config=cfg)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries: list[str | QuerySolution] = [
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

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config=cfg)

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config=cfg)

    new_docs = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County..",
    ]

    # Run indexing
    hipporag.index(docs=new_docs)

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )

    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County..",
    ]

    hipporag.delete(docs_to_delete)

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )


if __name__ == "__main__":
    main()
