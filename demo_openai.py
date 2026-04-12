import os

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())  # Load .env file if present


def _print_qa(query_solutions):
    print("\n=== Questions & Answers ===")
    for i, qs in enumerate(query_solutions, start=1):
        print(f"\n[{i}] Q: {qs.question}")
        print(f"    A: {qs.answer}")


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

    save_dir = "outputs/openai"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)

    llm_base_url = os.getenv("HIPPORAG_LLM_BASE_URL", "http://localhost:4000/v1")
    embedding_base_url = (
        os.getenv("HIPPORAG_EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or llm_base_url
    )

    cfg = BaseConfig(
        save_dir=save_dir,
        llm_name=os.getenv("HIPPORAG_LLM_NAME", "gpt-4o-mini"),
        llm_base_url=llm_base_url,
        embedding_model_name=os.getenv(
            "HIPPORAG_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        embedding_base_url=embedding_base_url,
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

    result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)

    query_solutions = result[0]
    overall_retrieval_result = result[3] if len(result) > 3 else None
    overall_qa_results = result[4] if len(result) > 4 else None

    _print_qa(query_solutions)

    print("\n=== Metrics ===")
    if overall_retrieval_result is not None:
        print("Retrieval:", overall_retrieval_result)
    if overall_qa_results is not None:
        print("QA:", overall_qa_results)


if __name__ == "__main__":
    main()
