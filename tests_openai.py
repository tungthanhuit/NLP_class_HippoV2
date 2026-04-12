import os

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def _print_qa_round(round_name, query_solutions):
    print(f"\n=== {round_name}: Questions & Answers ===")
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

    save_dir = "outputs/openai_test"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
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
    print("Initial indexing complete.")

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
    _print_qa_round("First round", query_solutions)
    print("\n=== First round: Metrics ===")
    print(overall_retrieval_result)
    print(overall_qa_results)
    print("First round.")

    # Startup a HippoRAG instance
    hipporag = HippoRAG(global_config=cfg)

    result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    query_solutions = result[0]
    overall_retrieval_result = result[3] if len(result) > 3 else None
    overall_qa_results = result[4] if len(result) > 4 else None
    _print_qa_round("Second round", query_solutions)
    print("\n=== Second round: Metrics ===")
    print(overall_retrieval_result)
    print(overall_qa_results)
    print("Second round.")

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

    result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    query_solutions = result[0]
    overall_retrieval_result = result[3] if len(result) > 3 else None
    overall_qa_results = result[4] if len(result) > 4 else None
    _print_qa_round("Third round", query_solutions)
    print("\n=== Third round: Metrics ===")
    print(overall_retrieval_result)
    print(overall_qa_results)
    print("Third round done.")

    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County..",
    ]

    hipporag.delete(docs_to_delete)

    result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    query_solutions = result[0]
    overall_retrieval_result = result[3] if len(result) > 3 else None
    overall_qa_results = result[4] if len(result) > 4 else None
    _print_qa_round("Final round", query_solutions)
    print("\n=== Final round: Metrics ===")
    print(overall_retrieval_result)
    print(overall_qa_results)
    print("Final round.")


if __name__ == "__main__":
    main()
