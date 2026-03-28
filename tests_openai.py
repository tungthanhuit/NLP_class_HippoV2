
from src.hipporag import HippoRAG


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
    llm_model_name = "gpt-4o-mini"  # Any OpenAI model name
    embedding_model_name = "text-embedding-3-small"  # Embedding model name
    llm_base_url = "http://localhost:4000/v1"
    embedding_base_url = "http://localhost:4000/v1"

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
    )

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

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )
    print("First round.")

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
    )

    print(
        hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:]
    )
    print("Second round.")

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        embedding_base_url=embedding_base_url,
    )

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
    print("Third round done.")

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
    print("Final round.")


if __name__ == "__main__":
    main()
