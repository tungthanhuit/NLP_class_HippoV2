import os

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def main():
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("Skipping Neo4j test: env NEO4J_PASSWORD not set")
        return

    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    ]

    cfg = BaseConfig(
        kb_backend="neo4j",
        save_dir="outputs/neo4j_test",
        llm_name=os.getenv("HIPPORAG_LLM_NAME", "gpt-4o-mini"),
        llm_base_url=os.getenv("HIPPORAG_LLM_BASE_URL", "http://localhost:4000/v1"),
        embedding_model_name=os.getenv(
            "HIPPORAG_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        embedding_base_url=os.getenv(
            "HIPPORAG_EMBEDDING_BASE_URL", "http://localhost:4000/v1"
        ),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=password,
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        neo4j_namespace=os.getenv("NEO4J_NAMESPACE", "hipporag_test"),
    )

    hipporag = HippoRAG(global_config=cfg)
    hipporag.index(docs)

    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
    ]

    out1 = hipporag.rag_qa(queries=queries)
    print("Round 1 OK", [qs.answer for qs in out1[0]])

    hipporag2 = HippoRAG(global_config=cfg)
    out2 = hipporag2.rag_qa(queries=queries)
    print("Round 2 OK", [qs.answer for qs in out2[0]])

    hipporag2.delete(["Thomas Marwick is a politician."])
    out3 = hipporag2.rag_qa(queries=queries)
    print("Round 3 OK", [qs.answer for qs in out3[0]])


if __name__ == "__main__":
    main()
