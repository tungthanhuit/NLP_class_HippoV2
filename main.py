import os
from typing import List, Optional
import json

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging


def get_gold_docs(samples: List, dataset_name: Optional[str] = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name and dataset_name.startswith("hotpotqa"):
                gold_doc = [
                    item[0] + "\n" + "".join(item[1])
                    for item in gold_title_and_content_list
                ]
            else:
                gold_doc = [
                    item[0] + "\n" + " ".join(item[1])
                    for item in gold_title_and_content_list
                ]
        elif "contexts" in sample:
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            assert (
                "paragraphs" in sample
            ), "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                item["title"]
                + "\n"
                + (item["text"] if "text" in item else item["paragraph_text"])
                for item in gold_paragraphs
            ]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            gold_ans = set(
                [sample["obj"]]
                + [sample["possible_answers"]]
                + [sample["o_wiki_title"]]
                + [sample["o_aliases"]]
            )
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if "answer_aliases" in sample:
            gold_ans.update(sample["answer_aliases"])

        gold_answers.append(gold_ans)

    return gold_answers


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset name")
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://localhost:4000/v1",
        help="LLM base URL",
    )
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", help="LLM name")
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default=None,
        help=(
            "Embedding model base URL (OpenAI-compatible). "
            "If omitted, defaults to env EMBEDDING_BASE_URL, else --llm_base_url."
        ),
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default="text-embedding-3-small",
        help="embedding model name",
    )
    parser.add_argument(
        "--force_index_from_scratch",
        type=str,
        default="false",
        help="If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.",
    )
    parser.add_argument(
        "--force_openie_from_scratch",
        type=str,
        default="false",
        help="If set to False, will try to first reuse openie results for the corpus if they exist.",
    )
    parser.add_argument(
        "--openie_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode: offline runs local Transformers-based OpenIE for indexing; online uses the configured LLM API",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Save directory"
    )

    # Storage backends
    # NOTE: BaseConfig defaults to kb_backend='parquet' and chunk_vector_backend='default'.
    # This script defaults to Neo4j+Milvus because it is commonly used for large-scale runs.
    # If you want to reuse/share the same KB across runs or LLMs, control it via the
    # Neo4j namespace flags below (namespace and whether to include llm_name).
    parser.add_argument(
        "--kb_backend",
        choices=["parquet", "neo4j"],
        default="neo4j",
        help="KB persistence backend. 'parquet' stores locally; 'neo4j' stores in Neo4j.",
    )
    parser.add_argument(
        "--chunk_vector_backend",
        choices=["default", "milvus"],
        default="milvus",
        help="Chunk/passage vector backend. 'default' uses the KB backend; 'milvus' uses Milvus.",
    )

    # Neo4j/Milvus runtime overrides
    parser.add_argument(
        "--neo4j_namespace",
        type=str,
        default=None,
        help="Neo4j namespace prefix. Defaults to env NEO4J_NAMESPACE or 'hipporag'.",
    )
    parser.add_argument(
        "--neo4j_namespace_include_llm",
        type=str,
        default="true",
        help="If false, reuse the same Neo4j KB across different --llm_name values (embedding model still remains part of the namespace).",
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default=None,
        help="Neo4j Bolt URI. Defaults to env NEO4J_URI or 'bolt://localhost:7687'.",
    )
    parser.add_argument(
        "--neo4j_user",
        type=str,
        default=None,
        help="Neo4j username. Defaults to env NEO4J_USER or 'neo4j'.",
    )
    parser.add_argument(
        "--neo4j_password",
        type=str,
        default=None,
        help="Neo4j password. Defaults to env NEO4J_PASSWORD.",
    )
    parser.add_argument(
        "--neo4j_database",
        type=str,
        default=None,
        help="Neo4j database name. Defaults to env NEO4J_DATABASE or 'neo4j'.",
    )
    parser.add_argument(
        "--milvus_uri",
        type=str,
        default=None,
        help="Milvus URI. Defaults to env MILVUS_URI or 'http://localhost:19530'.",
    )
    parser.add_argument(
        "--milvus_token",
        type=str,
        default=None,
        help="Milvus token. Defaults to env MILVUS_TOKEN.",
    )

    parser.add_argument(
        "--retrieval_recall_k_list",
        type=str,
        default=None,
        help=(
            "Comma-separated list of k values for retrieval evaluation Recall@k. "
            "Example: '1,2,5,10,20,50'. If omitted, uses defaults (auto-clipped to retrieved size)."
        ),
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    embedding_base_url = (
        args.embedding_base_url
        or os.getenv("EMBEDDING_BASE_URL")
        or llm_base_url
    )
    if save_dir == "outputs":
        save_dir = save_dir + "/" + dataset_name
    else:
        save_dir = save_dir + "_" + dataset_name

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    neo4j_namespace_include_llm = string_to_bool(args.neo4j_namespace_include_llm)
    neo4j_namespace = (args.neo4j_namespace or os.getenv("NEO4J_NAMESPACE") or "hipporag").strip()
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER") or "neo4j"
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    neo4j_database = args.neo4j_database or os.getenv("NEO4J_DATABASE") or "neo4j"
    milvus_uri = args.milvus_uri or os.getenv("MILVUS_URI") or "http://localhost:19530"
    milvus_token = args.milvus_token or os.getenv("MILVUS_TOKEN")

    retrieval_recall_k_list = None
    if args.retrieval_recall_k_list is not None:
        raw = args.retrieval_recall_k_list.strip()
        if raw:
            try:
                if raw.startswith("["):
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        raise ValueError("Expected a JSON list")
                    retrieval_recall_k_list = [int(x) for x in parsed]
                else:
                    retrieval_recall_k_list = [
                        int(x.strip()) for x in raw.split(",") if x.strip()
                    ]
            except Exception:
                parser.error(
                    "--retrieval_recall_k_list must be a comma-separated list like '1,2,5,10' or a quoted JSON list like '[1,2,5,10]'."
                )

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s["question"] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert (
            len(all_queries) == len(gold_docs) == len(gold_answers)
        ), "Length of queries, gold_docs, and gold_answers should be the same."
    except Exception:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        kb_backend=args.kb_backend,
        llm_base_url=llm_base_url,
        chunk_vector_backend=args.chunk_vector_backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_base_url=embedding_base_url,
        embedding_model_name=args.embedding_name,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        neo4j_namespace=neo4j_namespace,
        neo4j_namespace_include_llm=neo4j_namespace_include_llm,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        retrieval_recall_k_list=retrieval_recall_k_list,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
