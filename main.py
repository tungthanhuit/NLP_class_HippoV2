import os
from typing import List, Optional
import json
import time

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score

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


def log_retriever_response(
    retrieval_results,
    logger: logging.Logger,
    top_k: int = 3,
    preview_chars: int = 180,
):
    top_k = max(0, int(top_k))
    preview_chars = max(0, int(preview_chars))
    if top_k == 0:
        return

    for q_idx, retrieval_result in enumerate(retrieval_results):
        previews = []
        docs = retrieval_result.docs or []
        scores = retrieval_result.doc_scores
        limit = min(top_k, len(docs))
        for i in range(limit):
            doc_text = docs[i]
            title, _, body = doc_text.partition("\n")
            snippet = body.replace("\n", " ").strip()
            if preview_chars > 0:
                snippet = snippet[:preview_chars]

            score = None
            if scores is not None and len(scores) > i:
                try:
                    score = float(scores[i])
                except Exception:
                    score = None

            previews.append(
                {
                    "rank": i + 1,
                    "score": score,
                    "title": title,
                    "snippet": snippet,
                }
            )

        logger.info(
            "[Retriever Response][Q%d] %s",
            q_idx,
            json.dumps(previews, ensure_ascii=False),
        )


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
    parser.add_argument(
        "--ppr_mode",
        choices=["global", "teleportation_hybrid"],
        default="global",
        help=(
            "PPR backend mode. 'global' runs full-graph PPR. "
            "'teleportation_hybrid' uses community-local leaky PPR with bridge-triggered expansion."
        ),
    )
    parser.add_argument(
        "--teleportation_leakage_gamma",
        type=float,
        default=0.15,
        help="Leakage coefficient gamma for teleportation_hybrid mode.",
    )
    parser.add_argument(
        "--teleportation_trigger_threshold",
        type=float,
        default=0.003,
        help="Bridge activation threshold tau for teleportation triggers.",
    )
    parser.add_argument(
        "--teleportation_home_communities_top_k",
        type=int,
        default=2,
        help="Initial number of active home communities in teleportation_hybrid mode.",
    )
    parser.add_argument(
        "--teleportation_max_teleport_steps",
        type=int,
        default=3,
        help="Maximum number of community teleport expansions per query.",
    )
    parser.add_argument(
        "--teleportation_max_iterations",
        type=int,
        default=50,
        help="Maximum sparse power-iteration steps for teleportation_hybrid mode.",
    )
    parser.add_argument(
        "--teleportation_tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance for teleportation_hybrid sparse PPR.",
    )
    parser.add_argument(
        "--teleportation_bridge_betweenness_quantile",
        type=float,
        default=0.95,
        help="Betweenness quantile (0-1) for tagging additional bridge entities.",
    )
    parser.add_argument(
        "--teleportation_min_bridge_entities_per_chunk",
        type=int,
        default=2,
        help="Minimum bridge entities in a chunk to mark it as a bridge chunk.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help=(
            "If provided, run retrieval+QA for a single sample index from the dataset "
            "(0-based). Useful for pipeline debugging."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--allow_milvus_fallback_for_debug",
        type=str,
        default="true",
        help=(
            "If true, automatically fall back to chunk_vector_backend='default' "
            "when Milvus is unavailable."
        ),
    )
    parser.add_argument(
        "--quiet_external_logs",
        type=str,
        default="true",
        help=(
            "If true, silence noisy third-party loggers (neo4j, pymilvus, httpx, etc.) "
            "to focus on HippoRAG pipeline logs."
        ),
    )
    parser.add_argument(
        "--retrieval_log_top_k",
        type=int,
        default=3,
        help="Number of top retrieved passages to print per query in retrieval logs.",
    )
    parser.add_argument(
        "--retrieval_log_preview_chars",
        type=int,
        default=180,
        help="Preview length (chars) for each retrieved passage snippet in retrieval logs.",
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
    allow_milvus_fallback_for_debug = string_to_bool(args.allow_milvus_fallback_for_debug)
    quiet_external_logs = string_to_bool(args.quiet_external_logs)

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

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(samples):
            parser.error(
                f"--sample_index must be within [0, {len(samples) - 1}] for dataset {dataset_name}."
            )
        samples = [samples[args.sample_index]]

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
        ppr_mode=args.ppr_mode,
        teleportation_leakage_gamma=args.teleportation_leakage_gamma,
        teleportation_trigger_threshold=args.teleportation_trigger_threshold,
        teleportation_home_communities_top_k=args.teleportation_home_communities_top_k,
        teleportation_max_teleport_steps=args.teleportation_max_teleport_steps,
        teleportation_max_iterations=args.teleportation_max_iterations,
        teleportation_tolerance=args.teleportation_tolerance,
        teleportation_bridge_betweenness_quantile=args.teleportation_bridge_betweenness_quantile,
        teleportation_min_bridge_entities_per_chunk=args.teleportation_min_bridge_entities_per_chunk,
    )
    config.retrieval_log_top_k = max(0, int(args.retrieval_log_top_k))
    config.retrieval_log_preview_chars = max(0, int(args.retrieval_log_preview_chars))

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    if quiet_external_logs:
        noisy_logger_names = [
            "neo4j",
            "neo4j.io",
            "neo4j.pool",
            "pymilvus",
            "grpc",
            "httpx",
            "openai",
            "urllib3",
        ]
        for name in noisy_logger_names:
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Loaded %d query sample(s) from dataset '%s'", len(samples), dataset_name)
    logger.info(
        "Focused retrieval logs: top_k=%d, preview_chars=%d",
        config.retrieval_log_top_k,
        config.retrieval_log_preview_chars,
    )
    if args.sample_index is not None:
        logger.info("Running in single-sample mode with sample_index=%d", args.sample_index)
        logger.debug("Selected question: %s", all_queries[0])

    try:
        hipporag = HippoRAG(global_config=config)
    except RuntimeError as e:
        err_msg = str(e).lower()
        should_fallback = (
            allow_milvus_fallback_for_debug
            and args.chunk_vector_backend == "milvus"
            and ("failed connecting to milvus" in err_msg or "milvusexception" in err_msg)
        )
        if not should_fallback:
            raise

        logger.warning(
            "Milvus is unavailable. "
            "Falling back to chunk_vector_backend='default' (Neo4j chunk vectors)."
        )
        config.chunk_vector_backend = "default"
        hipporag = HippoRAG(global_config=config)

    # Step 1: indexing
    pipeline_start = time.time()
    index_start = time.time()
    hipporag.index(docs)
    index_time = time.time() - index_start
    logger.info("[Timing] index=%.3fs", index_time)

    # Step 2: retrieval
    retrieval_start = time.time()
    if gold_docs is not None:
        retrieval_results, overall_retrieval_result = hipporag.retrieve(
            queries=all_queries,
            gold_docs=gold_docs,
        )
        logger.info("[Retrieval Eval] %s", overall_retrieval_result)
    else:
        retrieval_results = hipporag.retrieve(queries=all_queries)
    retrieval_time = time.time() - retrieval_start
    logger.info("[Timing] retrieval=%.3fs", retrieval_time)

    # Focused retriever output for debugging
    log_retriever_response(
        retrieval_results,
        logger,
        top_k=config.retrieval_log_top_k,
        preview_chars=config.retrieval_log_preview_chars,
    )

    # Step 3: QA generation
    qa_start = time.time()
    queries_solutions, all_response_message, all_metadata = hipporag.qa(retrieval_results)
    qa_time = time.time() - qa_start
    logger.info("[Timing] qa_generation=%.3fs", qa_time)

    # Optional QA evaluation (kept for parity with previous rag_qa flow)
    if gold_answers is not None:
        qa_em_evaluator = QAExactMatch(global_config=config)
        qa_f1_evaluator = QAF1Score(global_config=config)
        overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=[qa_result.answer for qa_result in queries_solutions],
            aggregation_fn=max,
        )
        overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
            gold_answers=gold_answers,
            predicted_answers=[qa_result.answer for qa_result in queries_solutions],
            aggregation_fn=max,
        )
        overall_qa_results = {**overall_qa_em_result, **overall_qa_f1_result}
        overall_qa_results = {
            k: round(float(v), 4) for k, v in overall_qa_results.items()
        }
        logger.info("[QA Eval] %s", overall_qa_results)

    logger.info("[Timing] total_pipeline=%.3fs", time.time() - pipeline_start)


if __name__ == "__main__":
    main()
