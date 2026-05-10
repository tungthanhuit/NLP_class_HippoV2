import os
from typing import List
import json
from datetime import datetime

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name.startswith("hotpotqa"):
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
    parser.add_argument(
        "--mode",
        choices=["all", "index", "qa"],
        default="all",
        help="Run mode: 'index' to build index only, 'qa' to run QA pipeline only, 'all' to do both",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help="Path to a custom query JSON file (same format as dataset). Defaults to reproduce/dataset/{dataset}.json",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Run only this single sample index from the query file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--use_enhancements",
        action="store_true",
        default=False,
        help="Enable Enhancement 1 (query decomposition + RRF) and Enhancement 2 (coverage audit). "
             "Omit to run the standard pipeline for comparison.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
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

    # Prepare datasets and evaluation
    query_file = args.query_file or f"reproduce/dataset/{dataset_name}.json"
    samples = json.load(open(query_file, "r"))

    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]]

    all_queries = [s["question"] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert (
            len(all_queries) == len(gold_docs) == len(gold_answers)
        ), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
        use_enhancements=args.use_enhancements,
    )

    log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_fmt, datefmt=log_datefmt)

    os.makedirs(save_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"run_{run_ts}.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_level = getattr(logging, args.log_level.upper())

    # Root logger at WARNING so network/file helpers (httpx, openai, urllib3,
    # httpcore, requests, asyncio, parso, dspy) stay silent by default.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Apply the user-chosen level only to project loggers.
    for name in ("src.hipporag", "__main__"):
        logging.getLogger(name).setLevel(log_level)

    # Silence specific noisy third-party loggers regardless of log_level.
    for noisy in ("httpx", "httpcore", "openai", "anthropic",
                  "urllib3", "requests", "asyncio", "dspy",
                  "parso", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("__main__").info(f"Logging to {log_path}")

    hipporag = HippoRAG(global_config=config)

    if args.mode in ("all", "index"):
        hipporag.index(docs)

    if args.mode in ("all", "qa"):
        hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
