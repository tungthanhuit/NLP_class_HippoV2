"""
Data preparation utilities for KET-RAG experiments.

Loads benchmark data (from HippoRAG datasets), selects experiment splits,
converts to KET-RAG format, and writes to disk.
"""

import json
import random
from pathlib import Path


def load_hipporag_dataset(dataset_dir: Path, dataset_name: str):
    """
    Load corpus and queries from datasets/ directory.
    Returns (corpus, queries) in HippoRAG's native format.
    """
    corpus_path = dataset_dir / f"{dataset_name}_corpus.json"
    queries_path = dataset_dir / f"{dataset_name}.json"

    assert corpus_path.exists(), f"Missing: {corpus_path}"
    assert queries_path.exists(), f"Missing: {queries_path}"

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    print(f"  {dataset_name}: {len(corpus)} corpus docs, {len(queries)} queries")
    return corpus, queries


def select_split(queries, corpus, n_queries, seed=42):
    """
    Select n_queries queries and relevant corpus subset.
    Uses per-question context paragraphs (gold + benchmark distractors),
    pooled and deduplicated — matching the KET-RAG paper methodology.
    For large splits (>=500): full corpus.
    """
    rng = random.Random(seed)

    if n_queries >= len(queries):
        selected_queries = queries
    else:
        selected_queries = rng.sample(queries, n_queries)

    # Pool all per-question context paragraph titles (gold + distractors)
    needed_titles = set()
    for q in selected_queries:
        if "context" in q:  # HotpotQA / 2Wiki: list of [title, sentences]
            for title, _sentences in q["context"]:
                needed_titles.add(title)
        elif "paragraphs" in q:  # MuSiQue
            for p in q["paragraphs"]:
                needed_titles.add(p["title"])

    selected_corpus = [d for d in corpus if d.get("title") in needed_titles]
    print(f"  Pooled context: {len(needed_titles)} unique titles -> {len(selected_corpus)} corpus docs")
    return selected_queries, selected_corpus


def convert_queries_to_qa_pairs(queries: list) -> list:
    """
    Normalize HippoRAG query format to KET-RAG qa-pairs format.
    Handles both _id (HotpotQA/2Wiki) and id (MuSiQue).
    """
    qa_pairs = []
    for q in queries:
        qid = str(q.get("id") or q.get("_id"))
        answer = str(q.get("answer", ""))
        aliases = q.get("answer_aliases", [])
        answers_list = [answer] + [str(a) for a in aliases if str(a) != answer]

        qa_pairs.append({
            "id": qid,
            "question": q["question"],
            "answer": answer,
            "answers": answers_list,
        })
    return qa_pairs


def write_corpus_json(target_dir: Path, corpus: list, filename: str) -> Path:
    """Write corpus docs to a JSON file in the target directory.

    `filename` should include the suffix, e.g. "hotpotqa_corpus.json".
    """
    out_path = target_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    return out_path


def write_queries_json(target_dir: Path, queries: list, filename: str) -> Path:
    """Write queries JSON file in the target directory.

    `filename` should be the dataset filename, e.g. "hotpotqa.json".
    """
    out_path = target_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    return out_path


def prepare_experiment(
    project_root: Path,
    dataset_dir: Path,
    dataset_name: str,
    split_name: str,
    split_configs: dict,
):
    """
    Full pipeline: load HippoRAG data -> select split -> convert -> write.
    Skips if already prepared.
    """
    n_queries = split_configs[split_name]["n_queries"]
    key = f"{dataset_name}/{split_name}"

    # Write outputs back into the original dataset directory using
    # split-specific filenames so we don't overwrite the originals.
    corpus_fn = f"{dataset_name}_{split_name}_corpus.json"
    queries_fn = f"{dataset_name}_{split_name}.json"
    corpus_path = dataset_dir / corpus_fn
    queries_path = dataset_dir / queries_fn

    if corpus_path.exists() and queries_path.exists():
        n_corpus = len(json.loads(corpus_path.read_text(encoding="utf-8")))
        n_queries_count = len(json.loads(queries_path.read_text(encoding="utf-8")))
        print(f"{key}: already prepared ({n_corpus} docs, {n_queries_count} queries) -- skipping")
        return

    print(f"\nPreparing {key} ...")
    corpus, queries = load_hipporag_dataset(dataset_dir, dataset_name)
    sel_queries, sel_corpus = select_split(queries, corpus, n_queries)
    qa_pairs = convert_queries_to_qa_pairs(sel_queries)

    # Write split outputs into the dataset directory alongside originals
    write_corpus_json(dataset_dir, sel_corpus, corpus_fn)
    write_queries_json(dataset_dir, qa_pairs, queries_fn)

    print(f"  -> {len(sel_corpus)} docs, {len(qa_pairs)} queries written to {dataset_dir}")
