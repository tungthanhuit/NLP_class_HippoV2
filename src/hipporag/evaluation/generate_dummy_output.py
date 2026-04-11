"""
Generate dummy HippoRAG output from ground-truth datasets.

Used to create upper-bound baselines (perfect answers, controlled retrieval) or
empty-answer scaffolds for manual annotation, without running the full HippoRAG pipeline.

Supported input formats (auto-detected):
    - HotPotQA / 2WikiMultiHopQA : fields `_id`, `answer`, `supporting_facts`, `context`
    - MuSiQue                    : fields `id`, `answer`, `paragraphs` (with `is_supporting` flag)

Output schema (one record per question):
    - "id"             : str       — question identifier from ground truth
    - "question"       : str       — question text
    - "pred_answer"    : str       — gold answer (or empty string if --empty-answers)
    - "retrieved_docs" : list[str] — docs selected by retrieval strategy
    - "gold_answers"   : list[str] — accepted answers (including aliases for MuSiQue)
    - "gold_docs"      : list[str] — supporting documents

Retrieval strategies:
    all         : all context documents (upper bound for retrieval recall)
    first5      : first 5 documents
    random      : 5 random documents
    gold+random : gold docs + random padding up to 5

Usage:
    python -m src.hipporag.evaluation.generate_dummy_output \
        --ground-truth reproduce/dataset/hotpotqa.json \
        --output dummy-output.json \
        --limit 10 \
        --retrieval-strategy gold+random

Output is compatible with src.hipporag.evaluation.evaluate (--output flag).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Format detection & parsing
# ---------------------------------------------------------------------------

def _is_hotpot_format(records: list[dict]) -> bool:
    """Detect HotPotQA/2WikiMultiHopQA format."""
    if not records:
        return False
    sample = records[0]
    return "_id" in sample and "answer" in sample and "supporting_facts" in sample and "context" in sample


def _is_musique_format(records: list[dict]) -> bool:
    """Detect MuSiQue format."""
    if not records:
        return False
    sample = records[0]
    return "id" in sample and "answer" in sample and "paragraphs" in sample and isinstance(sample["paragraphs"], list)


def _build_doc_text(title: str, sentences: list[str]) -> str:
    return f"{title}\n{''.join(sentences)}"


def _parse_hotpot_record(rec: dict) -> dict[str, Any]:
    """Parse HotPotQA/2WikiMultiHopQA record to standard format."""
    context_map = {title: sentences for title, sentences in rec.get("context", [])}

    # All docs in context (for retrieval strategy)
    all_docs = []
    for title, sentences in rec.get("context", []):
        all_docs.append(_build_doc_text(title, sentences))

    # Gold docs (supporting facts)
    supporting_titles = {title for title, _ in rec.get("supporting_facts", [])}
    gold_docs = []
    for title in supporting_titles:
        if title in context_map:
            gold_docs.append(_build_doc_text(title, context_map[title]))

    return {
        "id": rec["_id"],
        "question": rec.get("question", ""),
        "answer": rec.get("answer", ""),
        "all_docs": all_docs,
        "gold_docs": gold_docs,
    }


def _parse_musique_record(rec: dict) -> dict[str, Any]:
    """Parse MuSiQue record to standard format."""
    all_docs = []
    gold_docs = []

    for para in rec.get("paragraphs", []):
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        if title and text:
            doc_text = f"{title}\n{text}"
            all_docs.append(doc_text)
            if para.get("is_supporting"):
                gold_docs.append(doc_text)

    answer_aliases = rec.get("answer_aliases", [])
    gold_answers_list = [rec["answer"]] + answer_aliases if rec.get("answer") else answer_aliases

    return {
        "id": rec["id"],
        "question": rec.get("question", ""),
        "answer": rec.get("answer", ""),
        "all_docs": all_docs,
        "gold_docs": gold_docs,
        "answer_aliases": answer_aliases,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")

    if text.startswith("["):
        return json.loads(text)

    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return records


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def _retrieve_all(all_docs: list[str]) -> list[str]:
    """Return all documents."""
    return all_docs


def _retrieve_first_n(all_docs: list[str], k: int = 5) -> list[str]:
    """Return first k documents."""
    return all_docs[:k]


def _retrieve_random(all_docs: list[str], k: int = 5) -> list[str]:
    """Return k random documents."""
    if len(all_docs) <= k:
        return all_docs
    return random.sample(all_docs, k)


def _retrieve_gold_plus_random(all_docs: list[str], gold_docs: list[str], k: int = 5) -> list[str]:
    """Return gold docs + random docs to reach k."""
    if len(all_docs) <= k:
        return all_docs

    gold_set = set(gold_docs)
    other_docs = [d for d in all_docs if d not in gold_set]

    result = gold_docs.copy()
    remaining = k - len(result)
    if remaining > 0 and other_docs:
        result.extend(random.sample(other_docs, min(remaining, len(other_docs))))

    return result[:k]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dummy_records(
    ground_truth_path: Path,
    limit: int,
    retrieval_strategy: Literal["all", "first5", "random", "gold+random"],
    use_gold_answers: bool = True,
) -> list[dict[str, Any]]:
    """Generate dummy output records from ground truth."""
    gt_records = _load_records(ground_truth_path)

    if not gt_records:
        raise ValueError("Ground truth file is empty")

    # Detect format and parse records
    if _is_hotpot_format(gt_records):
        parsed = [_parse_hotpot_record(r) for r in gt_records]
        dataset_name = "HotPotQA"
    elif _is_musique_format(gt_records):
        parsed = [_parse_musique_record(r) for r in gt_records]
        dataset_name = "MuSiQue"
    else:
        raise ValueError(
            "Unknown ground truth format. "
            "Supported: HotPotQA, 2WikiMultiHopQA, MuSiQue"
        )

    print(f"[info] Detected format: {dataset_name}", file=sys.stderr)
    print(f"[info] Loaded {len(parsed)} records", file=sys.stderr)

    # Truncate to limit
    parsed = parsed[:limit]
    print(f"[info] Using first {len(parsed)} records", file=sys.stderr)

    # Select retrieval strategy
    if retrieval_strategy == "all":
        retrieve_fn = _retrieve_all
    elif retrieval_strategy == "first5":
        retrieve_fn = lambda docs: _retrieve_first_n(docs, 5)
    elif retrieval_strategy == "random":
        retrieve_fn = lambda docs: _retrieve_random(docs, 5)
    elif retrieval_strategy == "gold+random":
        retrieve_fn = lambda docs: _retrieve_gold_plus_random(docs, parsed[0]["gold_docs"], 5)
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")

    # Generate output
    output = []
    for rec in parsed:
        retrieved_docs = retrieve_fn(rec["all_docs"])

        dummy_rec = {
            "id": rec["id"],
            "question": rec["question"],
            "pred_answer": rec["answer"] if use_gold_answers else "",
            "retrieved_docs": retrieved_docs,
            "gold_answers": [rec["answer"]],
            "gold_docs": rec["gold_docs"],
        }
        output.append(dummy_rec)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dummy HippoRAG output from ground-truth datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ground-truth", "-gt",
        required=True,
        metavar="FILE",
        help="Ground-truth file (HotPotQA, 2WikiMultiHopQA, or MuSiQue format).",
    )
    parser.add_argument(
        "--output", "-o",
        default="dummy-output.json",
        metavar="FILE",
        help="Output JSON file (default: dummy-output.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Number of records to generate (default: 10).",
    )
    parser.add_argument(
        "--retrieval-strategy",
        choices=["all", "first5", "random", "gold+random"],
        default="all",
        metavar="STRATEGY",
        help="How to select retrieved_docs from context: "
             "'all' (use all), 'first5', 'random', 'gold+random' (gold docs + random padding). "
             "Default: all",
    )
    parser.add_argument(
        "--gold-answers",
        action="store_true",
        default=True,
        help="Use gold answer as pred_answer (default: True).",
    )
    parser.add_argument(
        "--empty-answers",
        dest="gold_answers",
        action="store_false",
        help="Leave pred_answer empty (for manual annotation).",
    )
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    output_path = Path(args.output)

    if not gt_path.exists():
        print(f"[error] Ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    records = generate_dummy_records(
        gt_path,
        args.limit,
        args.retrieval_strategy,
        args.gold_answers,
    )

    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"[info] Wrote {len(records)} records to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
