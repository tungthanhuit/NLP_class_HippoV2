"""
Standalone evaluation script for HippoRAG outputs.

Usage:
    python -m src.hipporag.evaluation.evaluate --output <hipporag_output> --ground-truth <ground_truth>

    # Or with a single combined file (already contains gold_answers/gold_docs):
    python -m src.hipporag.evaluation.evaluate --output output.json

Input formats
-------------
Two output formats are accepted interchangeably:

HippoRAG native output (from QuerySolution.to_dict()), each record must contain:
    - "id"          : str       — unique question identifier (may be null; used for GT matching)
    - "question"    : str       — the question text
    - "answer"      : str       — model's predicted answer
    - "docs"        : list[str] — retrieved passages (ordered by rank)

Dummy output format (from generate_dummy_output.py), each record must contain:
    - "id"             : str       — unique question identifier
    - "question"       : str       — the question text
    - "pred_answer"    : str       — model's predicted answer  (alias for "answer")
    - "retrieved_docs" : list[str] — retrieved passages        (alias for "docs")

Note: "pred_answer" and "answer" are interchangeable; so are "retrieved_docs" and "docs".
      "pred_answer"/"retrieved_docs" take priority if both are present.

Ground truth file (JSON or JSONL), each record must contain:
    - "id"           : str        — matches the output file's id
    - "gold_answers" : list[str]  — one or more accepted answers
    - "gold_docs"    : list[str]  — supporting documents (passage texts)

Supported ground truth formats (auto-detected):
    - HotPotQA / 2WikiMultiHopQA : fields `_id`, `answer`, `supporting_facts`, `context`
    - MuSiQue                    : fields `id`, `answer`, `paragraphs` (with `is_supporting` flag)

If --ground-truth is omitted, the output file must already contain
"gold_answers" and "gold_docs" fields.

Metrics computed
----------------
Retrieval phase : Recall@2, Recall@5
QA phase        : Exact Match (EM), F1
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Text normalisation (MRQA / SQuAD convention)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lower-case, remove punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _exact_match(gold: str, pred: str) -> float:
    return 1.0 if _normalize_answer(gold) == _normalize_answer(pred) else 0.0


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    pred_tokens = _normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(
    gold_answers: list[list[str]],
    pred_answers: list[str],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return (overall, per_example) dicts for EM and F1."""
    per_example: list[dict[str, float]] = []
    total_em = total_f1 = 0.0

    for golds, pred in zip(gold_answers, pred_answers):
        em = max(_exact_match(g, pred) for g in golds)
        f1 = max(_token_f1(g, pred) for g in golds)
        per_example.append({"ExactMatch": em, "F1": f1})
        total_em += em
        total_f1 += f1

    n = len(gold_answers)
    overall = {
        "ExactMatch": round(total_em / n, 4) if n else 0.0,
        "F1":         round(total_f1 / n, 4) if n else 0.0,
    }
    return overall, per_example


def compute_recall(
    gold_docs: list[list[str]],
    retrieved_docs: list[list[str]],
    k_list: list[int] = (2, 5),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return (overall, per_example) dicts for Recall@k."""
    k_list = sorted(set(k_list))
    overall = {f"Recall@{k}": 0.0 for k in k_list}
    per_example: list[dict[str, float]] = []

    for golds, retrieved in zip(gold_docs, retrieved_docs):
        result: dict[str, float] = {}
        gold_set = set(golds)
        for k in k_list:
            top_k = set(retrieved[:k])
            recall = len(top_k & gold_set) / len(gold_set) if gold_set else 0.0
            result[f"Recall@{k}"] = recall
            overall[f"Recall@{k}"] += recall
        per_example.append(result)

    n = len(gold_docs)
    overall = {key: round(val / n, 4) if n else 0.0 for key, val in overall.items()}
    return overall, per_example


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")

    # Try JSON array first
    if text.startswith("["):
        return json.loads(text)

    # Try JSONL
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


def _index_by_id(records: list[dict], id_field: str = "id") -> dict[str, dict]:
    index: dict[str, dict] = {}
    for rec in records:
        key = str(rec.get(id_field, ""))
        if key in index:
            print(f"[warn] Duplicate id '{key}' — keeping first occurrence.", file=sys.stderr)
        else:
            index[key] = rec
    return index


# ---------------------------------------------------------------------------
# HotPotQA ground-truth parser
# ---------------------------------------------------------------------------

def _is_hotpot_format(records: list[dict]) -> bool:
    """Detect HotPotQA/2WikiMultiHopQA format by checking for its characteristic fields."""
    if not records:
        return False
    sample = records[0]
    return "_id" in sample and "answer" in sample and "supporting_facts" in sample and "context" in sample


def _is_musique_format(records: list[dict]) -> bool:
    """Detect MuSiQue format by checking for its characteristic fields."""
    if not records:
        return False
    sample = records[0]
    return "id" in sample and "answer" in sample and "paragraphs" in sample and isinstance(sample["paragraphs"], list)


def _build_doc_text(title: str, sentences: list[str]) -> str:
    return f"{title}\n{''.join(sentences)}"


def _parse_hotpot_ground_truth(records: list[dict]) -> list[dict]:
    """Convert HotPotQA records to standard ground-truth format."""
    result = []
    for rec in records:
        context_map = {title: sentences for title, sentences in rec.get("context", [])}
        supporting_titles = {title for title, _ in rec.get("supporting_facts", [])}

        gold_docs = []
        for title in supporting_titles:
            if title in context_map:
                gold_docs.append(_build_doc_text(title, context_map[title]))

        result.append({
            "id": rec["_id"],
            "gold_answers": [rec["answer"]],
            "gold_docs": gold_docs,
        })
    return result


def _parse_musique_ground_truth(records: list[dict]) -> list[dict]:
    """Convert MuSiQue records to standard ground-truth format."""
    result = []
    for rec in records:
        gold_docs = []
        for para in rec.get("paragraphs", []):
            if para.get("is_supporting"):
                title = para.get("title", "")
                text = para.get("paragraph_text", "")
                if title and text:
                    gold_docs.append(f"{title}\n{text}")

        answer_aliases = rec.get("answer_aliases", [])
        answers = [rec["answer"]] + answer_aliases if rec.get("answer") else answer_aliases

        result.append({
            "id": rec["id"],
            "gold_answers": answers if answers else [rec.get("answer", "")],
            "gold_docs": gold_docs,
        })
    return result


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(
    output_path: Path,
    ground_truth_path: Path | None,
    k_list: list[int],
) -> dict[str, Any]:
    output_records = _load_records(output_path)

    if ground_truth_path is not None:
        gt_records = _load_records(ground_truth_path)
        if _is_hotpot_format(gt_records):
            gt_records = _parse_hotpot_ground_truth(gt_records)
        elif _is_musique_format(gt_records):
            gt_records = _parse_musique_ground_truth(gt_records)
        gt_index = _index_by_id(gt_records)

        merged: list[dict] = []
        skipped = 0
        for rec in output_records:
            qid = str(rec.get("id", ""))
            gt = gt_index.get(qid)
            if gt is None:
                print(f"[warn] id '{qid}' not found in ground-truth file — skipping.", file=sys.stderr)
                skipped += 1
                continue
            merged.append({**rec, "gold_answers": gt["gold_answers"], "gold_docs": gt["gold_docs"]})

        if skipped:
            print(f"[warn] {skipped} records skipped (no matching ground-truth).", file=sys.stderr)

        # Fallback: if no matches found but output has gold fields, use them
        if not merged and skipped > 0:
            print(
                f"[warn] No matching records between output and ground-truth. "
                f"Checking if output file already contains gold_answers/gold_docs...",
                file=sys.stderr
            )
            # Try to use gold fields from output (if present)
            if all("gold_answers" in r and "gold_docs" in r for r in output_records):
                merged = output_records
                print(f"[info] Using gold fields from output file for {len(merged)} records.", file=sys.stderr)
            else:
                raise ValueError(
                    "No records matched between output and ground-truth file. "
                    "Either: (1) fix ID mismatch, or (2) ensure output file contains gold_answers/gold_docs fields."
                )
    else:
        # Combined format: gold fields already present
        merged = output_records
        for i, rec in enumerate(merged):
            if "gold_answers" not in rec or "gold_docs" not in rec:
                raise ValueError(
                    f"Record {i} (id={rec.get('id')!r}) is missing 'gold_answers' or 'gold_docs'. "
                    "Provide a --ground-truth file or use a combined output file."
                )

    if not merged:
        raise ValueError("No records to evaluate.")

    gold_answers   = [r["gold_answers"]                            for r in merged]
    pred_answers   = [r.get("pred_answer") or r.get("answer", "") for r in merged]
    gold_docs      = [r["gold_docs"]                              for r in merged]
    retrieved_docs = [r.get("retrieved_docs") or r.get("docs", []) for r in merged]

    qa_overall, qa_per_example       = compute_qa_metrics(gold_answers, pred_answers)
    ret_overall, ret_per_example     = compute_recall(gold_docs, retrieved_docs, k_list)

    return {
        "num_examples": len(merged),
        "retrieval_overall": ret_overall,
        "qa_overall": qa_overall,
        "retrieval_per_example": ret_per_example,
        "qa_per_example": qa_per_example,
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def _print_summary(result: dict[str, Any]) -> None:
    n = result["num_examples"]
    ret = result["retrieval_overall"]
    qa  = result["qa_overall"]

    width = 42
    sep   = "─" * width

    print(sep)
    print(f"  HippoRAG Evaluation  ({n} examples)")
    print(sep)
    print("  [Retrieval]")
    for key, val in sorted(ret.items()):
        print(f"    {key:<12}  {val:.4f}  ({val * 100:.2f}%)")
    print()
    print("  [QA]")
    print(f"    {'ExactMatch':<12}  {qa['ExactMatch']:.4f}  ({qa['ExactMatch'] * 100:.2f}%)")
    print(f"    {'F1':<12}  {qa['F1']:.4f}  ({qa['F1'] * 100:.2f}%)")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HippoRAG outputs: Recall@k (retrieval) + EM / F1 (QA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="FILE",
        help="HippoRAG output file (JSON array or JSONL). "
             "Must contain 'id', 'pred_answer', 'retrieved_docs' per record.",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        metavar="FILE",
        default=None,
        help="Ground-truth file (JSON array or JSONL). "
             "Must contain 'id', 'gold_answers', 'gold_docs' per record. "
             "Omit if the output file already contains these fields.",
    )
    parser.add_argument(
        "--k", "-k",
        nargs="+",
        type=int,
        default=[2, 5],
        metavar="K",
        help="k values for Recall@k (default: 2 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full results as JSON instead of the summary table.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    gt_path = Path(args.ground_truth) if args.ground_truth else None

    if not output_path.exists():
        print(f"[error] Output file not found: {output_path}", file=sys.stderr)
        sys.exit(1)
    if gt_path is not None and not gt_path.exists():
        print(f"[error] Ground-truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    result = evaluate(output_path, gt_path, args.k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_summary(result)


if __name__ == "__main__":
    main()
