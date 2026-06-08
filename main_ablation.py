"""
Ablation study runner for HippoRAG 2 enhancements.

Runs five configurations in sequence and prints a side-by-side comparison table:
  A: DPR only              — dense retrieval, single-shot QA
  B: HippoRAG base         — graph + reranker, no enhancements, single-shot QA
  C: HippoRAG + E1         — graph + query decomp + RRF + NER-seeded fallback, single-shot QA
  D: HippoRAG + E2         — graph (no query decomp), IRCoT hops only
  E: HippoRAG + E1+E2      — full system: E1 retrieval + IRCoT

Enhancement definitions:
  E1 = query decomposition (RAG Fusion) + RRF + NER-seeded DPR fallback
  E2 = IRCoT multi-hop reasoning loop

Usage:
  python main_ablation.py --dataset musique --llm_name gpt-4o-mini [options]
"""

import os
import json
import logging
import argparse
from typing import Dict, List

import numpy as np

from src.hipporag.EnhancedHippoRAG import EnhancedHippoRAG
from src.hipporag.StandardRAG import StandardRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [
                item for item in sample["context"] if item[0] in gold_title
            ]
            if dataset_name.startswith("hotpotqa"):
                gold_doc = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            gold_doc = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        else:
            gold_paragraphs = [
                item for item in sample.get("paragraphs", [])
                if item.get("is_supporting", True)
            ]
            gold_doc = [
                item["title"] + "\n" + (item.get("text") or item.get("paragraph_text", ""))
                for item in gold_paragraphs
            ]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def get_gold_answers(samples: List) -> List:
    gold_answers = []
    for sample in samples:
        gold_ans = sample.get("answer") or sample.get("gold_ans") or sample.get("reference")
        if gold_ans is None and "obj" in sample:
            gold_ans = list({sample["obj"]} | {sample.get("o_wiki_title", "")} | set(sample.get("o_aliases", [])))
        assert gold_ans is not None, f"No answer found in sample: {list(sample.keys())}"
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if "answer_aliases" in sample:
            gold_ans.update(sample["answer_aliases"])
        gold_answers.append(gold_ans)
    return gold_answers


ABLATION_CONFIGS = [
    {
        "name": "A: DPR only",
        "short": "dpr",
        "use_dpr": True,
        "use_enhancements": False,
        "max_qa_steps": 1,
    },
    {
        "name": "B: HippoRAG base",
        "short": "hipporag_base",
        "use_dpr": False,
        "use_enhancements": False,
        "max_qa_steps": 1,
    },
    {
        # E1: query decomp + RRF + NER-seeded fallback; single-shot QA (no IRCoT)
        "name": "C: HippoRAG + E1",
        "short": "hipporag_e1",
        "use_dpr": False,
        "use_enhancements": True,
        "max_qa_steps": 1,
    },
    {
        # E2: IRCoT hops only; initial retrieval uses plain fact scoring (no decomp)
        "name": "D: HippoRAG + E2",
        "short": "hipporag_e2",
        "use_dpr": False,
        "use_enhancements": False,
        "max_qa_steps": 4,
    },
    {
        # Full: E1 on initial retrieval + IRCoT hops
        "name": "E: HippoRAG + E1+E2",
        "short": "hipporag_e1e2",
        "use_dpr": False,
        "use_enhancements": True,
        "max_qa_steps": 4,
    },
    {
        # DPR for both initial retrieval and every IRCoT hop — no graph
        "name": "F: DPR + IRCoT",
        "short": "dpr_ircot",
        "use_dpr": True,
        "use_enhancements": False,
        "max_qa_steps": 4,
    },
]


def _pct(val, default="n/a"):
    if val is None:
        return default
    return f"{float(val):.1f}%"


def _tok(val):
    if val is None or val == 0:
        return "0"
    v = int(val)
    return f"{v:,}"


def print_comparison_table(results: List[Dict]) -> None:
    configs = [r["config"] for r in results]
    col_w = max(30, max(len(c["name"]) for c in configs) + 2)
    label_w = 32

    sep = "-" * (label_w + col_w * len(configs))
    header = " " * label_w + "".join(c["name"].ljust(col_w) for c in configs)

    def row(label, vals):
        return label.ljust(label_w) + "".join(str(v).ljust(col_w) for v in vals)

    def section(title):
        return f"\n  {title}\n" + "-" * (label_w + col_w * len(configs))

    def get(r, *keys, default="n/a"):
        v = r.get("qa_results", {})
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, default)
            else:
                return default
        return v

    lines = [
        "",
        "=" * (label_w + col_w * len(configs)),
        "  ABLATION COMPARISON",
        "=" * (label_w + col_w * len(configs)),
        header,
        sep,
    ]

    lines.append(section("Quality — Retrieval"))
    # Recall@5: proportion-based (|retrieved∩gold|/|gold| averaged) — matches HippoRAG 2 paper
    # AR@5: binary all-recall — fraction of queries where ALL gold passages found in top-5
    for metric, label in [("Recall@5", "  Recall@5 (prop)"), ("AR@K", "  AR@5 (all)")]:
        vals = []
        for r in results:
            if metric == "Recall@5":
                # from overall_retrieval_result (RetrievalRecall, standard metric)
                v = r.get("retrieval_result", {}).get("Recall@5")
                vals.append(f"{v*100:.1f}%" if isinstance(v, (int, float)) else "n/a")
            else:
                rm = r.get("retrieval_metrics", {})
                v = rm.get(metric)
                vals.append(f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a")
        lines.append(row(label, vals))

    # Supplementary: full retrieval pool (K=200)
    for metric, label in [("AR@K", "  AR@full (all)")]:
        vals = []
        for r in results:
            rm = r.get("retrieval_metrics", {}).get("full_k", {})
            v = rm.get(metric)
            vals.append(f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a")
        lines.append(row(label, vals))

    ctx_label = "  IRCoT Recall@N"
    vals = []
    for r in results:
        rm = r.get("retrieval_metrics", {}).get("ircot_context", {})
        v = rm.get("R@N")
        vals.append(f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a")
    lines.append(row(ctx_label, vals))

    lines.append(section("Quality — QA"))
    for metric_key, label in [("exact_match", "  EM"), ("f1", "  F1")]:
        vals = [str(r.get("qa_results", {}).get(metric_key, "n/a")) for r in results]
        lines.append(row(label, vals))
    for metric_key, label in [("context_coverage_pct", "  Context coverage"), ("reasoning_failure_rate_pct", "  Reasoning failure")]:
        vals = [_pct(r.get("qa_step_metrics", {}).get(metric_key)) for r in results]
        lines.append(row(label, vals))

    lines.append(section("LLM Budget"))
    for phase_key, label in [
        ("reformulation", "  Reform. calls"),
        ("ircot", "  IRCoT calls"),
        ("qa", "  QA calls"),
    ]:
        vals = [str(r.get("pipeline_snapshot", {}).get("llm_budget", {}).get(phase_key, {}).get("calls", 0)) for r in results]
        lines.append(row(label, vals))
    vals = [str(r.get("pipeline_snapshot", {}).get("llm_budget", {}).get("total_calls", "n/a")) for r in results]
    lines.append(row("  Total LLM calls", vals))
    vals = [_tok(r.get("pipeline_snapshot", {}).get("llm_budget", {}).get("total_tokens")) for r in results]
    lines.append(row("  Total tokens", vals))
    vals = [_tok(r.get("pipeline_snapshot", {}).get("llm_budget", {}).get("total_prompt_tokens")) for r in results]
    lines.append(row("  Input tokens", vals))
    vals = [_tok(r.get("pipeline_snapshot", {}).get("llm_budget", {}).get("total_completion_tokens")) for r in results]
    lines.append(row("  Output tokens", vals))

    lines.append(section("IRCoT"))
    for label, extractor in [
        ("  Avg steps/query", lambda r: f"{sum(getattr(qs,'ircot_steps',0) for qs in r.get('solutions',[])) / max(len(r.get('solutions',[])),1):.2f}"),
        ("  Terminal rate", lambda r: _pct(100*r.get("pipeline_snapshot",{}).get("enhancements",{}).get("ircot_terminal_count",0)/max(len(r.get("solutions",[])),1))),
    ]:
        vals = [extractor(r) for r in results]
        lines.append(row(label, vals))

    lines.append(section("Enhancement Effectiveness"))
    for label, key in [
        ("  E1 multi-query rate", "multi_query_rate"),
        ("  E2 coverage inj rate", "coverage_audit_rate"),
    ]:
        vals = [_pct(100*((r.get("pipeline_snapshot") or {}).get("enhancements") or {}).get(key) or 0) for r in results]
        lines.append(row(label, vals))
    for label, key in [
        ("  DPR fallback rate", "fallback_rate"),
        ("  Reranker keep rate", "reranker_keep_rate"),
    ]:
        vals = [_pct(100*((r.get("pipeline_snapshot") or {}).get("retrieval") or {}).get(key) or 0) for r in results]
        lines.append(row(label, vals))

    lines.append(section("Timing (seconds)"))
    for label, key in [
        ("  Total retrieval", "total_retrieval_sec"),
        ("  PPR", "ppr_sec"),
        ("  Rerank", "rerank_sec"),
    ]:
        vals = [str(((r.get("pipeline_snapshot") or {}).get("timing") or {}).get(key, "n/a")) for r in results]
        lines.append(row(label, vals))

    lines.append("=" * (label_w + col_w * len(configs)))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="HippoRAG ablation study")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:4000/v1")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding_name", type=str, default="text-embedding-3-small")
    parser.add_argument("--save_dir", type=str, default="outputs/ablation",
                        help="Directory for ablation result JSONs and logs.")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Parent directory of the pre-built index (e.g. 'outputs'). "
                             "The dataset subfolder is appended automatically. "
                             "Defaults to --save_dir/dataset if omitted.")
    parser.add_argument("--configs", type=str, default="A,B,C,D,E",
                        help="Comma-separated subset of ablation configs to run (A,B,C,D,E,F)")
    parser.add_argument("--force_index_from_scratch", type=str, default="false")
    parser.add_argument("--max_qa_steps_override", type=int, default=None,
                        help="Override max_qa_steps for the full config (default: 4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected = {c.strip().upper() for c in args.configs.split(",")}
    ablation_configs = [
        c for c in ABLATION_CONFIGS
        if c["name"][0].upper() in selected       # "F" from "F: DPR + IRCoT"
        or c["short"].upper() in selected         # full short key e.g. "DPR_IRCOT"
    ]

    dataset_name = args.dataset
    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path) as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json"))
    all_queries = [s["question"] for s in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
    except Exception:
        gold_docs = None

    force_scratch = string_to_bool(args.force_index_from_scratch)

    # index_save_dir: where the graph/embeddings live (read from existing index)
    # results go to args.save_dir/dataset_name/cfg["short"]/ablation_result.json
    index_save_dir = (
        os.path.join(args.index_dir, dataset_name)
        if args.index_dir
        else os.path.join(args.save_dir, dataset_name)
    )

    all_results = []

    # Build a single HippoRAG instance and reuse for all graph-based runs
    # (index is shared; only config differences matter for retrieval/QA)
    base_config = BaseConfig(
        save_dir=index_save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=1,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode="online",
    )

    graph_rag = EnhancedHippoRAG(global_config=base_config)
    graph_rag.index(docs)

    for cfg in ablation_configs:
        logging.info(f"\n{'='*60}\nRunning ablation: {cfg['name']}\n{'='*60}")

        graph_rag.reset_metrics()
        graph_rag.global_config.use_enhancements = cfg["use_enhancements"]
        max_steps = cfg["max_qa_steps"]
        if args.max_qa_steps_override and cfg["max_qa_steps"] > 1:
            max_steps = args.max_qa_steps_override
        graph_rag.global_config.max_qa_steps = max_steps

        if cfg["use_dpr"] and max_steps > 1:
            out = graph_rag.rag_qa_dpr_ircot(
                queries=all_queries,
                gold_docs=gold_docs,
                gold_answers=list(gold_answers),
            )
        elif cfg["use_dpr"]:
            out = graph_rag.rag_qa_dpr(
                queries=all_queries,
                gold_docs=gold_docs,
                gold_answers=list(gold_answers),
            )
        else:
            out = graph_rag.rag_qa(
                queries=all_queries,
                gold_docs=gold_docs,
                gold_answers=list(gold_answers),
            )

        if len(out) == 7:
            solutions, responses, metadata, retrieval_result, qa_results, retrieval_metrics, qa_step_metrics = out
        else:
            solutions, responses, metadata = out
            retrieval_result, qa_results, retrieval_metrics, qa_step_metrics = {}, {}, {}, {}

        pipeline_snapshot = graph_rag.get_metrics_snapshot()

        result = {
            "config": cfg,
            "solutions": solutions,
            "qa_results": qa_results,
            "retrieval_result": retrieval_result,
            "retrieval_metrics": retrieval_metrics or {},
            "qa_step_metrics": qa_step_metrics or {},
            "pipeline_snapshot": pipeline_snapshot,
        }
        all_results.append(result)

        # Save per-config results
        out_dir = os.path.join(args.save_dir, dataset_name, cfg["short"])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ablation_result.json"), "w") as f:
            json.dump({
                "config": cfg,
                "qa_results": qa_results,
                "retrieval_metrics": retrieval_metrics,
                "qa_step_metrics": qa_step_metrics,
                "pipeline_snapshot": pipeline_snapshot,
            }, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
        logging.info(f"Saved ablation result for {cfg['short']} to {out_dir}/ablation_result.json")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
