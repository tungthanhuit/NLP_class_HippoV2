import ast
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .HippoRAG import HippoRAG
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .evaluation.retrieval_eval import RetrievalRecall
from .utils.misc_utils import QuerySolution, text_processing, verbalize_fact

logger = logging.getLogger(__name__)


_REFORMULATE_SYSTEM_MSG = (
    "You are a search expert. Given a question, extract its named entities and generate "
    "diverse retrieval queries — one per named entity plus one extra query if the question "
    "contains a relational bridge phrase (e.g. 'director of', 'performer of', 'capital of', "
    "'father of', 'born in', 'composed by').\n"
    "Rules:\n"
    "  - Each query must be a complete, standalone search query (a phrase or sentence), "
    "    NOT a noun phrase or fragment extracted from the question.\n"
    "  - Generate exactly one query per named entity, focusing that query on retrieving "
    "    facts about that specific entity.\n"
    "  - If a relational bridge is present, add one query using only the relational terms "
    "    from the question to retrieve the bridge entity — NEVER introduce names from your "
    "    own knowledge to fill in the unknown side of the bridge.\n"
    "  - Minimum 1 query, maximum 4 queries total.\n"
    "  - NEVER introduce names, facts, or entity names not explicitly present in the original question.\n"
    "Respond ONLY with valid JSON: {\"entities\": [...], \"queries\": [...]}\n\n"
    "Example:\n"
    "Question: What is the place of birth of the performer of song Changed It?\n"
    "{\"entities\": [\"Changed It\"], "
    "\"queries\": [\"Changed It song performer\", \"birthplace of the performer of the song Changed It\"]}\n\n"
    "INCORRECT (do not copy fragments as queries):\n"
    "{\"entities\": [\"Changed It\"], \"queries\": [\"Changed It\", \"performer of song\"]}"
)

_REFORMULATE_USER_TMPL = "Question: {query}"


class EnhancedHippoRAG(HippoRAG):
    """Experimental HippoRAG variant containing the enhancement pipeline.

    The baseline ``HippoRAG`` class keeps the standard implementation. This
    subclass owns:
    - dual fact representation for fact embeddings,
    - entity-coverage query reformulation,
    - post-rerank coverage audit,
    - NER-seeded graph fallback,
    - IRCoT QA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_fallback_count = 0
        self.initial_retrieve_count = 0
        self.hop_fallback_count = 0
        self.hop_retrieve_count = 0
        self.reformulation_llm_calls = 0
        self.reformulation_prompt_tokens = 0
        self.reformulation_completion_tokens = 0
        self.ircot_llm_calls = 0
        self.ircot_prompt_tokens = 0
        self.ircot_completion_tokens = 0
        self.multi_query_count = 0
        self.coverage_audit_injections = 0
        self.ircot_terminal_count = 0
        self.total_hop_passages_added = 0
        self.total_ircot_hops_done = 0

    def insert_facts(self, facts: List[Tuple]) -> None:
        fact_contents = [str(fact) for fact in facts]
        if self.global_config.verbalize_facts:
            fact_embedding_texts = [verbalize_fact(fact) for fact in facts]
            logger.debug(f"  [insert_facts] verbalize_facts=True — {len(facts)} facts verbalized as sentences")
        else:
            fact_embedding_texts = fact_contents
            logger.debug(f"  [insert_facts] verbalize_facts=False — embedding raw tuple strings (ablation baseline)")
        self.fact_embedding_store.insert_strings_with_embedding_texts(
            fact_contents, fact_embedding_texts
        )

    def prepare_retrieval_objects(self):
        self._refresh_fact_retrieval_text_embeddings()
        return super().prepare_retrieval_objects()

    def _refresh_fact_retrieval_text_embeddings(self) -> None:
        fact_rows = self.fact_embedding_store.get_all_id_to_rows()
        stale_contents: List[str] = []
        stale_embedding_texts: List[str] = []
        unparseable = 0

        for row in fact_rows.values():
            content = row["content"]
            try:
                fact = ast.literal_eval(content)
            except (SyntaxError, ValueError):
                unparseable += 1
                logger.warning(
                    f"  [refresh] unparseable fact content, skipping re-embed: {content!r}"
                )
                continue

            if not isinstance(fact, (tuple, list)) or len(fact) != 3:
                continue

            embedding_text = verbalize_fact(fact) if self.global_config.verbalize_facts else content
            if row.get("embedding_text", content) != embedding_text:
                stale_contents.append(content)
                stale_embedding_texts.append(embedding_text)

        if unparseable:
            logger.warning(
                f"  [refresh] {unparseable} fact(s) could not be parsed — "
                "their embeddings remain as-is (mixed embedding space risk)"
            )

        if not stale_contents:
            return

        mode = "verbalized sentences" if self.global_config.verbalize_facts else "raw tuple strings (ablation)"
        logger.info(
            f"Refreshing {len(stale_contents)} fact embeddings as {mode}."
        )
        self.fact_embedding_store.insert_strings_with_embedding_texts(
            stale_contents, stale_embedding_texts
        )

    def get_metrics_snapshot(self) -> Dict:
        snap = super().get_metrics_snapshot()
        n = self.total_retrieve_queries or 1
        n_init = self.initial_retrieve_count or 1
        n_hop = self.hop_retrieve_count or 1
        avg_hop_passages = (
            self.total_hop_passages_added / self.total_ircot_hops_done
            if self.total_ircot_hops_done > 0
            else None
        )
        snap["retrieval"].update(
            {
                "initial_fallback_count": self.initial_fallback_count,
                "initial_retrieve_count": self.initial_retrieve_count,
                "initial_fallback_rate": round(
                    self.initial_fallback_count / n_init, 4
                ),
                "hop_fallback_count": self.hop_fallback_count,
                "hop_retrieve_count": self.hop_retrieve_count,
                "hop_fallback_rate": round(self.hop_fallback_count / n_hop, 4),
            }
        )
        snap["enhancements"] = {
            "multi_query_count": self.multi_query_count,
            "multi_query_rate": round(self.multi_query_count / n, 4),
            "coverage_audit_injections": self.coverage_audit_injections,
            "coverage_audit_rate": round(self.coverage_audit_injections / n, 4),
            "ircot_terminal_count": self.ircot_terminal_count,
            "total_ircot_hops_done": self.total_ircot_hops_done,
            "avg_hop_passages": round(avg_hop_passages, 2)
            if avg_hop_passages is not None
            else None,
        }
        snap["llm_budget"]["reformulation"] = {
            "calls": self.reformulation_llm_calls,
            "prompt_tokens": self.reformulation_prompt_tokens,
            "completion_tokens": self.reformulation_completion_tokens,
        }
        snap["llm_budget"]["ircot"] = {
            "calls": self.ircot_llm_calls,
            "prompt_tokens": self.ircot_prompt_tokens,
            "completion_tokens": self.ircot_completion_tokens,
        }
        snap["llm_budget"]["total_calls"] += (
            self.reformulation_llm_calls + self.ircot_llm_calls
        )
        snap["llm_budget"]["total_prompt_tokens"] += (
            self.reformulation_prompt_tokens + self.ircot_prompt_tokens
        )
        snap["llm_budget"]["total_completion_tokens"] += (
            self.reformulation_completion_tokens + self.ircot_completion_tokens
        )
        snap["llm_budget"]["total_tokens"] = (
            snap["llm_budget"]["total_prompt_tokens"]
            + snap["llm_budget"]["total_completion_tokens"]
        )
        return snap

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.initial_fallback_count = 0
        self.initial_retrieve_count = 0
        self.hop_fallback_count = 0
        self.hop_retrieve_count = 0
        self.reformulation_llm_calls = 0
        self.reformulation_prompt_tokens = 0
        self.reformulation_completion_tokens = 0
        self.ircot_llm_calls = 0
        self.ircot_prompt_tokens = 0
        self.ircot_completion_tokens = 0
        self.multi_query_count = 0
        self.coverage_audit_injections = 0
        self.ircot_terminal_count = 0
        self.total_hop_passages_added = 0
        self.total_ircot_hops_done = 0

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
        skip_enhancements: bool = False,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(
                global_config=self.global_config
            )

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []
        logger.info(f"Starting enhanced retrieval for {len(queries)} queries")

        for q_idx, query in tqdm(
            enumerate(queries), desc="Enhanced Retrieving", total=len(queries)
        ):
            query_start = time.time()
            logger.info(f"[Retrieve {q_idx + 1}/{len(queries)}] Query: {query[:120]!r}")

            rerank_start = time.time()
            query_entities: List[str] = []
            self.total_retrieve_queries += 1
            if skip_enhancements:
                self.hop_retrieve_count += 1
            else:
                self.initial_retrieve_count += 1

            # Enhanced pipeline only when: (a) not a hop retrieval AND (b) enhancements enabled in config
            use_enhanced_pipeline = not skip_enhancements and self.global_config.use_enhancements

            if use_enhanced_pipeline:
                logger.info("  [Enh1] Reformulating query (multi-query top-K union)")
                sub_queries, query_entities = self.reformulate_query(query)
                if len(sub_queries) > 1:
                    self.multi_query_count += 1
                    facts_per_sub = self.global_config.e1_facts_per_sub_query
                    union_map = {}
                    all_sub_scores = []
                    for sq in sub_queries:
                        sq_scores = self.get_fact_scores(sq)
                        all_sub_scores.append(sq_scores)
                        if len(sq_scores) > 0:
                            n = min(facts_per_sub, len(sq_scores))
                            for idx in np.argsort(sq_scores)[-n:][::-1].tolist():
                                union_map[idx] = max(
                                    union_map.get(idx, 0.0), float(sq_scores[idx])
                                )
                    query_fact_scores = (
                        np.max(np.stack(all_sub_scores, axis=0), axis=0)
                        if all_sub_scores
                        else np.array([])
                    )
                    logger.info(
                        f"  [Enh1] {len(sub_queries)} sub-queries -> "
                        f"{len(union_map)} union candidate facts"
                    )
                    top_k_fact_indices, top_k_facts, rerank_log = (
                        self._rerank_facts_from_indices(query, list(union_map.keys()))
                    )
                else:
                    query_fact_scores = self.get_fact_scores(
                        sub_queries[0] if sub_queries else query
                    )
                    top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                        query, query_fact_scores
                    )
            else:
                logger.debug(
                    f"  [Retrieve] base pipeline "
                    f"({'hop' if skip_enhancements else 'use_enhancements=False'})"
                )
                query_fact_scores = self.get_fact_scores(query)
                top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                    query, query_fact_scores
                )

            self.total_facts_before_rerank += len(
                rerank_log.get("facts_before_rerank", [])
            )
            self.total_facts_after_rerank += len(top_k_facts)

            if query_entities and rerank_log.get("facts_before_rerank"):
                facts_before_audit = len(top_k_facts)
                top_k_facts, top_k_fact_indices = self.coverage_audit(
                    kept_facts=top_k_facts,
                    kept_indices=top_k_fact_indices,
                    all_candidates=rerank_log["facts_before_rerank"],
                    all_candidate_indices=rerank_log["facts_before_rerank_indices"],
                    query_entities=query_entities,
                    required_relations=None,
                )
                if len(top_k_facts) > facts_before_audit:
                    self.coverage_audit_injections += 1

            rerank_end = time.time()
            logger.info(
                f"  Recognition memory: {len(rerank_log['facts_before_rerank'])} -> "
                f"{len(top_k_facts)} facts | {rerank_end - rerank_start:.2f}s"
            )
            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                self.fallback_count += 1
                if skip_enhancements:
                    self.hop_fallback_count += 1
                else:
                    self.initial_fallback_count += 1
                logger.info(
                    "  No facts after rerank - attempting NER-seeded graph fallback"
                )
                sorted_doc_ids, sorted_doc_scores = self._ner_seeded_fallback(query)
            else:
                sorted_doc_ids, sorted_doc_scores = (
                    self.graph_search_with_fact_entities(
                        query=query,
                        link_top_k=self.global_config.linking_top_k,
                        query_fact_scores=query_fact_scores,
                        top_k_facts=top_k_facts,
                        top_k_fact_indices=top_k_fact_indices,
                        passage_node_weight=self.global_config.passage_node_weight,
                    )
                )

            if gold_docs is not None:
                self._log_supporting_passage_ranks(q_idx, gold_docs, sorted_doc_ids)

            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])[
                    "content"
                ]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]

            query_elapsed = time.time() - query_start
            logger.info(
                f"  Retrieved {len(top_k_docs)} docs | "
                f"top score={sorted_doc_scores[0]:.4f} | {query_elapsed:.2f}s"
            )
            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

        self.all_retrieval_time += time.time() - retrieve_start_time

        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[
                        retrieval_result.docs
                        for retrieval_result in retrieval_results
                    ],
                    k_list=k_list,
                )
            )
            try:
                retrieval_metrics_k5 = self.compute_retrieval_metrics(
                    retrieval_results=retrieval_results, gold_docs=gold_docs, k=5
                )
                overall_retrieval_result["custom_metrics_k5"] = retrieval_metrics_k5
                metrics_path = os.path.join(
                    self.working_dir, "retrieval_metrics_k5.json"
                )
                with open(metrics_path, "w") as mf:
                    json.dump(
                        {
                            "overall": overall_retrieval_result,
                            "examples": example_retrieval_results,
                        },
                        mf,
                        indent=2,
                        default=lambda o: float(o)
                        if isinstance(o, (np.floating, np.integer))
                        else o,
                    )
                logger.info(f"Wrote retrieval k=5 metrics to {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to compute/export retrieval metrics: {e}")
            return retrieval_results, overall_retrieval_result

        return retrieval_results

    def reformulate_query(self, query: str) -> Tuple[List[str], List[str]]:
        messages = [
            {"role": "system", "content": _REFORMULATE_SYSTEM_MSG},
            {"role": "user", "content": _REFORMULATE_USER_TMPL.format(query=query)},
        ]
        try:
            response, meta, _cache = self.llm_model.infer(messages)
            self.reformulation_llm_calls += 1
            self.reformulation_prompt_tokens += meta.get("prompt_tokens", 0)
            self.reformulation_completion_tokens += meta.get(
                "completion_tokens", 0
            )
            text = response.strip()
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
            result = json.loads(text)
            entities = [
                e
                for e in (result.get("entities") or [])
                if isinstance(e, str) and e.strip()
            ]
            queries = [
                q
                for q in (result.get("queries") or [])
                if isinstance(q, str) and q.strip()
            ]
            logger.debug(
                f"  [Reformulate] {len(queries)} sub-queries={queries} | entities={entities}"
            )
            return queries or [query], entities
        except Exception as e:
            self.reformulation_llm_calls += 1
            logger.warning(f"  reformulate_query failed ({e}) - using original query")
            return [query], []

    def _rerank_facts_from_indices(
        self, query: str, candidate_indices: List[int]
    ) -> Tuple[List[int], List[Tuple], dict]:
        empty_log = {
            "facts_before_rerank": [],
            "facts_after_rerank": [],
            "facts_before_rerank_indices": [],
        }
        if not candidate_indices or len(self.fact_node_keys) == 0:
            return [], [], empty_log
        try:
            real_ids = [self.fact_node_keys[idx] for idx in candidate_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_ids)
            candidate_facts = [
                ast.literal_eval(fact_row_dict[fid]["content"]) for fid in real_ids
            ]
            top_k_indices, top_k_facts, _ = self.rerank_filter(
                query,
                candidate_facts,
                candidate_indices,
                len_after_rerank=self.global_config.linking_top_k,
            )
            return top_k_indices, top_k_facts, {
                "facts_before_rerank": candidate_facts,
                "facts_before_rerank_indices": candidate_indices,
                "facts_after_rerank": top_k_facts,
            }
        except Exception as e:
            logger.error(f"Error in _rerank_facts_from_indices: {e}", exc_info=True)
            return [], [], empty_log

    def coverage_audit(
        self,
        kept_facts: List[Tuple],
        kept_indices: List[int],
        all_candidates: List[Tuple],
        all_candidate_indices: List[int],
        query_entities: List[str],
        required_relations: List[str] = None,
    ) -> Tuple[List[Tuple], List[int]]:
        def _entity_in_fact(entity: str, fact: Tuple) -> bool:
            el = entity.lower()
            return any(el in str(part).lower() for part in fact)

        result_facts = list(kept_facts)
        result_indices = list(kept_indices)
        kept_idx_set = set(kept_indices)
        covered_tokens = {
            str(part).lower() for fact in kept_facts for part in fact
        }

        for entity in query_entities:
            entity_lower = entity.lower()
            if any(entity_lower in tok for tok in covered_tokens):
                continue
            for cand_fact, cand_idx in zip(all_candidates, all_candidate_indices):
                if cand_idx in kept_idx_set:
                    continue
                if _entity_in_fact(entity, cand_fact):
                    result_facts.append(cand_fact)
                    result_indices.append(cand_idx)
                    kept_idx_set.add(cand_idx)
                    for part in cand_fact:
                        covered_tokens.add(str(part).lower())
                    logger.info(
                        f"  Coverage audit injected for entity '{entity}': {cand_fact}"
                    )
                    break

        if required_relations:
            covered_predicates = {str(f[1]).lower() for f in result_facts}
            for relation in required_relations:
                rel_lower = relation.lower()
                if any(rel_lower in pred for pred in covered_predicates):
                    continue
                for cand_fact, cand_idx in zip(all_candidates, all_candidate_indices):
                    if cand_idx in kept_idx_set:
                        continue
                    if rel_lower in str(cand_fact[1]).lower():
                        result_facts.append(cand_fact)
                        result_indices.append(cand_idx)
                        kept_idx_set.add(cand_idx)
                        covered_predicates.add(str(cand_fact[1]).lower())
                        logger.info(
                            f"  Coverage audit injected for relation '{relation}': {cand_fact}"
                        )
                        break

        return result_facts, result_indices

    def entity_partitioned_selection(
        self,
        candidates: List[Tuple],
        candidate_indices: List[int],
        query_entities: List[str],
        quota_per_entity: int = 1,
        max_total: int = 5,
    ) -> Tuple[List[Tuple], List[int]]:
        entity_buckets: Dict[str, List[Tuple[Tuple, int]]] = {
            e: [] for e in query_entities
        }
        overflow: List[Tuple[Tuple, int]] = []

        for fact, idx in zip(candidates, candidate_indices):
            assigned = False
            for entity in query_entities:
                if entity.lower() in str(fact).lower():
                    entity_buckets[entity].append((fact, idx))
                    assigned = True
                    break
            if not assigned:
                overflow.append((fact, idx))

        selected_facts, selected_indices = [], []
        for bucket in entity_buckets.values():
            for fact, idx in bucket[:quota_per_entity]:
                selected_facts.append(fact)
                selected_indices.append(idx)

        remaining = max_total - len(selected_facts)
        for fact, idx in overflow[: max(remaining, 0)]:
            selected_facts.append(fact)
            selected_indices.append(idx)

        return selected_facts[:max_total], selected_indices[:max_total]

    def _ner_seeded_fallback(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        dpr_doc_ids, dpr_doc_scores = self.dense_passage_retrieval(query)

        top_passage_key = self.passage_node_keys[dpr_doc_ids[0]]
        top_passage_text = self.chunk_embedding_store.get_row(top_passage_key)[
            "content"
        ]
        ner_result = self.openie.ner(
            chunk_key=top_passage_key, passage=top_passage_text
        )
        processed = [text_processing(e) for e in ner_result.unique_entities if e.strip()]
        logger.debug(f"  [NER fallback] top_doc={top_passage_key!r} | entities={processed}")
        if not processed:
            logger.debug("  [NER fallback] no entities found - using flat DPR")
            return dpr_doc_ids, dpr_doc_scores

        synthetic_facts = [(e, "mentioned in", e) for e in processed]
        logger.debug(f"  [NER fallback] {len(synthetic_facts)} synthetic facts -> graph search")
        synthetic_scores = np.ones(len(synthetic_facts))
        synthetic_indices = list(range(len(synthetic_facts)))

        try:
            return self.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.global_config.linking_top_k,
                query_fact_scores=synthetic_scores,
                top_k_facts=synthetic_facts,
                top_k_fact_indices=synthetic_indices,
                passage_node_weight=self.global_config.passage_node_weight,
            )
        except Exception as e:
            logger.info(
                f"  NER-seeded fallback failed ({e}) - using flat DPR"
            )
            return dpr_doc_ids, dpr_doc_scores

    def rag_qa(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ):
        if self.global_config.max_qa_steps <= 1:
            return super().rag_qa(
                queries=queries, gold_docs=gold_docs, gold_answers=gold_answers
            )

        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        overall_retrieval_result = None
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve(queries=queries)

        logger.info(
            f"[IRCoT] enabled - max_qa_steps={self.global_config.max_qa_steps}"
        )
        queries_solutions, all_response_message, all_metadata = self.qa_with_ircot(
            queries, gold_docs=gold_docs
        )

        if gold_answers is None:
            return queries_solutions, all_response_message, all_metadata

        overall_qa_em_result, example_qa_em_results = (
            qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max,
            )
        )
        overall_qa_f1_result, example_qa_f1_results = (
            qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max,
            )
        )
        overall_qa_em_result.update(overall_qa_f1_result)
        key_norm = {"ExactMatch": "exact_match", "F1": "f1"}
        overall_qa_results = {
            key_norm.get(k, k): round(float(v), 4)
            for k, v in overall_qa_em_result.items()
        }
        logger.info(f"Evaluation results for QA: {overall_qa_results}")

        for idx, q in enumerate(queries_solutions):
            q.gold_answers = list(gold_answers[idx])
            if gold_docs is not None:
                q.gold_docs = gold_docs[idx]

        retrieval_metrics = None
        if gold_docs is not None:
            retrieval_metrics = self.compute_retrieval_metrics(
                retrieval_results=queries, gold_docs=gold_docs, k=5
            )
            logger.info(f"Retrieval metrics @K=5: {retrieval_metrics}")
            if any(getattr(qs, "ircot_context", None) for qs in queries_solutions):
                retrieval_metrics["ircot_context"] = self.compute_retrieval_metrics(
                    retrieval_results=queries_solutions,
                    gold_docs=gold_docs,
                    use_ircot_context=True,
                )
                logger.info(f"Retrieval metrics @Loop (AR@Loop): {retrieval_metrics['ircot_context']}")

        predictions = [qa_result.answer for qa_result in queries_solutions]
        qa_step_metrics = self.compute_qa_step_metrics(
            retrieval_results=queries,
            predictions=predictions,
            gold_answers=gold_answers,
            gold_docs=gold_docs,
        )
        logger.info(f"QA step metrics: {qa_step_metrics}")

        qa_metrics_path = os.path.join(self.working_dir, "qa_metrics.json")
        with open(qa_metrics_path, "w") as qf:
            json.dump(
                {
                    "overall_qa": overall_qa_results,
                    "per_example_em": example_qa_em_results,
                    "per_example_f1": example_qa_f1_results,
                },
                qf,
                indent=2,
                default=lambda o: float(o)
                if isinstance(o, (np.floating, np.integer))
                else o,
            )

        snapshot_path = os.path.join(
            self.working_dir, "pipeline_metrics_snapshot.json"
        )
        with open(snapshot_path, "w") as sf:
            json.dump(
                self.get_metrics_snapshot(),
                sf,
                indent=2,
                default=lambda o: float(o)
                if isinstance(o, (np.floating, np.integer))
                else o,
            )

        return (
            queries_solutions,
            all_response_message,
            all_metadata,
            overall_retrieval_result,
            overall_qa_results,
            retrieval_metrics,
            qa_step_metrics,
        )

    def rag_qa_dpr_ircot(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ) -> Tuple:
        """DPR retrieval + IRCoT multi-hop reasoning (ablation config F).

        Uses dense passage retrieval for the initial retrieval pass and for
        every IRCoT hop, keeping the knowledge graph out of the loop entirely.
        Returns a 7-tuple matching the shape of rag_qa() so main_ablation.py
        can handle it uniformly.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        overall_retrieval_result = None
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve_dpr(queries=queries)

        def _dpr_hop(query: str, k: int) -> List[str]:
            hop_qs = self.retrieve_dpr([query])
            return list(hop_qs[0].docs[:k])

        logger.info(
            f"[DPR+IRCoT] enabled - max_qa_steps={self.global_config.max_qa_steps}"
        )
        queries_solutions, all_response_message, all_metadata = self.qa_with_ircot(
            queries, gold_docs=gold_docs, hop_retriever=_dpr_hop
        )

        if gold_answers is None:
            return queries_solutions, all_response_message, all_metadata

        overall_qa_em_result, example_qa_em_results = (
            qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[qs.answer for qs in queries_solutions],
                aggregation_fn=np.max,
            )
        )
        overall_qa_f1_result, example_qa_f1_results = (
            qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[qs.answer for qs in queries_solutions],
                aggregation_fn=np.max,
            )
        )
        overall_qa_em_result.update(overall_qa_f1_result)
        key_norm = {"ExactMatch": "exact_match", "F1": "f1"}
        overall_qa_results = {
            key_norm.get(k, k): round(float(v), 4)
            for k, v in overall_qa_em_result.items()
        }
        logger.info(f"[DPR+IRCoT] QA results: {overall_qa_results}")

        for idx, q in enumerate(queries_solutions):
            q.gold_answers = list(gold_answers[idx])
            if gold_docs is not None:
                q.gold_docs = gold_docs[idx]

        retrieval_metrics = None
        if gold_docs is not None:
            retrieval_metrics = self.compute_retrieval_metrics(
                retrieval_results=queries, gold_docs=gold_docs, k=5
            )
            logger.info(f"[DPR+IRCoT] Retrieval metrics @K=5: {retrieval_metrics}")
            if any(getattr(qs, "ircot_context", None) for qs in queries_solutions):
                retrieval_metrics["ircot_context"] = self.compute_retrieval_metrics(
                    retrieval_results=queries_solutions,
                    gold_docs=gold_docs,
                    use_ircot_context=True,
                )
                logger.info(
                    f"[DPR+IRCoT] Retrieval metrics @Loop: {retrieval_metrics['ircot_context']}"
                )

        predictions = [qs.answer for qs in queries_solutions]
        qa_step_metrics = self.compute_qa_step_metrics(
            retrieval_results=queries,
            predictions=predictions,
            gold_answers=gold_answers,
            gold_docs=gold_docs,
        )

        return (
            queries_solutions,
            all_response_message,
            all_metadata,
            overall_retrieval_result,
            overall_qa_results,
            retrieval_metrics,
            qa_step_metrics,
        )

    def _ircot_step(
        self,
        template_name: str,
        query: str,
        passages: List[str],
        thoughts: List[str],
    ) -> Tuple[str, List[str], Optional[str], dict]:
        prompt_user = ""
        for passage in passages:
            prompt_user += f"Wikipedia Title: {passage}\n\n"
        prompt_user += "Question: " + query
        if thoughts:
            prompt_user += "\nPrevious reasoning: " + " ".join(thoughts)
        messages = self.prompt_template_manager.render(
            name=template_name, prompt_user=prompt_user
        )
        logger.debug(
            f"  [IRCoT step] passages={len(passages)}, thoughts={len(thoughts)}"
        )
        try:
            response, meta, _cache = self.llm_model.infer(messages)
            text = response.strip()
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
            result = json.loads(text)
            thought = str(result.get("thought", "")).strip()
            answer = str(result.get("answer", "")).strip() or None
            raw_mqs = result.get("missing_queries") or []
            missing_queries = [str(q).strip() for q in raw_mqs if str(q).strip()]
            logger.debug(
                f"  [IRCoT step] -> thought={thought[:80]!r} | "
                f"missing_queries={missing_queries} | answer={answer!r} | "
                f"tokens(p={meta.get('prompt_tokens',0)}, c={meta.get('completion_tokens',0)})"
            )
            if answer:
                return thought, [], answer, meta
            return thought, missing_queries, None, meta
        except Exception as e:
            logger.warning(f"  [IRCoT] step parse failed ({e}) - treating as no-op")
            return "", [], None, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "error",
            }

    def _ircot_final_qa(
        self,
        query: str,
        passages: List[str],
        cot_sentences: List[str],
    ) -> Tuple[str, str, dict]:
        prompt_user = ""
        for passage in passages:
            prompt_user += f"Wikipedia Title: {passage}\n\n"
        thought_prefix = " ".join(cot_sentences)
        if thought_prefix:
            prompt_user += "Question: " + query + "\nThought: " + thought_prefix + " "
        else:
            prompt_user += "Question: " + query + "\nThought: "

        dataset = self.global_config.dataset or "musique"
        qa_template = (
            f"rag_qa_{dataset}"
            if self.prompt_template_manager.is_template_name_valid(f"rag_qa_{dataset}")
            else "rag_qa_musique"
        )
        messages = self.prompt_template_manager.render(
            name=qa_template, prompt_user=prompt_user
        )
        logger.debug(
            f"  [IRCoT final QA] passages={len(passages)}, cot_sentences={len(cot_sentences)}"
        )
        response, metadata, _cache = self.llm_model.infer(messages)
        try:
            answer = response.split("Answer:")[1].strip()
        except Exception:
            answer = response
        logger.debug(
            f"  [IRCoT final QA] -> answer={answer[:80]!r} | "
            f"tokens(p={metadata.get('prompt_tokens',0)}, c={metadata.get('completion_tokens',0)})"
        )
        return answer, response, metadata

    def qa_with_ircot(
        self,
        queries: List[QuerySolution],
        gold_docs: List[List[str]] = None,
        hop_retriever=None,  # callable(query: str, k: int) -> List[str], or None → graph
    ) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        dataset = self.global_config.dataset or "musique"
        ircot_template = f"ircot_{dataset}"
        if not self.prompt_template_manager.is_template_name_valid(ircot_template):
            ircot_template = "ircot_musique"

        queries_solutions: List[QuerySolution] = []
        all_response_message: List[str] = []
        all_metadata: List[Dict] = []

        for q_idx, qs in enumerate(tqdm(queries, desc="IRCoT QA")):
            query = qs.question
            all_passages = list(qs.docs[: self.global_config.qa_top_k])
            cot_sentences: List[str] = []
            terminal_answer: str = None
            q_ircot_prompt_tokens = 0
            q_ircot_completion_tokens = 0
            q_ircot_llm_calls = 0
            q_qa_prompt_tokens = 0
            q_qa_completion_tokens = 0
            q_qa_llm_calls = 0

            logger.info(
                f"[IRCoT query {q_idx + 1}/{len(queries)}] {query!r} | "
                f"initial_passages={len(all_passages)}"
            )

            gold_set = set(gold_docs[q_idx]) if gold_docs is not None and q_idx < len(gold_docs) else None

            for step in range(self.global_config.max_qa_steps - 1):
                thought, missing_queries, answer, step_meta = self._ircot_step(
                    ircot_template, query, all_passages, cot_sentences
                )
                q_ircot_prompt_tokens += step_meta.get("prompt_tokens", 0)
                q_ircot_completion_tokens += step_meta.get("completion_tokens", 0)
                q_ircot_llm_calls += 1

                logger.debug(
                    f"  [Step {step + 1}] thought={thought[:60]!r} | "
                    f"missing_queries={missing_queries} | answer={answer!r} | "
                    f"passages_in_ctx={len(all_passages)}"
                )

                if not thought and not missing_queries and answer is None:
                    logger.debug(f"  [Step {step + 1}] no-op response - breaking loop early")
                    break
                if thought:
                    cot_sentences.append(thought)
                if answer is not None:
                    terminal_answer = answer.strip().rstrip(".")
                    logger.debug(f"  [Step {step + 1}] terminal answer found - stopping loop")
                    break
                if missing_queries:
                    passages_before = len(all_passages)
                    seen = set(all_passages)
                    for mq in missing_queries:
                        if hop_retriever is not None:
                            new_docs = hop_retriever(mq, self.global_config.qa_top_k)
                        else:
                            hop_results = self.retrieve(
                                [mq],
                                num_to_retrieve=self.global_config.qa_top_k,
                                skip_enhancements=True,
                            )
                            new_docs = hop_results[0].docs
                        for p in new_docs:
                            if p not in seen:
                                all_passages.append(p)
                                seen.add(p)
                    added = len(all_passages) - passages_before
                    self.total_hop_passages_added += added
                    self.total_ircot_hops_done += 1
                    logger.debug(
                        f"  [Step {step + 1}] hop retrieval: +{added} new passages "
                        f"({passages_before} -> {len(all_passages)} total)"
                    )
                    if gold_set is not None:
                        covered = gold_set.intersection(set(all_passages))
                        logger.debug(
                            f"  [Step {step + 1}] gold coverage after hop: "
                            f"{len(covered)}/{len(gold_set)} "
                            f"({'AR=1' if gold_set.issubset(set(all_passages)) else 'AR=0'})"
                        )

            qs.ircot_context = list(all_passages)

            if gold_set is not None:
                covered_final = gold_set.intersection(set(qs.ircot_context))
                ar_loop = gold_set.issubset(set(qs.ircot_context))
                logger.debug(
                    f"  [Loop done] ircot_context={len(qs.ircot_context)} passages | "
                    f"gold coverage={len(covered_final)}/{len(gold_set)} | AR@Loop={ar_loop}"
                )

            if terminal_answer is not None:
                self.ircot_terminal_count += 1
                qs.answer = terminal_answer
                raw_msg = terminal_answer
                metadata = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "ircot_terminal",
                }
            else:
                answer, raw_msg, metadata = self._ircot_final_qa(
                    query=query,
                    passages=all_passages,
                    cot_sentences=cot_sentences,
                )
                q_qa_prompt_tokens += metadata.get("prompt_tokens", 0)
                q_qa_completion_tokens += metadata.get("completion_tokens", 0)
                q_qa_llm_calls += 1
                qs.answer = answer

            qs.ircot_steps = len(cot_sentences)
            self.ircot_llm_calls += q_ircot_llm_calls
            self.ircot_prompt_tokens += q_ircot_prompt_tokens
            self.ircot_completion_tokens += q_ircot_completion_tokens
            self.qa_llm_calls += q_qa_llm_calls
            self.qa_prompt_tokens += q_qa_prompt_tokens
            self.qa_completion_tokens += q_qa_completion_tokens

            queries_solutions.append(qs)
            all_response_message.append(raw_msg)
            all_metadata.append(metadata)

        return queries_solutions, all_response_message, all_metadata
