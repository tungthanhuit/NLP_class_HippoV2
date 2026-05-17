import json
import os
import logging
from dataclasses import asdict
from typing import List, Set, Dict, Tuple
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import igraph as ig
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query decomposition helpers (Enhancement 1)
# ---------------------------------------------------------------------------

# RAG Fusion reformulation prompts (Enhancement 1)
# ---------------------------------------------------------------------------
# Entity-count-based approach: extract named entities from the query, generate
# one retrieval query per entity plus one bridge query when a relational phrase
# is detected.  The LLM is explicitly forbidden from introducing entity names
# that are not present in the original question, eliminating the second-hop
# information leakage of the old decomposition approach.

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


class HippoRAG:

    def __init__(
        self,
        global_config=None,
        save_dir=None,
        llm_model_name=None,
        llm_base_url=None,
        embedding_model_name=None,
        embedding_base_url=None,
        embedding_trust_remote_code=None,
    ):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (OpenIE): The Open Information Extraction module configured based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if embedding_trust_remote_code is not None:
            self.global_config.embedding_trust_remote_code = embedding_trust_remote_code

        _print_config = ",\n  ".join(
            [f"{k} = {v}" for k, v in asdict(self.global_config).items()]
        )
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")
        print(f"HippoRAG init with config:\n  {_print_config}\n")

        # LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(
            self.global_config.save_dir, f"{llm_label}_{embedding_label}"
        )

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode != "online":
            raise ValueError(
                "Only openie_mode='online' is supported in this build. "
                "Configure an OpenAI-compatible LLM API via llm_base_url/llm_name."
            )
        self.openie = OpenIE(llm_model=self.llm_model)

        self.graph = self.initialize_graph()

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name
        )(
            global_config=self.global_config,
            embedding_model_name=self.global_config.embedding_model_name,
        )
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size,
            "chunk",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "entity_embeddings"),
            self.global_config.embedding_batch_size,
            "entity",
        )
        self.fact_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "fact_embeddings"),
            self.global_config.embedding_batch_size,
            "fact",
        )

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json',
        )

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self.fallback_count = 0
        self.total_retrieve_queries = 0
        # Split fallback tracking: initial retrieval vs IRCoT hop retrievals
        self.initial_fallback_count = 0
        self.initial_retrieve_count = 0
        self.hop_fallback_count = 0
        self.hop_retrieve_count = 0

        # Phase-level LLM budget counters
        self.reformulation_llm_calls = 0
        self.reformulation_prompt_tokens = 0
        self.reformulation_completion_tokens = 0
        self.ircot_llm_calls = 0
        self.ircot_prompt_tokens = 0
        self.ircot_completion_tokens = 0
        self.qa_llm_calls = 0
        self.qa_prompt_tokens = 0
        self.qa_completion_tokens = 0

        # Enhancement effectiveness counters
        self.multi_query_count = 0         # E1: queries where >1 sub-query generated
        self.coverage_audit_injections = 0  # E2: queries where coverage audit injected ≥1 fact
        self.ircot_terminal_count = 0       # E3: queries where IRCoT found answer before max_steps

        # Reranker aggregate stats
        self.total_facts_before_rerank = 0
        self.total_facts_after_rerank = 0

        # IRCoT hop stats
        self.total_hop_passages_added = 0
        self.total_ircot_hops_done = 0

        self.ent_node_to_chunk_ids = None

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(self.working_dir, f"graph.pickle")

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing {len(docs)} documents")

        logger.info(f"Performing OpenIE")

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunk_to_rows.keys()
        )
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(
                new_openie_rows
            )
            self.merge_openie_results(
                all_openie_info,
                new_openie_rows,
                new_ner_results_dict,
                new_triple_results_dict,
            )

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert (
            len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)
        ), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [
            [text_processing(t) for t in triple_results_dict[chunk_id].triples]
            for chunk_id in chunk_ids
        ]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding {len(entity_nodes)} entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding {len(facts)} facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        # Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        # Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [
                self.chunk_embedding_store.text_to_hash_id[chunk]
                for chunk in docs_to_delete
            ]
        )

        # Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc["idx"] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc["extracted_triples"])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        # Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [
            [text_processing(list(triple)) for triple in true_triples_to_delete]
        ]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(
            processed_true_triples_to_delete
        )

        triple_ids_to_delete = set(
            [
                self.fact_embedding_store.text_to_hash_id[str(triple)]
                for triple in processed_true_triples_to_delete
            ]
        )

        # Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [
            self.entity_embedding_store.text_to_hash_id[ent]
            for ent in entities_to_delete
        ]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        # Delete Nodes from Graph
        self.graph.delete_vertices(
            list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete)
        )
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
        skip_enhancements: bool = False,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

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

        logger.info(f"Starting retrieval for {len(queries)} queries")

        for q_idx, query in tqdm(
            enumerate(queries), desc="Retrieving", total=len(queries)
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

            if self.global_config.use_enhancements and not skip_enhancements:
                # --- Enhancement 1: multi-query top-K union + single reranker pass ---
                # For each sub-query take the top-3 facts by embedding similarity,
                # union the candidate pools (dedup by index), then run the reranker
                # once on the full union using the *original* question.
                # This avoids the RRF problem (all entity sub-queries returning the
                # same first-hop passage) while still surfacing entity-specific facts.
                logger.info("  [Enh1] Reformulating query (multi-query top-K union)")
                sub_queries, query_entities = self.reformulate_query(query)
                if len(sub_queries) > 1:
                    self.multi_query_count += 1
                    facts_per_sub = self.global_config.e1_facts_per_sub_query
                    union_map = {}  # fact_index -> max score across sub-queries
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
                    # Point-wise max for PPR node weight seeding
                    query_fact_scores = (
                        np.max(np.stack(all_sub_scores, axis=0), axis=0)
                        if all_sub_scores else np.array([])
                    )
                    logger.info(
                        f"  [Enh1] {len(sub_queries)} sub-queries → "
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
                query_fact_scores = self.get_fact_scores(query)
                top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
                    query, query_fact_scores
                )
            self.total_facts_before_rerank += len(rerank_log.get("facts_before_rerank", []))
            self.total_facts_after_rerank += len(top_k_facts)

            if self.global_config.use_enhancements and query_entities and rerank_log.get("facts_before_rerank"):
                # --- Enhancement 2: coverage audit (post-rerank guard) ---
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
                f"  Recognition memory: {len(rerank_log['facts_before_rerank'])} → "
                f"{len(top_k_facts)} facts | {rerank_end - rerank_start:.2f}s"
            )
            if top_k_facts:
                logger.debug(f"  Top fact after rerank: {top_k_facts[0]}")

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                self.fallback_count += 1
                if skip_enhancements:
                    self.hop_fallback_count += 1
                else:
                    self.initial_fallback_count += 1
                if self.global_config.use_enhancements:
                    logger.info("  No facts after rerank — attempting NER-seeded graph fallback (Fix C)")
                    sorted_doc_ids, sorted_doc_scores = self._ner_seeded_fallback(query)
                else:
                    logger.info("  No facts after rerank — falling back to DPR")
                    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
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
            if top_k_docs:
                logger.debug(f"  Top doc preview: {top_k_docs[0][:120]!r}")

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(
            f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s"
        )

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[
                        retrieval_result.docs for retrieval_result in retrieval_results
                    ],
                    k_list=k_list,
                )
            )
            # compute additional requested metrics at K=5 and export for tracking
            try:
                retrieval_metrics_k5 = self.compute_retrieval_metrics(
                    retrieval_results=retrieval_results, gold_docs=gold_docs, k=5
                )
                # attach custom metrics into overall result under a clear key
                overall_retrieval_result["custom_metrics_k5"] = retrieval_metrics_k5

                # write to working dir for external tracking
                metrics_path = os.path.join(self.working_dir, "retrieval_metrics_k5.json")
                with open(metrics_path, "w") as mf:
                    json.dump({"overall": overall_retrieval_result, "examples": example_retrieval_results}, mf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
                logger.info(f"Wrote retrieval k=5 metrics to {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to compute/export custom retrieval metrics: {e}")

            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ) -> (
        Tuple[List[QuerySolution], List[str], List[Dict]]
        | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
    ):
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve(queries=queries)

        # IRCoT activates on max_qa_steps > 1 regardless of use_enhancements.
        # use_enhancements controls E1 (RAG Fusion on initial retrieval) only.
        # Config D ablation (E2 only) sets use_enhancements=False + max_qa_steps=4.
        use_ircot = self.global_config.max_qa_steps > 1
        if use_ircot:
            logger.info(
                f"[IRCoT] enabled — max_qa_steps={self.global_config.max_qa_steps}"
            )
            queries_solutions, all_response_message, all_metadata = self.qa_with_ircot(
                queries, gold_docs=gold_docs
            )
        else:
            queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = (
                qa_em_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )
            overall_qa_f1_result, example_qa_f1_results = (
                qa_f1_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )

            # round off to 4 decimal places; normalize evaluator keys to snake_case
            # (QAExactMatch returns "ExactMatch", QAF1Score returns "F1")
            overall_qa_em_result.update(overall_qa_f1_result)
            _key_norm = {"ExactMatch": "exact_match", "F1": "f1"}
            overall_qa_results = {
                _key_norm.get(k, k): round(float(v), 4)
                for k, v in overall_qa_em_result.items()
            }
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]
            # Compute additional metrics (retrieval-side and QA-step)
            retrieval_metrics = None
            qa_step_metrics = None

            try:
                if gold_docs is not None:
                    # Primary metric at K=5 (what the LLM actually reads)
                    retrieval_metrics = self.compute_retrieval_metrics(
                        retrieval_results=queries, gold_docs=gold_docs, k=5
                    )

                predictions = [qa_result.answer for qa_result in queries_solutions]
                qa_step_metrics = self.compute_qa_step_metrics(
                    retrieval_results=queries,
                    predictions=predictions,
                    gold_answers=gold_answers,
                    gold_docs=gold_docs,
                )
                # Also compute at full retrieval_top_k as supplementary
                try:
                    retrieval_metrics_full = None
                    if gold_docs is not None:
                        retrieval_metrics_full = self.compute_retrieval_metrics(
                            retrieval_results=queries, gold_docs=gold_docs,
                            k=self.global_config.retrieval_top_k
                        )
                        if retrieval_metrics is None:
                            retrieval_metrics = {}
                        retrieval_metrics["full_k"] = retrieval_metrics_full

                        # IRCoT context recall: recall of the full accumulated context
                        # (initial top-K + all hop passages) that the LLM actually reads from.
                        # Meaningful only when IRCoT ran; skipped otherwise.
                        if use_ircot and any(
                            getattr(qs, "ircot_context", None) for qs in queries_solutions
                        ):
                            ircot_context_metrics = self.compute_retrieval_metrics(
                                retrieval_results=queries_solutions,
                                gold_docs=gold_docs,
                                use_ircot_context=True,
                            )
                            retrieval_metrics["ircot_context"] = ircot_context_metrics
                            logger.info(
                                f"IRCoT context recall: R@N={ircot_context_metrics['R@N']:.2f}% "
                                f"AR@N={ircot_context_metrics['AR@N']:.2f}% "
                                f"(avg context N={ircot_context_metrics['N']})"
                            )

                    # write QA overall and per-example EM/F1 to file
                    qa_metrics_path = os.path.join(self.working_dir, "qa_metrics.json")
                    qa_export = {
                        "overall_qa": overall_qa_results,
                        "per_example_em": example_qa_em_results,
                        "per_example_f1": example_qa_f1_results,
                    }
                    with open(qa_metrics_path, "w") as qf:
                        json.dump(qa_export, qf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
                    logger.info(f"Wrote QA metrics to {qa_metrics_path}")
                except Exception as e:
                    logger.warning(f"Failed to compute/export K=5 retrieval or QA metrics: {e}")
            except Exception as e:
                logger.warning(f"Error computing supplemental metrics: {e}")

            # --- Pipeline summary ---
            try:
                n_q = len(queries_solutions)
                snap = self.get_metrics_snapshot()

                # Retrieval quality
                # overall_retrieval_result comes from RetrievalRecall (proportion-based,
                # the standard multi-hop QA metric, matches original HippoRAG 2 paper).
                # retrieval_metrics comes from compute_retrieval_metrics (binary any/all recall).
                recall_k5_str = "n/a"
                recall_full_str = "n/a"
                if overall_retrieval_result:
                    prop5 = overall_retrieval_result.get("Recall@5", overall_retrieval_result.get("Recall@K"))
                    ar5 = retrieval_metrics.get("AR@K", 0) if retrieval_metrics else 0
                    prop5_str = f"{prop5*100:.1f}%" if prop5 is not None else "n/a"
                    recall_k5_str = f"Recall@5={prop5_str}  AR@5={ar5:.1f}%"
                    k_full = self.global_config.retrieval_top_k
                    prop_full = overall_retrieval_result.get(f"Recall@{k_full}")
                    ar_full = (retrieval_metrics.get("full_k", {}) or {}).get("AR@K", 0) if retrieval_metrics else 0
                    if prop_full is not None:
                        recall_full_str = f"Recall@{k_full}={prop_full*100:.1f}%  AR@{k_full}={ar_full:.1f}%"
                elif retrieval_metrics:
                    r5 = retrieval_metrics.get("R@K", 0)
                    ar5 = retrieval_metrics.get("AR@K", 0)
                    recall_k5_str = f"R@5(any)={r5:.1f}%  AR@5={ar5:.1f}%"
                    kfull = retrieval_metrics.get("full_k", {})
                    if kfull:
                        k_val = kfull.get("K", self.global_config.retrieval_top_k)
                        recall_full_str = f"R@{k_val}(any)={kfull.get('R@K', 0):.1f}%  AR@{k_val}={kfull.get('AR@K', 0):.1f}%"

                ircot_ctx_str = "n/a"
                if retrieval_metrics and "ircot_context" in retrieval_metrics:
                    ctx = retrieval_metrics["ircot_context"]
                    ircot_ctx_str = (
                        f"R@N={ctx.get('R@N', 0):.1f}%  AR@N={ctx.get('AR@N', 0):.1f}%"
                        f"  (avg N={ctx.get('N', '?')})"
                    )

                em_str = overall_qa_results.get("exact_match", overall_qa_results.get("EM", "n/a"))
                f1_str = overall_qa_results.get("f1", overall_qa_results.get("F1", "n/a"))
                cov_str = "n/a"
                rfail_str = "n/a"
                if qa_step_metrics:
                    cov_str = f"{qa_step_metrics.get('context_coverage_pct', 0):.1f}%"
                    rfail_str = f"{qa_step_metrics.get('reasoning_failure_rate_pct', 0):.1f}%"

                # LLM budget
                lb = snap["llm_budget"]
                rf = lb["reformulation"]
                ir = lb["ircot"]
                qa_b = lb["qa"]

                # IRCoT steps
                ircot_step_counts = [getattr(qs, "ircot_steps", 0) for qs in queries_solutions]
                if use_ircot and ircot_step_counts:
                    step_avg = sum(ircot_step_counts) / len(ircot_step_counts)
                    step_min = min(ircot_step_counts)
                    step_max = max(ircot_step_counts)
                    ircot_step_str = f"avg={step_avg:.2f}  min={step_min}  max={step_max}"
                    terminal_rate = 100.0 * snap["enhancements"]["ircot_terminal_count"] / n_q if n_q > 0 else 0
                    ircot_terminal_str = f"{snap['enhancements']['ircot_terminal_count']}/{n_q} ({terminal_rate:.0f}%)"
                    avg_hop = snap["enhancements"]["avg_hop_passages"]
                    ircot_hop_str = f"{snap['enhancements']['total_ircot_hops_done']} hops  avg {avg_hop or 0:.1f} passages/hop"
                else:
                    ircot_step_str = "disabled"
                    ircot_terminal_str = "n/a"
                    ircot_hop_str = "n/a"

                # Reranker
                rr = snap["retrieval"]
                reranker_str = (
                    f"{rr['avg_facts_before_rerank']:.1f} → {rr['avg_facts_after_rerank']:.1f} facts/query"
                    f"  keep={100*(rr['reranker_keep_rate'] or 0):.1f}%"
                )
                init_fb = rr["initial_fallback_count"]
                init_tot = rr["initial_retrieve_count"]
                init_pct = 100 * init_fb / init_tot if init_tot > 0 else 0
                hop_fb = rr["hop_fallback_count"]
                hop_tot = rr["hop_retrieve_count"]
                hop_pct = 100 * hop_fb / hop_tot if hop_tot > 0 else 0
                fb_str = (
                    f"initial {init_fb}/{init_tot} ({init_pct:.1f}%)"
                    f"  hop {hop_fb}/{hop_tot} ({hop_pct:.1f}%)"
                )

                # E1/E2
                mq_pct = 100 * snap["enhancements"]["multi_query_rate"]
                ca_pct = 100 * snap["enhancements"]["coverage_audit_rate"]

                logger.info(
                    "\n" + "=" * 62 + "\n"
                    "  PIPELINE SUMMARY\n"
                    + "=" * 62 + "\n"
                    f"  Queries                : {n_q}\n"
                    "\n"
                    "  --- Retrieval (K=5 passages read by LLM) ---\n"
                    f"  Recall@5 / AR@5        : {recall_k5_str}\n"
                    f"  Recall@{self.global_config.retrieval_top_k} / AR@{self.global_config.retrieval_top_k}     : {recall_full_str}\n"
                    f"  IRCoT context recall   : {ircot_ctx_str}\n"
                    f"  DPR/NER fallback       : {fb_str}\n"
                    f"  Reranker (avg facts)   : {reranker_str}\n"
                    "\n"
                    "  --- QA ---\n"
                    f"  EM                     : {em_str}\n"
                    f"  F1                     : {f1_str}\n"
                    f"  Context coverage       : {cov_str}\n"
                    f"  Reasoning failure      : {rfail_str}\n"
                    "\n"
                    "  --- LLM Budget ---\n"
                    f"  Reformulation calls    : {rf['calls']}  "
                    f"(in={rf['prompt_tokens']} out={rf['completion_tokens']} tok)\n"
                    f"  IRCoT step calls       : {ir['calls']}  "
                    f"(in={ir['prompt_tokens']} out={ir['completion_tokens']} tok)\n"
                    f"  Final QA calls         : {qa_b['calls']}  "
                    f"(in={qa_b['prompt_tokens']} out={qa_b['completion_tokens']} tok)\n"
                    f"  Total LLM calls        : {lb['total_calls']}\n"
                    f"  Total tokens           : {lb['total_tokens']}  "
                    f"(in={lb['total_prompt_tokens']} out={lb['total_completion_tokens']})\n"
                    "\n"
                    "  --- IRCoT ---\n"
                    f"  Steps/query            : {ircot_step_str}\n"
                    f"  Terminal (early exit)  : {ircot_terminal_str}\n"
                    f"  Hop retrieval          : {ircot_hop_str}\n"
                    "\n"
                    "  --- Enhancement Effectiveness ---\n"
                    f"  E1 multi-query rate    : {mq_pct:.1f}%  ({snap['enhancements']['multi_query_count']}/{n_q} queries)\n"
                    f"  E2 coverage inj rate   : {ca_pct:.1f}%  ({snap['enhancements']['coverage_audit_injections']}/{n_q} queries)\n"
                    "\n"
                    "  --- Timing ---\n"
                    f"  Retrieval              : {snap['timing']['total_retrieval_sec']}s\n"
                    f"  PPR                    : {snap['timing']['ppr_sec']}s\n"
                    f"  Rerank                 : {snap['timing']['rerank_sec']}s\n"
                    + "=" * 62
                )

                # Persist snapshot to disk for ablation comparison
                snapshot_path = os.path.join(self.working_dir, "pipeline_metrics_snapshot.json")
                with open(snapshot_path, "w") as sf:
                    json.dump(snap, sf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
                logger.info(f"Wrote pipeline metrics snapshot to {snapshot_path}")

            except Exception as e:
                logger.warning(f"Failed to render pipeline summary: {e}")

            return (
                queries_solutions,
                all_response_message,
                all_metadata,
                overall_retrieval_result,
                overall_qa_results,
                retrieval_metrics,
                qa_step_metrics,
            )
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

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

        logger.info(f"Starting DPR retrieval for {len(queries)} queries")

        for q_idx, query in tqdm(
            enumerate(queries), desc="Retrieving", total=len(queries)
        ):
            query_start = time.time()
            logger.info(f"[DPR {q_idx + 1}/{len(queries)}] Query: {query[:120]!r}")

            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

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
            if top_k_docs:
                logger.debug(f"  Top doc preview: {top_k_docs[0][:120]!r}")

            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[:num_to_retrieve],
                )
            )

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[
                        retrieval_result.docs for retrieval_result in retrieval_results
                    ],
                    k_list=k_list,
                )
            )
            # compute additional requested metrics at K=5 and export for tracking
            try:
                retrieval_metrics_k5 = self.compute_retrieval_metrics(
                    retrieval_results=retrieval_results, gold_docs=gold_docs, k=5
                )
                overall_retrieval_result["custom_metrics_k5"] = retrieval_metrics_k5

                metrics_path = os.path.join(self.working_dir, "retrieval_metrics_k5.json")
                with open(metrics_path, "w") as mf:
                    json.dump({"overall": overall_retrieval_result, "examples": example_retrieval_results}, mf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
                logger.info(f"Wrote retrieval k=5 metrics to {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to compute/export custom retrieval metrics: {e}")

            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(
        self,
        queries: List[str | QuerySolution],
        gold_docs: List[List[str]] = None,
        gold_answers: List[List[str]] = None,
    ) -> (
        Tuple[List[QuerySolution], List[str], List[Dict]]
        | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
    ):
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = (
                qa_em_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )
            overall_qa_f1_result, example_qa_f1_results = (
                qa_f1_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers,
                    predicted_answers=[
                        qa_result.answer for qa_result in queries_solutions
                    ],
                    aggregation_fn=np.max,
                )
            )

            # round off to 4 decimal places; normalize evaluator keys to snake_case
            # (QAExactMatch returns "ExactMatch", QAF1Score returns "F1")
            overall_qa_em_result.update(overall_qa_f1_result)
            _key_norm = {"ExactMatch": "exact_match", "F1": "f1"}
            overall_qa_results = {
                _key_norm.get(k, k): round(float(v), 4)
                for k, v in overall_qa_em_result.items()
            }
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            # Compute additional metrics (retrieval-side and QA-step)
            retrieval_metrics = None
            qa_step_metrics = None

            try:
                predictions = [qa_result.answer for qa_result in queries_solutions]
                qa_step_metrics = self.compute_qa_step_metrics(
                    retrieval_results=queries,
                    predictions=predictions,
                    gold_answers=gold_answers,
                    gold_docs=gold_docs,
                )
                # Primary metric at K=5 (top-level); full_k stored as supplementary.
                # Matches the key layout in rag_qa so the ablation table reads R@K/AR@K uniformly.
                try:
                    if gold_docs is not None:
                        retrieval_metrics = self.compute_retrieval_metrics(
                            retrieval_results=queries, gold_docs=gold_docs, k=5
                        )
                        retrieval_metrics["full_k"] = self.compute_retrieval_metrics(
                            retrieval_results=queries, gold_docs=gold_docs,
                            k=self.global_config.retrieval_top_k
                        )

                    qa_metrics_path = os.path.join(self.working_dir, "qa_metrics.json")
                    qa_export = {
                        "overall_qa": overall_qa_results,
                        "per_example_em": example_qa_em_results,
                        "per_example_f1": example_qa_f1_results,
                    }
                    with open(qa_metrics_path, "w") as qf:
                        json.dump(qa_export, qf, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
                    logger.info(f"Wrote QA metrics to {qa_metrics_path}")
                except Exception as e:
                    logger.warning(f"Failed to compute/export K=5 retrieval or QA metrics: {e}")
            except Exception as e:
                logger.warning(f"Error computing supplemental metrics: {e}")

            return (
                queries_solutions,
                all_response_message,
                all_metadata,
                overall_retrieval_result,
                overall_qa_results,
                retrieval_metrics,
                qa_step_metrics,
            )
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(
        self, queries: List[QuerySolution]
    ) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # Running inference for QA
        all_qa_messages = []

        logger.info(f"Building QA prompts for {len(queries)} queries (top_k={self.global_config.qa_top_k})")

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[: self.global_config.qa_top_k]

            logger.debug(
                f"[QA prompt] Q: {query_solution.question[:100]!r} | "
                f"passages={len(retrieved_passages)}"
            )

            prompt_user = ""
            for passage in retrieved_passages:
                prompt_user += f"Wikipedia Title: {passage}\n\n"
            prompt_user += "Question: " + query_solution.question + "\nThought: "

            logger.debug(f"  Prompt length: {len(prompt_user)} chars (~{len(prompt_user.split())} words)")

            if self.prompt_template_manager.is_template_name_valid(
                name=f"rag_qa_{self.global_config.dataset}"
            ):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead."
                )
                prompt_dataset_name = "musique"
            all_qa_messages.append(
                self.prompt_template_manager.render(
                    name=f"rag_qa_{prompt_dataset_name}", prompt_user=prompt_user
                )
            )

        logger.info(f"Running LLM inference for {len(all_qa_messages)} QA prompts")
        all_qa_results = [
            self.llm_model.infer(qa_messages)
            for qa_messages in tqdm(all_qa_messages, desc="QA Reading")
        ]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(
            all_metadata
        )
        self.qa_llm_calls += len(all_qa_results)
        for _m in all_metadata:
            self.qa_prompt_tokens += _m.get("prompt_tokens", 0)
            self.qa_completion_tokens += _m.get("completion_tokens", 0)

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(
            enumerate(queries), desc="Extraction Answers from LLM Response"
        ):
            response_content = all_response_message[query_solution_idx]
            logger.debug(
                f"[QA {query_solution_idx + 1}/{len(queries)}] "
                f"Response length: {len(response_content)} chars"
            )
            try:
                pred_ans = response_content.split("Answer:")[1].strip()
            except Exception as e:
                logger.warning(
                    f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!"
                )
                pred_ans = response_content

            logger.info(
                f"[QA {query_solution_idx + 1}/{len(queries)}] "
                f"Q: {query_solution.question[:80]!r} → Answer: {pred_ans[:80]!r}"
            )
            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(
                        content=triple[2], prefix=("entity-")
                    )

                    self.node_to_node_stats[(node_key, node_2_key)] = (
                        self.node_to_node_stats.get((node_key, node_2_key), 0.0) + 1
                    )
                    self.node_to_node_stats[(node_2_key, node_key)] = (
                        self.node_to_node_stats.get((node_2_key, node_key), 0.0) + 1
                    )

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(
                        node, set()
                    ).union(set([chunk_key]))

    def add_passage_edges(
        self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]
    ):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(
            f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)})."
        )

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size,
        )

        num_synonym_triple = 0
        synonym_candidates = (
            []
        )  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(
            query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)
        ):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub("[^A-Za-z0-9]", "", entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if (
                        score < self.global_config.synonymy_edge_sim_threshold
                        or num_nns > 100
                    ):
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != "":
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = (
                            score  # Need to seriously discuss on this
                        )
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(
        self, chunk_keys: List[str]
    ) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(
            self.openie_results_path
        ):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get("docs", [])

            # Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info["idx"] = compute_mdhash_id(openie_info["passage"], "chunk-")
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info["idx"] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(
        self,
        all_openie_info: List[dict],
        chunks_to_save: Dict[str, dict],
        ner_results_dict: Dict[str, NerRawOutput],
        triple_results_dict: Dict[str, TripleRawOutput],
    ) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row["content"]
            try:
                chunk_openie_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_entities": ner_results_dict[chunk_key].unique_entities,
                    "extracted_triples": triple_results_dict[chunk_key].triples,
                }
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_entities": [],
                    "extracted_triples": [],
                }
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum(
            [len(e) for chunk in all_openie_info for e in chunk["extracted_entities"]]
        )
        sum_phrase_words = sum(
            [
                len(e.split())
                for chunk in all_openie_info
                for e in chunk["extracted_entities"]
            ]
        )
        num_phrases = sum(
            [len(chunk["extracted_entities"]) for chunk in all_openie_info]
        )

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                "docs": all_openie_info,
                "avg_ent_chars": avg_ent_chars,
                "avg_ent_words": avg_ent_words,
            }

            with open(self.openie_results_path, "w") as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {
            v["name"]: v for v in self.graph.vs if "name" in v.attributes()
        }

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node["name"] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(
                n=len(next(iter(new_nodes.values()))), attributes=new_nodes
            )

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(
            edge_source_node_keys, edge_target_node_keys, edge_metadata
        ):
            if (
                source_node_id in current_node_ids
                and target_node_id in current_node_ids
            ):
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(
                    f"Edge {source_node_id} -> {target_node_id} is not valid."
                )
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = (
            graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]
        )

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(
            self.fact_embedding_store.get_all_ids()
        )

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1
            for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info["num_triples_with_passage_node"] = num_triples_with_passage_node

        graph_info["num_synonymy_triples"] = (
            len(self.node_to_node_stats)
            - graph_info["num_extracted_triples"]
            - num_triples_with_passage_node
        )

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def get_metrics_snapshot(self) -> Dict:
        """Return a JSON-serialisable dict of all accumulated pipeline counters.

        Call once per run (after rag_qa completes) to capture a complete record
        for ablation comparison.  All counters reset with reset_metrics().
        """
        n = self.total_retrieve_queries or 1
        total_llm_calls = (
            self.reformulation_llm_calls
            + self.ircot_llm_calls
            + self.qa_llm_calls
        )
        total_prompt = (
            self.reformulation_prompt_tokens
            + self.ircot_prompt_tokens
            + self.qa_prompt_tokens
        )
        total_completion = (
            self.reformulation_completion_tokens
            + self.ircot_completion_tokens
            + self.qa_completion_tokens
        )
        reranker_keep_rate = (
            self.total_facts_after_rerank / self.total_facts_before_rerank
            if self.total_facts_before_rerank > 0
            else None
        )
        avg_hop_passages = (
            self.total_hop_passages_added / self.total_ircot_hops_done
            if self.total_ircot_hops_done > 0
            else None
        )
        n_init = self.initial_retrieve_count or 1
        n_hop = self.hop_retrieve_count or 1
        return {
            "total_retrieve_queries": self.total_retrieve_queries,
            "retrieval": {
                "fallback_count": self.fallback_count,
                "fallback_rate": round(self.fallback_count / n, 4),
                "initial_fallback_count": self.initial_fallback_count,
                "initial_retrieve_count": self.initial_retrieve_count,
                "initial_fallback_rate": round(self.initial_fallback_count / n_init, 4),
                "hop_fallback_count": self.hop_fallback_count,
                "hop_retrieve_count": self.hop_retrieve_count,
                "hop_fallback_rate": round(self.hop_fallback_count / n_hop, 4),
                "total_facts_before_rerank": self.total_facts_before_rerank,
                "total_facts_after_rerank": self.total_facts_after_rerank,
                "reranker_keep_rate": round(reranker_keep_rate, 4) if reranker_keep_rate is not None else None,
                "avg_facts_before_rerank": round(self.total_facts_before_rerank / n, 2),
                "avg_facts_after_rerank": round(self.total_facts_after_rerank / n, 2),
            },
            "enhancements": {
                "multi_query_count": self.multi_query_count,
                "multi_query_rate": round(self.multi_query_count / n, 4),
                "coverage_audit_injections": self.coverage_audit_injections,
                "coverage_audit_rate": round(self.coverage_audit_injections / n, 4),
                "ircot_terminal_count": self.ircot_terminal_count,
                "total_ircot_hops_done": self.total_ircot_hops_done,
                "avg_hop_passages": round(avg_hop_passages, 2) if avg_hop_passages is not None else None,
            },
            "llm_budget": {
                "reformulation": {
                    "calls": self.reformulation_llm_calls,
                    "prompt_tokens": self.reformulation_prompt_tokens,
                    "completion_tokens": self.reformulation_completion_tokens,
                },
                "ircot": {
                    "calls": self.ircot_llm_calls,
                    "prompt_tokens": self.ircot_prompt_tokens,
                    "completion_tokens": self.ircot_completion_tokens,
                },
                "qa": {
                    "calls": self.qa_llm_calls,
                    "prompt_tokens": self.qa_prompt_tokens,
                    "completion_tokens": self.qa_completion_tokens,
                },
                "total_calls": total_llm_calls,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
            "timing": {
                "total_retrieval_sec": round(self.all_retrieval_time, 2),
                "ppr_sec": round(self.ppr_time, 2),
                "rerank_sec": round(self.rerank_time, 2),
            },
        }

    def reset_metrics(self) -> None:
        """Reset all accumulated counters. Call before each ablation run."""
        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self.fallback_count = 0
        self.total_retrieve_queries = 0
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
        self.qa_llm_calls = 0
        self.qa_prompt_tokens = 0
        self.qa_completion_tokens = 0
        self.multi_query_count = 0
        self.coverage_audit_injections = 0
        self.ircot_terminal_count = 0
        self.total_facts_before_rerank = 0
        self.total_facts_after_rerank = 0
        self.total_hop_passages_added = 0
        self.total_ircot_hops_done = 0

    # --- Evaluation / Metric helpers ---
    def _approx_token_count(self, text: str) -> int:
        """
        Approximate token count for given text using a simple heuristic.

        This multiplies whitespace-separated word count by 1.33 to approximate
        subword tokenization. It's intentionally lightweight to avoid adding a
        hard dependency (e.g., tiktoken). Results are suitable for relative
        comparisons and large-scale monitoring.
        """
        if text is None:
            return 0
        words = len(text.split())
        return int(words * 1.33)

    def compute_retrieval_metrics(
        self,
        retrieval_results: List["QuerySolution"],
        gold_docs: List[List[str]],
        k: int = 5,
        use_ircot_context: bool = False,
    ) -> Dict:
        """
        Compute retrieval-centered metrics.

        When use_ircot_context=True, evaluates against the full accumulated context
        (initial retrieval + all IRCoT hop passages) stored in qs.ircot_context,
        rather than just the initial top-K docs.  K is set to the actual context
        size per query in that mode and reported as 'N' in the output dict.

        Returns a dict with:
        - ground_truth_counts: list of counts of gold passages per query
        - R@K / R@N:      any gold passage in retrieved set
        - AR@K / AR@N:    all gold passages in retrieved set
        - FirstHop@K/N:   first gold passage in retrieved set
        - LastHop@K/N:    last gold passage in retrieved set
        - K or N:         the cutoff used
        """
        assert len(retrieval_results) == len(gold_docs), "retrieval_results and gold_docs must align"

        total = len(gold_docs)
        ground_truth_counts = [len(g) for g in gold_docs]

        r_count = 0
        ar_count = 0
        lasthop_count = 0
        firsthop_count = 0
        context_sizes = []

        for rr, gold in zip(retrieval_results, gold_docs):
            if use_ircot_context and getattr(rr, "ircot_context", None):
                retrieved_set = set(rr.ircot_context)
                context_sizes.append(len(rr.ircot_context))
            else:
                retrieved_set = set(rr.docs[:k])
                context_sizes.append(k)

            gold_set = set(gold)

            if retrieved_set.intersection(gold_set):
                r_count += 1
            if gold_set.issubset(retrieved_set):
                ar_count += 1
            if gold and gold[-1] in retrieved_set:
                lasthop_count += 1
            if gold and gold[0] in retrieved_set:
                firsthop_count += 1

        label = "N" if use_ircot_context else "K"
        cutoff = round(sum(context_sizes) / len(context_sizes), 1) if context_sizes else k

        metrics = {
            "ground_truth_counts": ground_truth_counts,
            label: cutoff,
            f"R@{label}": round((r_count / total) * 100.0, 4) if total > 0 else 0.0,
            f"AR@{label}": round((ar_count / total) * 100.0, 4) if total > 0 else 0.0,
            f"FirstHop@{label}": round((firsthop_count / total) * 100.0, 4) if total > 0 else 0.0,
            f"LastHop@{label}": round((lasthop_count / total) * 100.0, 4) if total > 0 else 0.0,
        }

        return metrics

    def compute_qa_step_metrics(
        self,
        retrieval_results: List["QuerySolution"],
        predictions: List[str],
        gold_answers: List[List[str]],
        gold_docs: List[List[str]] | None = None,
    ) -> Dict:
        """
        Compute QA-stage metrics described in the spec.

        Returns a dict containing:
        - avg_context_tokens: average approximate input tokens supplied to the generator
        - context_coverage_pct: percent queries where at least one gold answer appears in context
        - final_accuracy_pct: percent queries where any predicted answer matches any gold answer
        - reasoning_failure_rate: percent of queries where gold evidence present but prediction incorrect
        - tokens_per_accuracy_point: total_tokens / (accuracy_pct) — useful to compare model cost-efficiency

        Notes:
        - "Context" is taken as the concatenation of retrieved passages for a query.
        - Token counts are approximate via `_approx_token_count`.
        """
        assert len(retrieval_results) == len(predictions) == len(gold_answers)

        total_queries = len(gold_answers)
        total_tokens = 0
        context_coverage_count = 0
        correct_count = 0
        reasoning_failure_count = 0

        for rr, pred, gold_ans_list, gold_doc_list in zip(
            retrieval_results, predictions, gold_answers, (gold_docs or [None] * total_queries)
        ):
            # build context
            context_text = "\n".join(rr.docs)
            tokens = self._approx_token_count(context_text)
            total_tokens += tokens

            # context coverage: check if any gold answer string appears in the context
            # fallback: if gold_doc_list provided, treat presence of any gold doc as coverage
            has_gold_in_context = False
            if gold_doc_list:
                retrieved_set = set(rr.docs)
                if any(gd in retrieved_set for gd in gold_doc_list):
                    has_gold_in_context = True
            else:
                for g_ans in gold_ans_list:
                    if g_ans and g_ans in context_text:
                        has_gold_in_context = True
                        break

            if has_gold_in_context:
                context_coverage_count += 1

            # final accuracy: exact-match against any gold answer (simple heuristic)
            is_correct = any(pred.strip() == g.strip() for g in gold_ans_list)
            if is_correct:
                correct_count += 1

            # reasoning failure: gold evidence present but prediction incorrect
            if has_gold_in_context and not is_correct:
                reasoning_failure_count += 1

        avg_context_tokens = total_tokens / total_queries if total_queries > 0 else 0
        final_accuracy_pct = (correct_count / total_queries) * 100.0 if total_queries > 0 else 0.0
        context_coverage_pct = (context_coverage_count / total_queries) * 100.0 if total_queries > 0 else 0.0
        reasoning_failure_rate = (reasoning_failure_count / total_queries) * 100.0 if total_queries > 0 else 0.0

        tokens_per_accuracy_point = (
            (total_tokens / final_accuracy_pct) if final_accuracy_pct > 0 else None
        )

        metrics = {
            "avg_context_tokens": int(avg_context_tokens),
            "total_tokens": int(total_tokens),
            "context_coverage_pct": round(context_coverage_pct, 4),
            "final_accuracy_pct": round(final_accuracy_pct, 4),
            "reasoning_failure_rate_pct": round(reasoning_failure_rate, 4),
            "tokens_per_accuracy_point": tokens_per_accuracy_point,
        }

        return metrics

    def _log_supporting_passage_ranks(
        self,
        q_idx: int,
        gold_docs: List[List[str]],
        sorted_doc_ids: np.ndarray,
    ) -> None:
        """
        For a single query, logs the 1-indexed rank of every gold/supporting passage
        in the full sorted retrieval list.  'NOT_INDEXED' means the passage was never
        inserted into the corpus; 'NOT_FOUND' means it was indexed but scored outside
        the returned sorted list (should not happen with a complete ranking).
        """
        text_to_hash = getattr(self.chunk_embedding_store, "text_to_hash_id", {})
        current_gold = gold_docs[q_idx]
        ranks = []
        for gold_doc in current_gold:
            h = text_to_hash.get(gold_doc)
            if h is None:
                ranks.append("NOT_INDEXED")
                continue
            local_idx = self.passage_key_to_local_idx.get(h)
            if local_idx is None:
                ranks.append("NOT_INDEXED")
                continue
            pos = np.nonzero(sorted_doc_ids == local_idx)[0]
            ranks.append(int(pos[0]) + 1 if len(pos) > 0 else "NOT_FOUND")
        logger.info(f"  Supporting passage ranks (1-indexed, {len(current_gold)} gold): {ranks}")

    # ------------------------------------------------------------------
    # Enhancement 1 — Query decomposition + multi-query fact retrieval
    # ------------------------------------------------------------------

    def reformulate_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Entity-count-based RAG Fusion (Enhancement 1).

        Extracts named entities from the query and generates one diverse
        retrieval query per entity, plus one bridge query when the question
        contains a relational phrase.  The LLM is forbidden from introducing
        entity names not present in the original question, preventing the
        second-hop information leakage of the old decomposition approach.

        Returns (queries, entities).  Falls back to ([query], []) on error.
        """
        messages = [
            {"role": "system", "content": _REFORMULATE_SYSTEM_MSG},
            {"role": "user", "content": _REFORMULATE_USER_TMPL.format(query=query)},
        ]
        try:
            response, _meta, _cache = self.llm_model.infer(messages)
            self.reformulation_llm_calls += 1
            self.reformulation_prompt_tokens += _meta.get("prompt_tokens", 0)
            self.reformulation_completion_tokens += _meta.get("completion_tokens", 0)
            text = response.strip()
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
            result = json.loads(text)
            entities = [e for e in (result.get("entities") or []) if isinstance(e, str) and e.strip()]
            queries = [q for q in (result.get("queries") or []) if isinstance(q, str) and q.strip()]
            if not queries:
                return [query], entities
            logger.info(
                f"  [RAG Fusion] {len(entities)} entities → {len(queries)} queries: {queries}"
            )
            logger.debug(f"  [RAG Fusion] entities: {entities}")
            return queries, entities
        except Exception as e:
            self.reformulation_llm_calls += 1
            logger.warning(f"  reformulate_query failed ({e}) — using original query")
            return [query], []

    def get_multi_query_fact_scores(
        self, sub_queries: List[str], rrf_k: int = 60
    ) -> np.ndarray:
        """
        Run one fact-embedding lookup per sub-query, then merge with
        Reciprocal Rank Fusion (RRF).  Returns a normalised score array
        of shape (num_facts,) — same shape as get_fact_scores(), so the
        rest of the pipeline is unchanged.
        """
        rrf_scores = np.zeros(len(self.fact_node_keys))

        for sq in sub_queries:
            scores = self.get_fact_scores(sq)
            if len(scores) == 0:
                continue
            ranks = np.argsort(scores)[::-1]  # descending; rank 0 = best
            for rank_pos, fact_idx in enumerate(ranks):
                rrf_scores[fact_idx] += 1.0 / (rank_pos + 1 + rrf_k)

        max_score = rrf_scores.max()
        if max_score > 0:
            rrf_scores /= max_score

        logger.info(
            f"  RRF ({len(sub_queries)} sub-queries): "
            f"nonzero={np.count_nonzero(rrf_scores)}, "
            f"max={rrf_scores.max():.4f}, mean={rrf_scores[rrf_scores > 0].mean():.4f}"
        )
        return rrf_scores

    # ------------------------------------------------------------------
    # Enhancement 2 — Coverage-constrained reranking
    # ------------------------------------------------------------------

    def coverage_audit(
        self,
        kept_facts: List[Tuple],
        kept_indices: List[int],
        all_candidates: List[Tuple],
        all_candidate_indices: List[int],
        query_entities: List[str],
        required_relations: List[str] = None,
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Post-rerank guard (Enhancement 2 — Fix B).

        Phase 1 — entity coverage: for every query entity absent from kept_facts,
        inject the highest-scoring candidate that mentions it.

        Phase 2 — relation coverage: for every predicate type in required_relations
        absent from the kept facts' predicates, inject the best candidate whose
        predicate contains that relation keyword.
        """
        def _entity_in_fact(entity: str, fact: Tuple) -> bool:
            el = entity.lower()
            return any(el in str(part).lower() for part in fact)

        def _relation_in_fact(relation: str, fact: Tuple) -> bool:
            return relation.lower() in str(fact[1]).lower()

        # Build token coverage set from already-kept facts
        covered_tokens: Set[str] = set()
        for fact in kept_facts:
            for part in fact:
                covered_tokens.add(str(part).lower())

        result_facts = list(kept_facts)
        result_indices = list(kept_indices)
        kept_idx_set = set(kept_indices)
        injected: List[Tuple[str, Tuple]] = []

        # Phase 1: entity coverage
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
                    injected.append((f"entity:{entity}", cand_fact))
                    for part in cand_fact:
                        covered_tokens.add(str(part).lower())
                    break

        # Phase 2: relation coverage (Fix B)
        if required_relations:
            covered_predicates = {str(f[1]).lower() for f in result_facts}
            for relation in required_relations:
                rel_lower = relation.lower()
                if any(rel_lower in pred for pred in covered_predicates):
                    continue

                for cand_fact, cand_idx in zip(all_candidates, all_candidate_indices):
                    if cand_idx in kept_idx_set:
                        continue
                    if _relation_in_fact(relation, cand_fact):
                        result_facts.append(cand_fact)
                        result_indices.append(cand_idx)
                        kept_idx_set.add(cand_idx)
                        injected.append((f"relation:{relation}", cand_fact))
                        covered_predicates.add(str(cand_fact[1]).lower())
                        break

        if injected:
            for label, fact in injected:
                logger.info(f"  Coverage audit ✚ injected for '{label}': {fact}")
        else:
            logger.debug("  Coverage audit: all entities and relations already covered")

        return result_facts, result_indices

    def entity_partitioned_selection(
        self,
        candidates: List[Tuple],
        candidate_indices: List[int],
        query_entities: List[str],
        quota_per_entity: int = 1,
        max_total: int = 5,
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Enhancement 2 Step 2c — hard quota guarantee.

        Partitions the candidate pool by entity and ensures at least
        quota_per_entity facts per entity appear in the final selection,
        up to max_total facts total.  Can be used instead of coverage_audit
        for a zero-LLM-cost constraint.
        """
        entity_buckets: Dict[str, List[Tuple[Tuple, int]]] = {e: [] for e in query_entities}
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
        for entity, bucket in entity_buckets.items():
            for fact, idx in bucket[:quota_per_entity]:
                selected_facts.append(fact)
                selected_indices.append(idx)
            logger.debug(
                f"  entity_partitioned_selection: entity='{entity}' "
                f"bucket={len(bucket)}, kept={min(len(bucket), quota_per_entity)}"
            )

        remaining = max_total - len(selected_facts)
        for fact, idx in overflow[:max(remaining, 0)]:
            selected_facts.append(fact)
            selected_indices.append(idx)

        logger.info(
            f"  entity_partitioned_selection: {len(candidates)} → {len(selected_facts)} "
            f"(quota={quota_per_entity}/entity, max={max_total})"
        )
        return selected_facts[:max_total], selected_indices[:max_total]

    # ------------------------------------------------------------------
    # Enhancement 2 Fix C — NER-seeded graph fallback
    # ------------------------------------------------------------------

    def _ner_seeded_fallback(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fix C: when rerank_facts returns 0 kept facts, run NER on the top DPR
        passage and use the extracted entities as PPR seeds rather than falling
        back to flat dense retrieval.

        The top DPR passage is correct (gold rank 1-3) in the majority of
        zero-fact cases, so NER on it gives graph seeds that let PPR walk to
        the second-hop passage.  Falls back to flat DPR on any failure.
        """
        dpr_doc_ids, dpr_doc_scores = self.dense_passage_retrieval(query)

        top_passage_key = self.passage_node_keys[dpr_doc_ids[0]]
        top_passage_text = self.chunk_embedding_store.get_row(top_passage_key)["content"]

        ner_result = self.openie.ner(chunk_key=top_passage_key, passage=top_passage_text)
        entities = ner_result.unique_entities

        if not entities:
            logger.info("  NER-seeded fallback: no entities found — using flat DPR")
            return dpr_doc_ids, dpr_doc_scores

        logger.info(
            f"  NER-seeded fallback: {len(entities)} entities extracted: {entities[:6]}"
        )

        # Apply same text normalisation used during graph construction
        processed = [text_processing(e) for e in entities if e.strip()]
        if not processed:
            return dpr_doc_ids, dpr_doc_scores

        # Synthetic (entity, predicate, entity) stubs — subject == object so PPR
        # seeds from that entity node without introducing spurious edges.
        synthetic_facts = [(e, "mentioned in", e) for e in processed]
        synthetic_scores = np.ones(len(synthetic_facts))
        synthetic_indices = list(range(len(synthetic_facts)))

        try:
            doc_ids, doc_scores = self.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.global_config.linking_top_k,
                query_fact_scores=synthetic_scores,
                top_k_facts=synthetic_facts,
                top_k_fact_indices=synthetic_indices,
                passage_node_weight=self.global_config.passage_node_weight,
            )
            logger.info("  NER-seeded fallback: graph search succeeded")
            return doc_ids, doc_scores
        except AssertionError:
            logger.info("  NER-seeded fallback: no NER entities found in graph — using flat DPR")
            return dpr_doc_ids, dpr_doc_scores
        except Exception as e:
            logger.warning(f"  NER-seeded fallback: graph search failed ({e}) — using flat DPR")
            return dpr_doc_ids, dpr_doc_scores

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {"triple": {}, "passage": {}}

        self.entity_node_keys: List = list(
            self.entity_embedding_store.get_all_ids()
        )  # a list of phrase node keys
        self.passage_node_keys: List = list(
            self.chunk_embedding_store.get_all_ids()
        )  # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # reverse map: passage hash_id → position in passage_node_keys (used for gold-doc rank lookup)
        self.passage_key_to_local_idx: Dict[str, int] = {
            key: idx for idx, key in enumerate(self.passage_node_keys)
        }

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()

        if expected_node_count != actual_node_count:
            logger.warning(
                f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}"
            )
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {
                node["name"]: idx for idx, node in enumerate(self.graph.vs)
            }  # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx

            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [
                node_key
                for node_key in self.entity_node_keys
                if node_key not in igraph_name_to_idx
            ]
            missing_passage_nodes = [
                node_key
                for node_key in self.passage_node_keys
                if node_key not in igraph_name_to_idx
            ]

            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(
                    f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes"
                )
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {
                    node["name"]: idx for idx, node in enumerate(self.graph.vs)
                }
                self.node_name_to_vertex_idx = igraph_name_to_idx

            self.entity_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.entity_node_keys
            ]  # a list of backbone graph node index
            self.passage_node_idxs = [
                igraph_name_to_idx[node_key] for node_key in self.passage_node_keys
            ]  # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(
            self.entity_embedding_store.get_embeddings(self.entity_node_keys)
        )
        self.passage_embeddings = np.array(
            self.chunk_embedding_store.get_embeddings(self.passage_node_keys)
        )

        self.fact_embeddings = np.array(
            self.fact_embedding_store.get_embeddings(self.fact_node_keys)
        )

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc["extracted_triples"]])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = (
                        self.proc_triples_to_docs.get(str(proc_triple), set()).union(
                            set([doc["idx"]])
                        )
                    )

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(
                all_openie_info
            )

            # Check if the lengths match
            if not (
                len(self.passage_node_keys)
                == len(ner_results_dict)
                == len(triple_results_dict)
            ):
                logger.warning(
                    f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}"
                )

                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[],
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id, response=None, metadata={}, triples=[]
                        )

            # prepare data_store
            chunk_triples = [
                [text_processing(t) for t in triple_results_dict[chunk_id].triples]
                for chunk_id in self.passage_node_keys
            ]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                query.question not in self.query_to_embedding["triple"]
                or query.question not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query.question)
            elif (
                query not in self.query_to_embedding["triple"]
                or query not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(
                all_query_strings,
                instruction=get_query_instruction("query_to_fact"),
                norm=True,
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding["triple"][query] = embedding

            logger.info(
                f"Encoding {len(all_query_strings)} queries for query_to_passage."
            )
            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings,
                instruction=get_query_instruction("query_to_passage"),
                norm=True,
            )
            for query, embedding in zip(
                all_query_strings, query_embeddings_for_passage
            ):
                self.query_to_embedding["passage"][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding["triple"].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_fact"), norm=True
            )

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        try:
            query_fact_scores = np.dot(
                self.fact_embeddings, query_embedding.T
            )  # shape: (#facts, )
            query_fact_scores = (
                np.squeeze(query_fact_scores)
                if query_fact_scores.ndim == 2
                else query_fact_scores
            )
            query_fact_scores = min_max_normalize(query_fact_scores)
            # logger.debug(
            #     f"get_fact_scores: n={len(query_fact_scores)}, "
            #     f"min={query_fact_scores.min():.4f}, max={query_fact_scores.max():.4f}, "
            #     f"mean={query_fact_scores.mean():.4f}"
            # )
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding["passage"].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_passage"), norm=True
            )
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = (
            np.squeeze(query_doc_scores)
            if query_doc_scores.ndim == 2
            else query_doc_scores
        )
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(
        self,
        link_top_k: int,
        all_phrase_weights: np.ndarray,
        linking_score_map: Dict[str, float],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(
            sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[
                :link_top_k
            ]
        )

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [
                compute_mdhash_id(content=top_k_phrase, prefix="entity-")
                for top_k_phrase in top_k_phrases
            ]
        )

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(
        self,
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,
        top_k_facts: List[Tuple],
        top_k_fact_indices: List[str],
        passage_node_weight: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = (
            {}
        )  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = (
            {}
        )  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs["name"]))
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        number_of_occurs = np.zeros(len(self.graph.vs["name"]))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = (
                query_fact_scores[top_k_fact_indices[rank]]
                if query_fact_scores.ndim > 0
                else query_fact_scores
            )

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(
                            self.ent_node_to_chunk_ids[phrase_key]
                        )

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(
                link_top_k, phrase_weights, linking_score_map
            )  # at this stage, the length of linking_scope_map is determined by link_top_k

        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)[
                "content"
            ]
            linking_score_map[passage_node_text] = (
                passage_dpr_score * passage_node_weight
            )

        # Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
            )

        assert (
            sum(node_weights) > 0
        ), f"No phrases found in the graph for the given facts: {top_k_facts}"

        nonzero_phrases = int(np.count_nonzero(phrase_weights))
        logger.debug(
            f"graph_search: {nonzero_phrases} nonzero phrase nodes, "
            f"{len(linking_score_map)} entries in linking_score_map"
        )

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights, damping=self.global_config.damping
        )
        ppr_end = time.time()

        self.ppr_time += ppr_end - ppr_start
        logger.debug(f"  PPR done in {ppr_end - ppr_start:.2f}s | top doc score={ppr_sorted_doc_scores[0]:.4f}")

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs
        ), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def _rerank_facts_from_indices(
        self, query: str, candidate_indices: List[int]
    ) -> Tuple[List[int], List[Tuple], dict]:
        """Rerank a pre-selected set of candidate fact indices with the LLM reranker.

        Used by Enhancement 1 multi-query union path: caller collects the top-N facts
        from each sub-query, unions the index sets, then calls this to let the reranker
        pick the relevant subset using the original question.
        """
        empty_log = {"facts_before_rerank": [], "facts_after_rerank": [], "facts_before_rerank_indices": []}
        if not candidate_indices or len(self.fact_node_keys) == 0:
            return [], [], empty_log
        try:
            real_ids = [self.fact_node_keys[idx] for idx in candidate_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_ids)
            candidate_facts = [eval(fact_row_dict[fid]["content"]) for fid in real_ids]
            logger.info(
                f"  _rerank_facts_from_indices: {len(candidate_facts)} candidates → reranker"
            )
            top_k_indices, top_k_facts, _ = self.rerank_filter(
                query,
                candidate_facts,
                candidate_indices,
                len_after_rerank=self.global_config.linking_top_k,
            )
            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_before_rerank_indices": candidate_indices,
                "facts_after_rerank": top_k_facts,
            }
            return top_k_indices, top_k_facts, rerank_log
        except Exception as e:
            logger.error(f"Error in _rerank_facts_from_indices: {e}", exc_info=True)
            return [], [], empty_log

    def rerank_facts(
        self, query: str, query_fact_scores: np.ndarray
    ) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][
                    ::-1
                ].tolist()

            candidate_scores = [float(query_fact_scores[i]) for i in candidate_fact_indices]
            logger.info(
                f"  rerank_facts: selected {len(candidate_fact_indices)} candidates "
                f"(scores {candidate_scores[0]:.4f} … {candidate_scores[-1]:.4f})"
            )

            # Get the actual fact IDs
            real_candidate_fact_ids = [
                self.fact_node_keys[idx] for idx in candidate_fact_indices
            ]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [
                eval(fact_row_dict[id]["content"]) for id in real_candidate_fact_ids
            ]

            logger.debug("  Candidate facts (score → triple):")
            for score, fact in zip(candidate_scores, candidate_facts):
                logger.debug(f"    {score:.4f}  {fact}")

            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=link_top_k,
            )

            logger.info(
                f"  rerank_facts: {len(candidate_facts)} → {len(top_k_facts)} facts kept"
            )
            if top_k_facts:
                logger.debug("  Facts after rerank:")
                for fact in top_k_facts:
                    logger.debug(f"    {fact}")

            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_before_rerank_indices": candidate_fact_indices,
                "facts_after_rerank": top_k_facts,
            }

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}", exc_info=True)
            return (
                [],
                [],
                {"facts_before_rerank": [], "facts_after_rerank": [], "error": str(e)},
            )

    # ------------------------------------------------------------------
    # Enhancement 3 — IRCoT Iterative Retrieval Loop
    # ------------------------------------------------------------------

    def _ircot_step(
        self,
        template_name: str,
        query: str,
        passages: List[str],
        thoughts: List[str],
    ) -> Tuple[str, List[str], Optional[str], dict]:
        """Generate one IRCoT reasoning step.

        Returns (thought, missing_queries, answer, metadata).
        - missing_queries is a non-empty list when the model identifies gaps and
          wants targeted retrieval (one question per missing entity/fact).
        - answer is non-None when the model has enough evidence to conclude.
        - Both missing_queries==[] and answer==None on parse failure.
        """
        prompt_user = ""
        for passage in passages:
            prompt_user += f"Wikipedia Title: {passage}\n\n"
        prompt_user += "Question: " + query
        if thoughts:
            prompt_user += "\nPrevious reasoning: " + " ".join(thoughts)
        messages = self.prompt_template_manager.render(
            name=template_name, prompt_user=prompt_user
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
            if answer:
                return thought, [], answer, meta
            return thought, missing_queries, None, meta
        except Exception as e:
            logger.warning(f"  [IRCoT] step parse failed ({e}) — treating as no-op")
            return "", [], None, {"prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "error"}

    def _ircot_final_qa(
        self,
        query: str,
        passages: List[str],
        cot_sentences: List[str],
    ) -> Tuple[str, str, dict]:
        """Run final QA with accumulated CoT injected as Thought prefix."""
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
        response, metadata, _cache = self.llm_model.infer(messages)
        try:
            answer = response.split("Answer:")[1].strip()
        except Exception:
            answer = response
        return answer, response, metadata

    def qa_with_ircot(
        self,
        queries: List["QuerySolution"],
        gold_docs: List[List[str]] = None,
    ) -> Tuple[List["QuerySolution"], List[str], List[Dict]]:
        """
        Enhancement 3: structured-output IRCoT loop.

        Each step the LLM receives the current passages + question + accumulated
        reasoning and returns JSON with one of two keys:
          - "next_query": the LLM has identified a gap — retrieve with this
            targeted query and continue.
          - "answer": the passages contain sufficient evidence — terminate.

        When max_qa_steps is exhausted without an answer, fall back to a final
        single-shot QA call with all accumulated passages.

        Activated when max_qa_steps > 1.  use_enhancements controls only E1
        (RAG Fusion on the *initial* retrieval); hop retrievals always use
        skip_enhancements=True regardless.
        """
        dataset = self.global_config.dataset or "musique"
        ircot_template = f"ircot_{dataset}"
        if not self.prompt_template_manager.is_template_name_valid(ircot_template):
            ircot_template = "ircot_musique"
        logger.info(
            f"[IRCoT] using template={ircot_template!r}, max_steps={self.global_config.max_qa_steps}"
        )

        queries_solutions: List["QuerySolution"] = []
        all_response_message: List[str] = []
        all_metadata: List[Dict] = []

        total_prompt_tokens_all = 0
        total_completion_tokens_all = 0
        total_llm_calls_all = 0
        total_ircot_steps_all = 0
        min_ircot_steps = float("inf")
        max_ircot_steps = 0

        # Per-hop recall accumulators: hop_recall_any[i] = # queries where any gold
        # was found in accumulated context after hop i (hop 0 = initial retrieval).
        max_possible_steps = self.global_config.max_qa_steps
        hop_recall_any: List[int] = [0] * max_possible_steps
        hop_recall_all: List[int] = [0] * max_possible_steps
        hop_recall_counts: List[int] = [0] * max_possible_steps  # queries that reached this hop

        n_queries = len(queries)
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

            # Per-query gold set for hop recall tracking
            q_gold = set(gold_docs[q_idx]) if gold_docs is not None and q_idx < len(gold_docs) else None

            logger.info(
                f"[IRCoT query {q_idx + 1}/{n_queries}] {query!r} | "
                f"initial_passages={len(all_passages)}"
            )

            # Record recall at hop 0 (initial retrieval)
            if q_gold is not None:
                ctx0 = set(all_passages)
                hop_recall_counts[0] += 1
                if ctx0 & q_gold:
                    hop_recall_any[0] += 1
                if q_gold.issubset(ctx0):
                    hop_recall_all[0] += 1

            for step in range(self.global_config.max_qa_steps - 1):
                logger.info(
                    f"  [IRCoT step {step + 1}/{self.global_config.max_qa_steps - 1}] "
                    f"passages_in_context={len(all_passages)} cot_so_far={len(cot_sentences)}"
                )
                thought, missing_queries, answer, step_meta = self._ircot_step(
                    ircot_template, query, all_passages, cot_sentences
                )
                q_ircot_prompt_tokens += step_meta.get("prompt_tokens", 0)
                q_ircot_completion_tokens += step_meta.get("completion_tokens", 0)
                q_ircot_llm_calls += 1

                if not thought and not missing_queries and answer is None:
                    logger.info("  [IRCoT] empty/unparseable response — stopping loop")
                    break

                if thought:
                    cot_sentences.append(thought)
                    logger.info(f"  [IRCoT] thought: {thought[:120]!r}")

                if answer is not None:
                    terminal_answer = answer.strip().rstrip(".")
                    logger.info(f"  [IRCoT] answer found → {terminal_answer!r}")
                    break

                if missing_queries:
                    logger.info(
                        f"  [IRCoT] missing_queries ({len(missing_queries)}) → "
                        f"direct retrieval (no reformulation): {missing_queries}"
                    )
                    passages_before = len(all_passages)
                    seen = set(all_passages)
                    cap = self.global_config.qa_top_k * 2
                    for mq in missing_queries:
                        hop_results = self.retrieve(
                            [mq],
                            num_to_retrieve=self.global_config.qa_top_k,
                            skip_enhancements=True,
                        )
                        for p in hop_results[0].docs:
                            if p not in seen and len(all_passages) < cap:
                                all_passages.append(p)
                                seen.add(p)
                    hop_added = len(all_passages) - passages_before
                    self.total_hop_passages_added += hop_added
                    self.total_ircot_hops_done += 1
                    hop_idx = step + 1  # hop 0 = initial; hop 1 = after first IRCoT step, etc.
                    if q_gold is not None and hop_idx < max_possible_steps:
                        ctx_now = set(all_passages)
                        hop_recall_counts[hop_idx] += 1
                        if ctx_now & q_gold:
                            hop_recall_any[hop_idx] += 1
                        if q_gold.issubset(ctx_now):
                            hop_recall_all[hop_idx] += 1
                        logger.info(
                            f"  [IRCoT] hop {hop_idx} recall: "
                            f"any={'yes' if ctx_now & q_gold else 'no'} "
                            f"all={'yes' if q_gold.issubset(ctx_now) else 'no'}"
                        )
                    logger.info(
                        f"  [IRCoT] hop added {hop_added} new passages "
                        f"(context now {len(all_passages)})"
                    )
                else:
                    logger.info("  [IRCoT] no missing_queries — continuing with same context")

            # Record the full accumulated context so context-recall metrics can be computed
            qs.ircot_context = list(all_passages)

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
                logger.info(
                    f"  [IRCoT] max steps reached — running final QA with "
                    f"{len(all_passages)} passages and {len(cot_sentences)} reasoning step(s)"
                )
                answer, raw_msg, metadata = self._ircot_final_qa(
                    query=query,
                    passages=all_passages,
                    cot_sentences=cot_sentences,
                )
                q_qa_prompt_tokens += metadata.get("prompt_tokens", 0)
                q_qa_completion_tokens += metadata.get("completion_tokens", 0)
                q_qa_llm_calls += 1
                qs.answer = answer
                logger.info(f"  [IRCoT] final answer={qs.answer!r}")

            q_ircot_steps = len(cot_sentences)
            qs.ircot_steps = q_ircot_steps
            q_total_ircot_tokens = q_ircot_prompt_tokens + q_ircot_completion_tokens
            q_total_qa_tokens = q_qa_prompt_tokens + q_qa_completion_tokens
            q_llm_calls = q_ircot_llm_calls + q_qa_llm_calls
            logger.info(
                f"  [IRCoT metrics] steps={q_ircot_steps} "
                f"ircot_calls={q_ircot_llm_calls} (tokens={q_total_ircot_tokens}) | "
                f"qa_calls={q_qa_llm_calls} (tokens={q_total_qa_tokens}) | "
                f"terminal={'yes' if terminal_answer is not None else 'no'}"
            )

            self.ircot_llm_calls += q_ircot_llm_calls
            self.ircot_prompt_tokens += q_ircot_prompt_tokens
            self.ircot_completion_tokens += q_ircot_completion_tokens
            self.qa_llm_calls += q_qa_llm_calls
            self.qa_prompt_tokens += q_qa_prompt_tokens
            self.qa_completion_tokens += q_qa_completion_tokens

            total_prompt_tokens_all += q_ircot_prompt_tokens + q_qa_prompt_tokens
            total_completion_tokens_all += q_ircot_completion_tokens + q_qa_completion_tokens
            total_llm_calls_all += q_llm_calls
            total_ircot_steps_all += q_ircot_steps
            min_ircot_steps = min(min_ircot_steps, q_ircot_steps)
            max_ircot_steps = max(max_ircot_steps, q_ircot_steps)

            queries_solutions.append(qs)
            all_response_message.append(raw_msg)
            all_metadata.append(metadata)

        n = len(queries)
        if n > 0:
            avg_steps = total_ircot_steps_all / n
            min_steps = min_ircot_steps if min_ircot_steps != float("inf") else 0
            logger.info(
                f"[IRCoT summary] queries={n} | "
                f"steps avg={avg_steps:.2f} min={min_steps} max={max_ircot_steps} | "
                f"avg_llm_calls={total_llm_calls_all / n:.2f} "
                f"avg_tokens={(total_prompt_tokens_all + total_completion_tokens_all) / n:.1f} "
                f"(avg_prompt={total_prompt_tokens_all / n:.1f} "
                f"avg_completion={total_completion_tokens_all / n:.1f}) | "
                f"total_tokens={total_prompt_tokens_all + total_completion_tokens_all}"
            )

            if gold_docs is not None:
                hop_recall_lines = []
                for h in range(max_possible_steps):
                    cnt = hop_recall_counts[h]
                    if cnt == 0:
                        break
                    r_any = 100.0 * hop_recall_any[h] / cnt
                    r_all = 100.0 * hop_recall_all[h] / cnt
                    label = "initial" if h == 0 else f"hop {h}"
                    hop_recall_lines.append(
                        f"    {label:>10}: R@ctx={r_any:.1f}%  AR@ctx={r_all:.1f}%  (n={cnt})"
                    )
                if hop_recall_lines:
                    logger.info(
                        "[IRCoT per-hop recall (incremental context)]\n"
                        + "\n".join(hop_recall_lines)
                    )

        return queries_solutions, all_response_message, all_metadata

    def run_ppr(
        self, reset_prob: np.ndarray, damping: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None:
            damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
