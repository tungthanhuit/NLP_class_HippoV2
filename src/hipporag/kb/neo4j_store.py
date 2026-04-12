from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from neo4j import Driver, GraphDatabase

from ..utils.misc_utils import compute_mdhash_id, text_processing

logger = logging.getLogger(__name__)


def _process_text_to_str(text: str) -> str:
    processed = text_processing(text)
    if isinstance(processed, list):
        return " ".join(str(x) for x in processed)
    return str(processed)


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    namespace: str = "hipporag"
    batch_size: int = 500


class Neo4jKB:
    """Neo4j-backed persistence for HippoRAG KB.

    Stores:
      - Chunk/Entity/Fact nodes with content + embedding
      - OpenIE extraction on Chunk nodes
      - Graph edges as relationships with weights

    Notes:
      - No Neo4j vector indexes are used.
      - PageRank stays in Python; Neo4j stores the graph topology only.
    """

    CHUNK_LABEL = "HippoChunk"
    ENTITY_LABEL = "HippoEntity"
    FACT_LABEL = "HippoFact"

    REL_HAS_ENTITY = "HAS_ENTITY"
    REL_HAS_FACT = "HAS_FACT"
    REL_CO_OCCUR = "CO_OCCUR"
    REL_SYNONYM = "SYNONYM"

    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver: Driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.user, cfg.password),
        )
        self._ensure_schema()

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            logger.exception("Failed closing Neo4j driver")

    def _ensure_schema(self) -> None:
        stmts = [
            f"CREATE CONSTRAINT hippo_chunk_key IF NOT EXISTS FOR (n:{self.CHUNK_LABEL}) REQUIRE (n.ns, n.id) IS UNIQUE",
            f"CREATE CONSTRAINT hippo_entity_key IF NOT EXISTS FOR (n:{self.ENTITY_LABEL}) REQUIRE (n.ns, n.id) IS UNIQUE",
            f"CREATE CONSTRAINT hippo_fact_key IF NOT EXISTS FOR (n:{self.FACT_LABEL}) REQUIRE (n.ns, n.id) IS UNIQUE",
        ]
        with self.driver.session(database=self.cfg.database) as session:
            for stmt in stmts:
                try:
                    session.run(stmt)
                except Exception:
                    # Neo4j Community/older instances might not support composite constraints in this form.
                    logger.exception("Schema statement failed: %s", stmt)

    def clear_namespace(
        self,
        namespace: Optional[str] = None,
        database: Optional[str] = None,
    ) -> int:
        """Delete *all* nodes/relationships for a namespace in a Neo4j database.

        This deletes everything that matches `n.ns = namespace` (any label), which includes
        HippoRAG chunk/entity/fact nodes and their embedding properties.

        Returns:
            The number of nodes matched (best-effort count before deletion).
        """

        ns = (namespace or self.cfg.namespace or "").strip()
        if not ns:
            raise ValueError("namespace must be a non-empty string")

        db = (database or self.cfg.database or "neo4j").strip() or "neo4j"

        count_query = "MATCH (n {ns: $ns}) RETURN count(n) AS c"
        delete_query = "MATCH (n {ns: $ns}) DETACH DELETE n"

        with self.driver.session(database=db) as session:
            rec = session.execute_read(
                lambda tx: tx.run(count_query, ns=ns).single()
            )
            matched = int(rec["c"]) if rec and rec.get("c") is not None else 0

            session.execute_write(lambda tx: tx.run(delete_query, ns=ns))

        return matched

    @staticmethod
    def clear_neo4j_namespace(
        *,
        uri: str,
        user: str,
        password: str,
        namespace: str,
        database: str = "neo4j",
        batch_size: int = 500,
    ) -> int:
        """Convenience helper to clear a namespace without constructing HippoRAG.

        Returns:
            The number of nodes matched (best-effort count before deletion).
        """

        kb = Neo4jKB(
            Neo4jConfig(
                uri=uri,
                user=user,
                password=password,
                database=database,
                namespace=namespace,
                batch_size=batch_size,
            )
        )
        try:
            return kb.clear_namespace(namespace=namespace, database=database)
        finally:
            kb.close()

    # -------------------- Embedding nodes --------------------
    def upsert_nodes(
        self,
        label: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        if not rows:
            return

        query = (
            f"UNWIND $rows AS row\n"
            f"MERGE (n:{label} {{ns: $ns, id: row.id}})\n"
            f"SET n.content = row.content, n.embedding = row.embedding"
        )

        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(rows, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, rows=b)
                )

    def delete_nodes(self, label: str, ids: List[str]) -> None:
        if not ids:
            return
        query = f"UNWIND $ids AS id MATCH (n:{label} {{ns: $ns, id: id}}) DETACH DELETE n"
        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(ids, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, ids=b)
                )

    def load_all_nodes(self, label: str) -> Tuple[List[str], List[str], List[List[float]]]:
        query = (
            f"MATCH (n:{label} {{ns: $ns}}) "
            "RETURN n.id AS id, n['content'] AS content, n['embedding'] AS embedding"
        )
        ids: List[str] = []
        contents: List[str] = []
        embeddings: List[List[float]] = []
        with self.driver.session(database=self.cfg.database) as session:
            result = session.execute_read(lambda tx: list(tx.run(query, ns=self.cfg.namespace)))
            for rec in result:
                ids.append(rec["id"])
                contents.append(rec["content"])
                embeddings.append(rec.get("embedding") or [])
        return ids, contents, embeddings

    def load_nodes_by_ids(self, label: str, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        query = (
            f"UNWIND $ids AS id\n"
            f"MATCH (n:{label} {{ns: $ns, id: id}})\n"
            "RETURN n.id AS id, n['content'] AS content"
        )
        out: Dict[str, Dict[str, Any]] = {}
        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(ids, self.cfg.batch_size):
                records = session.execute_read(
                    lambda tx, b=batch: list(tx.run(query, ns=self.cfg.namespace, ids=b))
                )
                for rec in records:
                    out[rec["id"]] = {"hash_id": rec["id"], "content": rec["content"]}
        return out

    # -------------------- OpenIE --------------------
    def upsert_chunk_openie(self, openie_docs: List[Dict[str, Any]]) -> None:
        if not openie_docs:
            return

        # Properties must be primitives; store triples as JSON strings.
        rows = []
        for doc in openie_docs:
            triples = doc.get("extracted_triples") or []
            rows.append(
                {
                    "id": doc["idx"],
                    "entities": doc.get("extracted_entities") or [],
                    "triples": [json.dumps(t, ensure_ascii=False) for t in triples],
                }
            )

        query = (
            f"UNWIND $rows AS row\n"
            f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns, id: row.id}})\n"
            "SET c.openie_entities = row.entities, c.openie_triples = row.triples"
        )

        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(rows, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, rows=b)
                )

    def load_openie_docs(self, chunk_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if chunk_ids is None:
            query = (
                f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns}}) "
                "WITH c "
                "RETURN c.id AS id, "
                "c[$content_key] AS passage, "
                "CASE WHEN $ents_key IN keys(c) THEN c[$ents_key] ELSE [] END AS ents, "
                "CASE WHEN $triples_key IN keys(c) THEN c[$triples_key] ELSE [] END AS triples, "
                "$triples_key IN keys(c) AS has_openie"
            )
            params = {
                "ns": self.cfg.namespace,
                "content_key": "content",
                "ents_key": "openie_entities",
                "triples_key": "openie_triples",
            }
        else:
            query = (
                "UNWIND $ids AS id\n"
                f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns, id: id}})\n"
                "WITH c "
                "RETURN c.id AS id, "
                "c[$content_key] AS passage, "
                "CASE WHEN $ents_key IN keys(c) THEN c[$ents_key] ELSE [] END AS ents, "
                "CASE WHEN $triples_key IN keys(c) THEN c[$triples_key] ELSE [] END AS triples, "
                "$triples_key IN keys(c) AS has_openie"
            )
            params = {
                "ns": self.cfg.namespace,
                "ids": chunk_ids,
                "content_key": "content",
                "ents_key": "openie_entities",
                "triples_key": "openie_triples",
            }

        docs: List[Dict[str, Any]] = []
        with self.driver.session(database=self.cfg.database) as session:
            records = session.execute_read(lambda tx: list(tx.run(query, **params)))
            for rec in records:
                triples = rec.get("triples") or []
                parsed_triples: List[List[str]] = []
                for t in triples:
                    try:
                        parsed_triples.append(json.loads(t))
                    except Exception:
                        # Backward/dirty data; keep best-effort.
                        parsed_triples.append([str(t)])
                docs.append(
                    {
                        "idx": rec["id"],
                        "passage": rec.get("passage") or "",
                        "extracted_entities": rec.get("ents") or [],
                        "extracted_triples": parsed_triples,
                        "has_openie": bool(rec.get("has_openie")),
                    }
                )
        return docs

    # -------------------- Graph edges --------------------
    def upsert_chunk_entity_edges(self, chunk_id: str, entity_ids: List[str]) -> None:
        if not entity_ids:
            return
        query = (
            "UNWIND $entity_ids AS eid\n"
            f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns, id: $cid}})\n"
            f"MATCH (e:{self.ENTITY_LABEL} {{ns: $ns, id: eid}})\n"
            f"MERGE (c)-[:{self.REL_HAS_ENTITY}]->(e)"
        )
        with self.driver.session(database=self.cfg.database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    query,
                    ns=self.cfg.namespace,
                    cid=chunk_id,
                    entity_ids=entity_ids,
                )
            )

    def upsert_chunk_fact_edges(self, chunk_id: str, fact_ids: List[str]) -> None:
        if not fact_ids:
            return
        query = (
            "UNWIND $fact_ids AS fid\n"
            f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns, id: $cid}})\n"
            f"MATCH (f:{self.FACT_LABEL} {{ns: $ns, id: fid}})\n"
            f"MERGE (c)-[:{self.REL_HAS_FACT}]->(f)"
        )
        with self.driver.session(database=self.cfg.database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    query,
                    ns=self.cfg.namespace,
                    cid=chunk_id,
                    fact_ids=fact_ids,
                )
            )

    def increment_cooccur_edges(self, pairs: List[Tuple[str, str]], weight: int = 1) -> None:
        if not pairs:
            return

        rows = [{"src": s, "dst": d, "w": int(weight)} for (s, d) in pairs]
        query = (
            "UNWIND $rows AS row\n"
            f"MATCH (s:{self.ENTITY_LABEL} {{ns: $ns, id: row.src}})\n"
            f"MATCH (t:{self.ENTITY_LABEL} {{ns: $ns, id: row.dst}})\n"
            f"MERGE (s)-[r:{self.REL_CO_OCCUR}]->(t)\n"
            "ON CREATE SET r.weight = row.w\n"
            "ON MATCH SET r.weight = coalesce(r.weight, 0) + row.w"
        )
        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(rows, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, rows=b)
                )

    def decrement_cooccur_edges(self, pairs: List[Tuple[str, str]], weight: int = 1) -> None:
        if not pairs:
            return

        rows = [{"src": s, "dst": d, "w": int(weight)} for (s, d) in pairs]
        query = (
            "UNWIND $rows AS row\n"
            f"MATCH (s:{self.ENTITY_LABEL} {{ns: $ns, id: row.src}})-[r:{self.REL_CO_OCCUR}]->(t:{self.ENTITY_LABEL} {{ns: $ns, id: row.dst}})\n"
            "SET r.weight = coalesce(r.weight, 0) - row.w\n"
            "WITH r WHERE r.weight <= 0\n"
            "DELETE r"
        )
        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(rows, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, rows=b)
                )

    def upsert_synonym_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        if not edges:
            return

        rows = [{"src": s, "dst": d, "w": float(w)} for (s, d, w) in edges]
        query = (
            "UNWIND $rows AS row\n"
            f"MATCH (s:{self.ENTITY_LABEL} {{ns: $ns, id: row.src}})\n"
            f"MATCH (t:{self.ENTITY_LABEL} {{ns: $ns, id: row.dst}})\n"
            f"MERGE (s)-[r:{self.REL_SYNONYM}]->(t)\n"
            "SET r.weight = row.w"
        )
        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(rows, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(query, ns=self.cfg.namespace, rows=b)
                )

    def load_all_graph_edges(self) -> Dict[Tuple[str, str], float]:
        """Return an edge map compatible with HippoRAG.node_to_node_stats."""
        edge_map: Dict[Tuple[str, str], float] = {}

        queries = [
            (
                f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns}})-[:{self.REL_HAS_ENTITY}]->(e:{self.ENTITY_LABEL} {{ns: $ns}}) "
                "RETURN c.id AS src, e.id AS dst, 1.0 AS w"
            ),
            (
                f"MATCH (s:{self.ENTITY_LABEL} {{ns: $ns}})-[r:{self.REL_CO_OCCUR}]->(t:{self.ENTITY_LABEL} {{ns: $ns}}) "
                "RETURN s.id AS src, t.id AS dst, toFloat(r.weight) AS w"
            ),
            (
                f"MATCH (s:{self.ENTITY_LABEL} {{ns: $ns}})-[r:{self.REL_SYNONYM}]->(t:{self.ENTITY_LABEL} {{ns: $ns}}) "
                "RETURN s.id AS src, t.id AS dst, toFloat(r.weight) AS w"
            ),
        ]

        with self.driver.session(database=self.cfg.database) as session:
            for q in queries:
                records = session.execute_read(lambda tx, qq=q: list(tx.run(qq, ns=self.cfg.namespace)))
                for rec in records:
                    src = rec["src"]
                    dst = rec["dst"]
                    w = rec.get("w")
                    if src is None or dst is None:
                        continue
                    if w is None:
                        w = 1.0
                    edge_map[(src, dst)] = float(w)

        return edge_map

    def load_ent_node_to_chunk_ids(self) -> Dict[str, set]:
        query = (
            f"MATCH (c:{self.CHUNK_LABEL} {{ns: $ns}})-[:{self.REL_HAS_ENTITY}]->(e:{self.ENTITY_LABEL} {{ns: $ns}}) "
            "RETURN e.id AS eid, collect(c.id) AS cids"
        )
        out: Dict[str, set] = {}
        with self.driver.session(database=self.cfg.database) as session:
            records = session.execute_read(lambda tx: list(tx.run(query, ns=self.cfg.namespace)))
            for rec in records:
                out[rec["eid"]] = set(rec.get("cids") or [])
        return out

    def delete_chunks_and_cleanup(self, chunk_ids: List[str]) -> None:
        """Delete chunks and remove orphan facts/entities after decrementing co-occur weights.

        Expected behavior:
          - CO_OCCUR weights are decremented using each chunk's stored OpenIE triples.
          - Chunks are removed.
          - Facts with no remaining supporting chunks are removed.
          - Entities with no remaining supporting chunks are removed.
        """
        if not chunk_ids:
            return

        # Load triples before deleting chunks.
        openie_docs = self.load_openie_docs(chunk_ids)

        cooccur_pairs: List[Tuple[str, str]] = []
        for doc in openie_docs:
            for triple in doc.get("extracted_triples") or []:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                subj = _process_text_to_str(str(triple[0]))
                obj = _process_text_to_str(str(triple[2]))
                s_id = compute_mdhash_id(subj, prefix="entity-")
                o_id = compute_mdhash_id(obj, prefix="entity-")
                # Mirror the original implementation (both directions).
                cooccur_pairs.append((s_id, o_id))
                cooccur_pairs.append((o_id, s_id))

        # Decrement co-occur edges then delete chunks.
        self.decrement_cooccur_edges(cooccur_pairs, weight=1)

        with self.driver.session(database=self.cfg.database) as session:
            for batch in _batched(chunk_ids, self.cfg.batch_size):
                session.execute_write(
                    lambda tx, b=batch: tx.run(
                        f"UNWIND $ids AS id MATCH (c:{self.CHUNK_LABEL} {{ns: $ns, id: id}}) DETACH DELETE c",
                        ns=self.cfg.namespace,
                        ids=b,
                    )
                )

            # Delete orphan facts/entities (no remaining supporting chunks)
            session.execute_write(
                lambda tx: tx.run(
                    (
                        f"MATCH (f:{self.FACT_LABEL} {{ns: $ns}})\n"
                        f"WHERE NOT EXISTS {{ MATCH (:{self.CHUNK_LABEL} {{ns: $ns}})-[:{self.REL_HAS_FACT}]->(f) }}\n"
                        "DETACH DELETE f"
                    ),
                    ns=self.cfg.namespace,
                )
            )
            session.execute_write(
                lambda tx: tx.run(
                    (
                        f"MATCH (e:{self.ENTITY_LABEL} {{ns: $ns}})\n"
                        f"WHERE NOT EXISTS {{ MATCH (:{self.CHUNK_LABEL} {{ns: $ns}})-[:{self.REL_HAS_ENTITY}]->(e) }}\n"
                        "DETACH DELETE e"
                    ),
                    ns=self.cfg.namespace,
                )
            )


def clear_neo4j_namespace(
    *,
    uri: str,
    user: str,
    password: str,
    namespace: str,
    database: str = "neo4j",
    batch_size: int = 500,
) -> int:
    """Module-level convenience helper.

    This is what external callers (like standalone scripts) should import.
    """

    return Neo4jKB.clear_neo4j_namespace(
        uri=uri,
        user=user,
        password=password,
        namespace=namespace,
        database=database,
        batch_size=batch_size,
    )


class Neo4jEmbeddingStore:
    """Drop-in replacement for EmbeddingStore using Neo4j for persistence."""

    def __init__(
        self,
        embedding_model,
        kb: Neo4jKB,
        label: str,
        namespace: str,
    ):
        self.embedding_model = embedding_model
        self.kb = kb
        self.label = label
        self.namespace = namespace
        self.batch_size = kb.cfg.batch_size

        self.hash_ids: List[str] = []
        self.texts: List[str] = []
        self.embeddings: List[List[float]] = []
        self._embedding_dim: Optional[int] = None
        self.hash_id_to_idx: Dict[str, int] = {}
        self.hash_id_to_row: Dict[str, Dict[str, Any]] = {}
        self.hash_id_to_text: Dict[str, str] = {}
        self.text_to_hash_id: Dict[str, str] = {}

        self._load_data()

    def refresh(self) -> None:
        self._load_data()

    def _load_data(self) -> None:
        ids, texts, embs = self.kb.load_all_nodes(self.label)
        self.hash_ids, self.texts, self.embeddings = ids, texts, embs
        self._embedding_dim = None
        for e in self.embeddings:
            try:
                if e is not None and len(e) > 0:
                    self._embedding_dim = int(len(e))
                    break
            except Exception:
                continue
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts)
        }
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}

    def insert_strings(self, texts: List[str]):
        if not texts:
            return

        nodes_dict: Dict[str, Dict[str, str]] = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {
                "content": text
            }

        all_hash_ids = list(nodes_dict.keys())
        existing = set(self.hash_id_to_row.keys())
        missing_ids = [hid for hid in all_hash_ids if hid not in existing]

        logger.info(
            "[Neo4jEmbeddingStore:%s] inserting %d new, %d existing",
            self.namespace,
            len(missing_ids),
            len(all_hash_ids) - len(missing_ids),
        )

        if not missing_ids:
            print(
                f"[Neo4jEmbeddingStore:{self.namespace}] cache hit: {len(all_hash_ids)} records already exist"
            )
            return

        texts_to_encode = [nodes_dict[hid]["content"] for hid in missing_ids]
        print(f"[Neo4jEmbeddingStore:{self.namespace}] encoding {len(texts_to_encode)} new texts...")
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        rows = [
            {
                "id": hid,
                "content": text,
                "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
            }
            for hid, text, emb in zip(missing_ids, texts_to_encode, missing_embeddings)
        ]
        self.kb.upsert_nodes(self.label, rows)
        self._load_data()

    def delete(self, hash_ids):
        if not hash_ids:
            return
        self.kb.delete_nodes(self.label, list(hash_ids))
        self._load_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}
        return {id: self.hash_id_to_row[id] for id in hash_ids}

    def get_all_ids(self):
        return list(self.hash_ids)

    def get_all_id_to_rows(self):
        return dict(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row["content"] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        emb = self.embeddings[self.hash_id_to_idx[hash_id]]
        if emb is None or len(emb) == 0:
            raise ValueError(
                f"Missing/empty embedding for id='{hash_id}' (label={self.label}, ns={self.kb.cfg.namespace}). "
                "This usually indicates legacy or partially-written Neo4j data; re-index with force_index_from_scratch=True."
            )
        arr = np.asarray(emb, dtype=dtype)
        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1D embedding for id='{hash_id}', got shape={getattr(arr, 'shape', None)}"
            )
        return arr

    def get_embeddings(self, hash_ids, dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            dim = int(self._embedding_dim or 0)
            return np.empty((0, dim), dtype=dtype)

        # Build a strict 2D float array; fail fast on missing/ragged embeddings.
        indices = [self.hash_id_to_idx[h] for h in hash_ids]
        selected = [self.embeddings[i] for i in indices]

        dim: Optional[int] = None
        stacked: List[np.ndarray] = []
        missing: List[str] = []

        for hid, emb in zip(hash_ids, selected):
            if emb is None or len(emb) == 0:
                missing.append(hid)
                continue
            arr = np.asarray(emb, dtype=dtype)
            if arr.ndim != 1:
                raise ValueError(
                    f"Invalid embedding shape for id='{hid}' (expected 1D, got {getattr(arr, 'shape', None)})"
                )
            if dim is None:
                dim = int(arr.shape[0])
            elif int(arr.shape[0]) != dim:
                raise ValueError(
                    f"Ragged embeddings detected in Neo4jEmbeddingStore (label={self.label}, ns={self.kb.cfg.namespace}): "
                    f"expected dim={dim}, got dim={int(arr.shape[0])} for id='{hid}'."
                )
            stacked.append(arr)

        if missing:
            raise ValueError(
                f"Found {len(missing)} nodes with missing/empty embeddings in Neo4jEmbeddingStore "
                f"(label={self.label}, ns={self.kb.cfg.namespace}). Example id='{missing[0]}'. "
                "Re-index with force_index_from_scratch=True to regenerate embeddings."
            )

        if dim is None:
            return np.empty((0, 0), dtype=dtype)

        return np.stack(stacked, axis=0)
