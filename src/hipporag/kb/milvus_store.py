from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import numpy as np

from ..utils.config_utils import BaseConfig
from ..utils.misc_utils import compute_mdhash_id

logger = logging.getLogger(__name__)


def _sanitize_collection_name(name: str) -> str:
    # Milvus collection name rules are fairly strict; keep it conservative.
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not safe:
        safe = "hipporag"
    if not re.match(r"^[A-Za-z_]", safe):
        safe = "_" + safe
    return safe[:255]


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass(frozen=True)
class MilvusSearchResult:
    ids: List[str]
    scores: List[float]


class MilvusVectorStore:
    """Minimal Milvus wrapper for storing and searching float vectors by string id.

    This is intentionally small: HippoRAG uses it for chunk/passage dense retrieval.
    """

    def __init__(
        self,
        cfg: BaseConfig,
        collection_name: str,
        id_field: str = "id",
        vector_field: str = "vector",
    ) -> None:
        self.cfg = cfg
        if not self.cfg.milvus_uri:
            raise ValueError(
                "milvus_uri must be set when chunk_vector_backend='milvus'"
            )

        self.collection_name = _sanitize_collection_name(collection_name)
        self.id_field = id_field
        self.vector_field = vector_field

        self._collection = None
        self._dim: Optional[int] = None

        try:
            from pymilvus import connections

            # Keep a dedicated alias per process.
            try:
                if self.cfg.milvus_token:
                    connections.connect(
                        alias=self._alias,
                        uri=self.cfg.milvus_uri,
                        token=self.cfg.milvus_token,
                    )
                else:
                    connections.connect(
                        alias=self._alias,
                        uri=self.cfg.milvus_uri,
                    )
            except TypeError:
                # Older pymilvus may not support `token=`.
                connections.connect(
                    alias=self._alias,
                    uri=self.cfg.milvus_uri,
                )
        except ImportError as e:
            raise ImportError(
                "Milvus backend requested but pymilvus is not installed. Install 'pymilvus'."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed connecting to Milvus at {self.cfg.milvus_uri}: {e}")

    @property
    def _alias(self) -> str:
        return f"hipporag_{self.collection_name}"

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        from pymilvus import Collection

        self._collection = Collection(self.collection_name, using=self._alias)
        return self._collection

    def ensure_collection(self, dim: int) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if dim <= 0:
            raise ValueError("Vector dimension must be > 0")

        if utility.has_collection(self.collection_name, using=self._alias):
            col = Collection(self.collection_name, using=self._alias)
            # Best-effort dimension check. IMPORTANT: do not swallow dim mismatches.
            for f in getattr(col.schema, "fields", []) or []:
                if getattr(f, "name", None) != self.vector_field:
                    continue
                params = getattr(f, "params", {}) or {}
                try:
                    dim_param = params.get("dim")
                    existing_dim = int(dim_param) if dim_param is not None else None
                except Exception:
                    logger.debug(
                        "Could not parse Milvus vector dimension for '%s'; continuing",
                        self.collection_name,
                        exc_info=True,
                    )
                    existing_dim = None

                if existing_dim is not None and existing_dim != dim:
                    raise ValueError(
                        f"Milvus collection '{self.collection_name}' dim={existing_dim} does not match requested dim={dim}. "
                        "This usually means you changed embedding models without clearing/recreating the Milvus collection."
                    )
                break

            self._collection = col
            self._dim = dim
            return

        logger.info(
            "Creating Milvus collection %s (dim=%d)", self.collection_name, dim
        )

        fields = [
            FieldSchema(
                name=self.id_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=256,
            ),
            FieldSchema(
                name=self.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
        ]
        schema = CollectionSchema(fields, description="HippoRAG chunk vectors")
        col = Collection(self.collection_name, schema=schema, using=self._alias)

        index_params: Dict[str, Any] = {
            "index_type": self.cfg.milvus_index_type,
            "metric_type": self.cfg.milvus_metric_type,
            "params": self.cfg.milvus_index_params or {},
        }
        try:
            col.create_index(field_name=self.vector_field, index_params=index_params)  # type: ignore
        except Exception:
            logger.exception("Failed creating Milvus index; will rely on default")

        self._collection = col
        self._dim = dim

    def count(self) -> int:
        try:
            from pymilvus import utility

            if not utility.has_collection(self.collection_name, using=self._alias):
                return 0
            col = self._get_collection()
            return int(getattr(col, "num_entities", 0) or 0)
        except Exception:
            logger.debug("Milvus count failed", exc_info=True)
            return 0

    def upsert(self, ids: List[str], vectors: np.ndarray, batch_size: int = 512) -> None:
        if not ids:
            return

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(ids) != vectors.shape[0]:
            raise ValueError("len(ids) must match vectors rows")

        dim = int(vectors.shape[1])
        self.ensure_collection(dim)
        col = self._get_collection()

        # Load is required for searching; insert/upsert does not require load.
        # We do a best-effort load here to reduce latency for immediate searches.
        try:
            col.load()
        except Exception as e:
            logger.warning(
                "Milvus collection '%s' failed to load before upsert: %s",
                self.collection_name,
                e,
            )

        for start in range(0, len(ids), batch_size):
            id_batch = ids[start : start + batch_size]
            vec_batch = vectors[start : start + batch_size]
            vec_list = vec_batch.tolist()
            data = [id_batch, vec_list]
            try:
                if hasattr(col, "upsert"):
                    col.upsert(data)
                else:
                    # Fallback: delete then insert.
                    self.delete(list(id_batch))
                    col.insert(data)
            except Exception:
                logger.exception("Milvus upsert/insert failed")
                raise

        try:
            col.flush()
        except Exception:
            pass

    def delete(self, ids: List[str], batch_size: int = 512) -> None:
        if not ids:
            return

        from pymilvus import utility

        if not utility.has_collection(self.collection_name, using=self._alias):
            return

        col = self._get_collection()

        def _expr(batch: List[str]) -> str:
            escaped = [s.replace('\\', '\\\\').replace('"', '\\"') for s in batch]
            return f"{self.id_field} in [\"" + "\",\"".join(escaped) + "\"]"

        for b in _batched(ids, batch_size):
            try:
                col.delete(expr=_expr(list(b)))
            except Exception:
                logger.exception("Milvus delete failed")
                raise

        try:
            col.flush()
        except Exception:
            pass

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> MilvusSearchResult:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 2 and query_vector.shape[0] == 1:
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be 1D")

        dim = int(query_vector.shape[0])
        self.ensure_collection(dim)
        col = self._get_collection()

        try:
            col.load()
        except Exception as e:
            logger.warning(
                "Milvus collection '%s' failed to load before search: %s",
                self.collection_name,
                e,
            )

        params = {
            "metric_type": self.cfg.milvus_metric_type,
            "params": search_params or (self.cfg.milvus_search_params or {}),
        }

        # pymilvus expects a batch of queries
        results = cast(
            Any,
            col.search(
            data=[query_vector.tolist()],
            anns_field=self.vector_field,
            param=params,
            limit=int(top_k),
            output_fields=[],
            ),
        )

        if not results:
            return MilvusSearchResult(ids=[], scores=[])

        hits = results[0]
        ids: List[str] = []
        scores: List[float] = []
        for hit in hits:
            # hit.id is the primary key value; hit.distance is the score/distance
            ids.append(str(getattr(hit, "id", "")))
            scores.append(float(getattr(hit, "distance", 0.0)))

        return MilvusSearchResult(ids=ids, scores=scores)


class Neo4jMilvusEmbeddingStore:
    """Adapter that keeps Neo4j as the authoritative store, and mirrors new vectors to Milvus.

    For now this is used for chunk/passage embeddings:
      - content + embedding are stored in Neo4j (via the wrapped Neo4jEmbeddingStore)
      - embeddings are additionally upserted to Milvus for scalable top-k search

    Deletion is *vectors-only* to avoid breaking Neo4j cleanup logic that relies on chunk OpenIE metadata.
    """

    def __init__(
        self,
        neo4j_store: Any,
        milvus: MilvusVectorStore,
    ) -> None:
        self._neo4j = neo4j_store
        self._milvus = milvus
        self.namespace = getattr(self._neo4j, "namespace", "chunk")

    def __getattr__(self, name: str):
        # Delegate unknown attributes (e.g. text_to_hash_id/hash_id_to_row) to the
        # wrapped Neo4j store to preserve the EmbeddingStore-like interface.
        return getattr(self._neo4j, name)

    # --- passthroughs used by HippoRAG ---
    def refresh(self) -> None:
        if hasattr(self._neo4j, "refresh"):
            self._neo4j.refresh()

    def ensure_milvus_populated(self, ids: Optional[List[str]] = None, batch_size: int = 512) -> None:
        """Best-effort backfill vectors into Milvus.

        This is mainly for the case where Neo4j already contains embeddings (cache hit)
        but Milvus is empty (new backend enabled later).
        """
        if ids is None:
            ids = list(self.get_all_ids())
        if not ids:
            return

        # Best-effort: if Milvus has at least as many entities as Neo4j, assume populated.
        # If counts look low (including partial population), we attempt an idempotent upsert of all ids.
        existing_cnt = int(self._milvus.count() or 0)
        if existing_cnt >= len(ids) and existing_cnt > 0:
            return

        logger.info(
            "Milvus collection '%s' appears empty/underpopulated (have=%d need=%d); backfilling from Neo4j",
            self._milvus.collection_name,
            existing_cnt,
            len(ids),
        )

        def _stack_strict(batch_ids: List[str]) -> np.ndarray:
            vecs: List[np.ndarray] = []
            dim_local: Optional[int] = None
            missing: List[str] = []
            for hid in batch_ids:
                try:
                    v = self._neo4j.get_embedding(hid, dtype=np.float32)
                except Exception:
                    missing.append(hid)
                    continue
                if v.ndim != 1 or v.size == 0:
                    missing.append(hid)
                    continue
                if dim_local is None:
                    dim_local = int(v.shape[0])
                elif int(v.shape[0]) != dim_local:
                    raise ValueError(
                        f"Inconsistent embedding dims detected while backfilling Milvus: expected {dim_local}, got {int(v.shape[0])} for id='{hid}'"
                    )
                vecs.append(v)

            if missing:
                raise ValueError(
                    f"Cannot backfill Milvus: {len(missing)} embeddings missing/invalid in Neo4j (example id='{missing[0]}'). "
                    "Re-index with force_index_from_scratch=True to regenerate embeddings."
                )
            if not vecs:
                return np.empty((0, 0), dtype=np.float32)
            return np.stack(vecs, axis=0)

        # Ensure collection dimension is initialized using a few non-empty samples.
        try:
            sample_ids = list(ids[: min(10, len(ids))])
            sample_vecs = _stack_strict(sample_ids)
            if sample_vecs.ndim != 2 or sample_vecs.shape[0] == 0:
                raise ValueError("Could not infer embedding dimension from Neo4j (no valid sample vectors)")
            dim = int(sample_vecs.shape[1])
            self._milvus.ensure_collection(dim)
        except Exception:
            logger.exception("Failed initializing Milvus collection for backfill")
            raise

        for batch in _batched(ids, batch_size):
            batch_ids = list(batch)
            vecs = _stack_strict(batch_ids)
            self._milvus.upsert(batch_ids, vecs)

    def insert_strings(self, texts: List[str]):
        if not texts:
            return

        existing = set(getattr(self._neo4j, "hash_id_to_row", {}).keys())
        all_ids = [compute_mdhash_id(t, prefix=self.namespace + "-") for t in texts]
        missing_ids = [hid for hid in all_ids if hid not in existing]

        self._neo4j.insert_strings(texts)

        if not missing_ids:
            return

        try:
            embs = self._neo4j.get_embeddings(missing_ids)
        except Exception:
            logger.exception("Failed reading embeddings from Neo4j store")
            raise

        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2 or embs.shape[0] != len(missing_ids):
            raise ValueError(
                f"Unexpected embedding array shape from Neo4j store: shape={getattr(embs, 'shape', None)}, expected=({len(missing_ids)}, dim)"
            )

        self._milvus.upsert(missing_ids, embs)

    def delete(self, hash_ids: List[str]):
        # IMPORTANT: do NOT delete chunk nodes directly from Neo4j here.
        # Neo4j cleanup logic depends on chunk OpenIE metadata.
        self._milvus.delete(list(hash_ids))

    def get_row(self, hash_id: str):
        return self._neo4j.get_row(hash_id)

    def get_hash_id(self, text: str):
        return self._neo4j.get_hash_id(text)

    def get_rows(self, hash_ids, dtype=np.float32):
        return self._neo4j.get_rows(hash_ids, dtype=dtype)

    def get_all_ids(self):
        return self._neo4j.get_all_ids()

    def get_all_id_to_rows(self):
        return self._neo4j.get_all_id_to_rows()

    def get_all_texts(self):
        return self._neo4j.get_all_texts()

    # Helpful for callers that still want local scoring
    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self._neo4j.get_embedding(hash_id, dtype=dtype)

    def get_embeddings(self, hash_ids, dtype=np.float32):
        return self._neo4j.get_embeddings(hash_ids, dtype=dtype)
