from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Set, Tuple

import igraph as ig
import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - handled by runtime fallback
    sp = None


logger = logging.getLogger(__name__)
_EPS = 1e-12


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.where(np.isnan(arr) | (arr < 0), 0.0, arr)
    total = float(arr.sum())
    if total <= 0:
        if arr.size == 0:
            return arr
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float64)
    return arr / total


def _row_normalize_csr(mat: "sp.csr_matrix") -> "sp.csr_matrix":
    if mat.shape[0] == 0:
        return mat
    row_sums = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    inv = np.zeros_like(row_sums)
    nonzero = row_sums > 0
    inv[nonzero] = 1.0 / row_sums[nonzero]
    d_inv = sp.diags(inv, offsets=0, format="csr")
    return d_inv @ mat


@dataclass(frozen=True)
class TeleportationHybridStats:
    num_nodes: int
    num_entity_nodes: int
    num_communities: int
    num_bridge_entities: int
    num_bridge_chunks: int


class TeleportationHybridIndex:
    """Offline index + online leaky teleportation PPR runner.

    The index partitions entity nodes into communities and marks bridge entities/chunks.
    Query-time runs a sparse iterative PPR on active communities and expands active
    communities when bridge entities pass a score threshold.
    """

    def __init__(
        self,
        transition_csr: "sp.csr_matrix",
        transition_rows: np.ndarray,
        transition_cols: np.ndarray,
        transition_data: np.ndarray,
        node_communities: Tuple[frozenset[int], ...],
        bridge_entity_idxs: np.ndarray,
        bridge_chunk_idxs: np.ndarray,
        bridge_entity_to_external_communities: Dict[int, frozenset[int]],
        stats: TeleportationHybridStats,
    ):
        self.transition_csr = transition_csr
        self.transition_rows = transition_rows
        self.transition_cols = transition_cols
        self.transition_data = transition_data

        self.node_communities = node_communities
        self.bridge_entity_idxs = bridge_entity_idxs
        self.bridge_chunk_idxs = bridge_chunk_idxs
        self.bridge_entity_to_external_communities = bridge_entity_to_external_communities

        self.stats = stats
        self.num_nodes = stats.num_nodes
        self.num_communities = stats.num_communities

    @classmethod
    def build(
        cls,
        graph: ig.Graph,
        node_name_to_vertex_idx: Mapping[str, int],
        entity_node_keys: Sequence[str],
        passage_node_keys: Sequence[str],
        ent_node_to_chunk_ids: Mapping[str, Set[str]],
        bridge_betweenness_quantile: float = 0.95,
        min_bridge_entities_per_chunk: int = 2,
    ) -> "TeleportationHybridIndex":
        if sp is None:
            raise ImportError(
                "scipy is required for teleportation_hybrid mode. Install scipy first."
            )

        num_nodes = graph.vcount()
        if num_nodes <= 0:
            raise ValueError("Cannot build teleportation index from an empty graph.")

        entity_idxs = [
            node_name_to_vertex_idx[k]
            for k in entity_node_keys
            if k in node_name_to_vertex_idx
        ]
        passage_idxs = [
            node_name_to_vertex_idx[k]
            for k in passage_node_keys
            if k in node_name_to_vertex_idx
        ]

        entity_idx_set = set(entity_idxs)
        entity_idx_to_local = {idx: i for i, idx in enumerate(entity_idxs)}

        # Build entity-only projection edges (undirected, weighted).
        pair_weight: Dict[Tuple[int, int], float] = defaultdict(float)
        seen_entity_pairs: Set[Tuple[int, int]] = set()
        for edge in graph.es:
            src, dst = edge.tuple
            if src not in entity_idx_set or dst not in entity_idx_set or src == dst:
                continue

            local_u = entity_idx_to_local[src]
            local_v = entity_idx_to_local[dst]
            key = (local_u, local_v) if local_u < local_v else (local_v, local_u)

            w = edge["weight"] if "weight" in edge.attributes() else 1.0
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
            if not np.isfinite(weight) or weight <= 0:
                weight = 1.0

            pair_weight[key] += weight

            # For bridge detection, deduplicate entity edge pairs.
            key_global = (src, dst) if src < dst else (dst, src)
            seen_entity_pairs.add(key_global)

        entity_graph = ig.Graph(
            n=len(entity_idxs),
            edges=list(pair_weight.keys()),
            directed=False,
        )
        if entity_graph.ecount() > 0:
            entity_graph.es["weight"] = [pair_weight[p] for p in pair_weight.keys()]

        if entity_graph.vcount() == 0:
            raise ValueError("No entity nodes found to build teleportation index.")

        # Leiden over entity projection; fall back to multilevel.
        if entity_graph.ecount() == 0:
            entity_membership = list(range(entity_graph.vcount()))
        else:
            try:
                partition = entity_graph.community_leiden(
                    objective_function="modularity",
                    weights="weight",
                )
                entity_membership = partition.membership
            except Exception:
                logger.warning(
                    "Leiden failed; falling back to community_multilevel for partitioning.",
                    exc_info=True,
                )
                partition = entity_graph.community_multilevel(weights="weight")
                entity_membership = partition.membership

        entity_idx_to_community: Dict[int, int] = {}
        for local_idx, community_id in enumerate(entity_membership):
            entity_idx_to_community[entity_idxs[local_idx]] = int(community_id)

        num_communities = (
            int(max(entity_membership) + 1) if len(entity_membership) > 0 else 0
        )

        # Node -> communities.
        node_communities = [set() for _ in range(num_nodes)]
        for entity_idx, community_id in entity_idx_to_community.items():
            node_communities[entity_idx].add(community_id)

        for entity_key, chunk_ids in ent_node_to_chunk_ids.items():
            entity_idx = node_name_to_vertex_idx.get(entity_key)
            if entity_idx is None:
                continue
            community_id = entity_idx_to_community.get(entity_idx)
            if community_id is None:
                continue
            for chunk_key in chunk_ids:
                chunk_idx = node_name_to_vertex_idx.get(chunk_key)
                if chunk_idx is not None:
                    node_communities[chunk_idx].add(community_id)

        # Bridge entities from cross-community connectors.
        bridge_entities: Set[int] = set()
        bridge_entity_to_external: Dict[int, Set[int]] = defaultdict(set)
        for src, dst in seen_entity_pairs:
            c_src = entity_idx_to_community.get(src)
            c_dst = entity_idx_to_community.get(dst)
            if c_src is None or c_dst is None or c_src == c_dst:
                continue

            bridge_entities.add(src)
            bridge_entities.add(dst)
            bridge_entity_to_external[src].add(c_dst)
            bridge_entity_to_external[dst].add(c_src)

        # Optional high-betweenness bridge expansion.
        q = float(bridge_betweenness_quantile)
        if 0.0 < q < 1.0 and entity_graph.vcount() > 2 and entity_graph.ecount() > 0:
            try:
                raw_weights = entity_graph.es["weight"]
                distance_weights = [1.0 / max(float(w), _EPS) for w in raw_weights]
                betweenness = np.asarray(
                    entity_graph.betweenness(weights=distance_weights, directed=False),
                    dtype=np.float64,
                )
                if betweenness.size > 0:
                    threshold = float(np.quantile(betweenness, q))
                    if threshold > 0:
                        high_idxs = np.where(betweenness >= threshold)[0]
                    else:
                        high_idxs = np.where(betweenness > 0)[0]
                    for local_idx in high_idxs.tolist():
                        bridge_entities.add(entity_idxs[local_idx])
            except Exception:
                logger.warning(
                    "Betweenness bridge expansion failed; continuing with connector-only bridges.",
                    exc_info=True,
                )

        # Bridge chunks = chunks linked to multiple bridge entities.
        min_bridge_entities_per_chunk = max(1, int(min_bridge_entities_per_chunk))
        bridge_entity_keys = {
            graph.vs[idx]["name"]
            for idx in bridge_entities
            if "name" in graph.vs[idx].attributes()
        }
        chunk_bridge_count: Dict[int, int] = defaultdict(int)
        for entity_key in bridge_entity_keys:
            for chunk_key in ent_node_to_chunk_ids.get(entity_key, set()):
                chunk_idx = node_name_to_vertex_idx.get(chunk_key)
                if chunk_idx is not None:
                    chunk_bridge_count[chunk_idx] += 1

        bridge_chunk_idxs = {
            idx
            for idx, count in chunk_bridge_count.items()
            if count >= min_bridge_entities_per_chunk
        }

        # Build row-stochastic transition matrix over the full bipartite graph.
        row_idx = []
        col_idx = []
        data = []

        is_directed = graph.is_directed()
        for edge in graph.es:
            src, dst = edge.tuple
            w = edge["weight"] if "weight" in edge.attributes() else 1.0
            try:
                weight = float(w)
            except Exception:
                weight = 1.0
            if not np.isfinite(weight) or weight <= 0:
                weight = 1.0

            row_idx.append(src)
            col_idx.append(dst)
            data.append(weight)

            if not is_directed:
                row_idx.append(dst)
                col_idx.append(src)
                data.append(weight)

        if len(row_idx) == 0:
            # Degenerate graph fallback: identity transition.
            row_idx = list(range(num_nodes))
            col_idx = list(range(num_nodes))
            data = [1.0 for _ in range(num_nodes)]

        transition = sp.coo_matrix(
            (np.asarray(data, dtype=np.float64), (row_idx, col_idx)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        transition = _row_normalize_csr(transition)
        transition_coo = transition.tocoo()

        stats = TeleportationHybridStats(
            num_nodes=num_nodes,
            num_entity_nodes=len(entity_idxs),
            num_communities=num_communities,
            num_bridge_entities=len(bridge_entities),
            num_bridge_chunks=len(bridge_chunk_idxs),
        )

        logger.info(
            "Teleportation hybrid index built: nodes=%d, entities=%d, communities=%d, bridge_entities=%d, bridge_chunks=%d",
            stats.num_nodes,
            stats.num_entity_nodes,
            stats.num_communities,
            stats.num_bridge_entities,
            stats.num_bridge_chunks,
        )

        return cls(
            transition_csr=transition,
            transition_rows=np.asarray(transition_coo.row, dtype=np.int64),
            transition_cols=np.asarray(transition_coo.col, dtype=np.int64),
            transition_data=np.asarray(transition_coo.data, dtype=np.float64),
            node_communities=tuple(frozenset(x) for x in node_communities),
            bridge_entity_idxs=np.asarray(sorted(bridge_entities), dtype=np.int64),
            bridge_chunk_idxs=np.asarray(sorted(bridge_chunk_idxs), dtype=np.int64),
            bridge_entity_to_external_communities={
                int(k): frozenset(v) for k, v in bridge_entity_to_external.items()
            },
            stats=stats,
        )

    def _select_home_communities(
        self,
        reset_prob: np.ndarray,
        top_k: int,
    ) -> Set[int]:
        top_k = max(1, int(top_k))
        scores: Dict[int, float] = defaultdict(float)

        nonzero = np.flatnonzero(reset_prob > 0)
        for node_idx in nonzero.tolist():
            communities = self.node_communities[node_idx]
            if not communities:
                continue
            weight = float(reset_prob[node_idx])
            for community_id in communities:
                scores[int(community_id)] += weight

        if not scores:
            if self.num_communities <= 0:
                return set()
            return set(range(min(self.num_communities, top_k)))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return set([community_id for community_id, _ in ranked[:top_k]])

    def _build_active_node_mask(
        self,
        active_communities: Set[int],
        seed_node_idxs: np.ndarray,
    ) -> np.ndarray:
        active = np.zeros(self.num_nodes, dtype=bool)
        for node_idx, node_comm in enumerate(self.node_communities):
            if node_comm and not node_comm.isdisjoint(active_communities):
                active[node_idx] = True

        if seed_node_idxs.size > 0:
            active[seed_node_idxs] = True

        # Always keep bridge chunks active if they are already in current active region.
        if self.bridge_chunk_idxs.size > 0:
            bridge_mask = np.zeros(self.num_nodes, dtype=bool)
            bridge_mask[self.bridge_chunk_idxs] = True
            active = active | (bridge_mask & active)

        return active

    def _build_local_bridge_transitions(
        self,
        active_node_mask: np.ndarray,
    ) -> Tuple["sp.csr_matrix", "sp.csr_matrix", np.ndarray, np.ndarray]:
        row_active = active_node_mask[self.transition_rows]
        col_active = active_node_mask[self.transition_cols]

        local_mask = row_active & col_active
        bridge_mask = row_active & (~col_active)

        local = sp.coo_matrix(
            (
                self.transition_data[local_mask],
                (
                    self.transition_rows[local_mask],
                    self.transition_cols[local_mask],
                ),
            ),
            shape=(self.num_nodes, self.num_nodes),
        ).tocsr()
        bridge = sp.coo_matrix(
            (
                self.transition_data[bridge_mask],
                (
                    self.transition_rows[bridge_mask],
                    self.transition_cols[bridge_mask],
                ),
            ),
            shape=(self.num_nodes, self.num_nodes),
        ).tocsr()

        row_has_local = (np.diff(local.indptr) > 0).astype(bool)
        row_has_bridge = (np.diff(bridge.indptr) > 0).astype(bool)

        local = _row_normalize_csr(local)
        bridge = _row_normalize_csr(bridge)

        return local, bridge, row_has_local, row_has_bridge

    def _trigger_new_communities(
        self,
        rank_vector: np.ndarray,
        active_communities: Set[int],
        threshold: float,
        budget: int,
    ) -> Set[int]:
        if budget <= 0 or self.bridge_entity_idxs.size == 0:
            return set()

        candidates: Dict[int, float] = {}
        for entity_idx in self.bridge_entity_idxs.tolist():
            node_comms = self.node_communities[entity_idx]
            if not node_comms or node_comms.isdisjoint(active_communities):
                continue

            score = float(rank_vector[entity_idx])
            if score < threshold:
                continue

            for ext_comm in self.bridge_entity_to_external_communities.get(
                int(entity_idx), frozenset()
            ):
                if ext_comm in active_communities:
                    continue
                prev = candidates.get(ext_comm)
                if prev is None or score > prev:
                    candidates[ext_comm] = score

        if not candidates:
            return set()

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return set([comm for comm, _ in ranked[:budget]])

    def run(
        self,
        reset_prob: np.ndarray,
        damping: float,
        leakage_gamma: float,
        home_communities_top_k: int,
        teleport_threshold: float,
        max_teleport_steps: int,
        max_iterations: int,
        tolerance: float,
    ) -> Tuple[np.ndarray, Dict[str, int | float]]:
        if reset_prob.shape[0] != self.num_nodes:
            raise ValueError(
                f"reset_prob has shape {reset_prob.shape}, expected ({self.num_nodes},)"
            )

        reset_prob = _safe_normalize(reset_prob)

        alpha = float(np.clip(1.0 - float(damping), 0.0, 1.0))
        walk_prob = 1.0 - alpha
        gamma = float(np.clip(leakage_gamma, 0.0, 1.0))

        max_iterations = max(1, int(max_iterations))
        max_teleport_steps = max(0, int(max_teleport_steps))
        tolerance = max(float(tolerance), 0.0)
        teleport_threshold = max(float(teleport_threshold), 0.0)

        seed_node_idxs = np.flatnonzero(reset_prob > 0)
        active_communities = self._select_home_communities(
            reset_prob, top_k=home_communities_top_k
        )

        if not active_communities and self.num_communities > 0:
            active_communities = set(range(self.num_communities))

        active_node_mask = self._build_active_node_mask(active_communities, seed_node_idxs)
        local_t, bridge_t, row_has_local, row_has_bridge = (
            self._build_local_bridge_transitions(active_node_mask)
        )

        rank = reset_prob.copy()
        teleport_count = 0
        iterations = 0

        for it in range(max_iterations):
            iterations = it + 1

            has_both = row_has_local & row_has_bridge
            local_only = row_has_local & (~row_has_bridge)
            bridge_only = (~row_has_local) & row_has_bridge

            local_mix = np.zeros(self.num_nodes, dtype=np.float64)
            bridge_mix = np.zeros(self.num_nodes, dtype=np.float64)

            local_mix[has_both] = 1.0 - gamma
            bridge_mix[has_both] = gamma
            local_mix[local_only] = 1.0
            bridge_mix[bridge_only] = 1.0

            local_source = rank * local_mix
            bridge_source = rank * bridge_mix

            walk_rank = np.zeros(self.num_nodes, dtype=np.float64)
            if local_t.nnz > 0:
                walk_rank += np.asarray(local_t.transpose().dot(local_source)).ravel()
            if bridge_t.nnz > 0:
                walk_rank += np.asarray(bridge_t.transpose().dot(bridge_source)).ravel()

            # Preserve probability mass for nodes currently outside active communities.
            inactive_mask = ~active_node_mask
            if np.any(inactive_mask):
                walk_rank[inactive_mask] += rank[inactive_mask]

            # Active nodes with no eligible transitions keep their own mass.
            dangling_active = active_node_mask & (~row_has_local) & (~row_has_bridge)
            if np.any(dangling_active):
                walk_rank[dangling_active] += rank[dangling_active]

            next_rank = alpha * reset_prob + walk_prob * walk_rank
            next_rank = _safe_normalize(next_rank)

            delta = float(np.linalg.norm(next_rank - rank, ord=1))
            rank = next_rank

            new_communities = set()
            if teleport_count < max_teleport_steps:
                new_communities = self._trigger_new_communities(
                    rank_vector=rank,
                    active_communities=active_communities,
                    threshold=teleport_threshold,
                    budget=max_teleport_steps - teleport_count,
                )

            if new_communities:
                active_communities.update(new_communities)
                teleport_count += len(new_communities)
                active_node_mask = self._build_active_node_mask(
                    active_communities, seed_node_idxs
                )
                local_t, bridge_t, row_has_local, row_has_bridge = (
                    self._build_local_bridge_transitions(active_node_mask)
                )

            if delta <= tolerance and not new_communities:
                break

        run_meta: Dict[str, int | float] = {
            "iterations": int(iterations),
            "teleports": int(teleport_count),
            "active_communities": int(len(active_communities)),
            "home_communities": int(min(home_communities_top_k, len(active_communities))),
            "leakage_gamma": float(gamma),
            "teleport_threshold": float(teleport_threshold),
        }
        return rank, run_meta