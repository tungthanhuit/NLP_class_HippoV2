This document outlines the proposed **Teleportation Hybrid** architecture, a framework designed to synthesize the cognitive associative power of **HippoRAG 2** with the structural efficiency of **Fast Think-on-Graph (FastToG)**. 

The primary objective is to solve the "Global PPR Bottleneck" and "Community Boundary Loss" observed in current state-of-the-art systems, specifically targeting the **2WikiMultiHopQA** benchmark.

---

# Hybrid Graph RAG: Integrating Cognitive Memory with Community-Based Teleportation

## 1. Introduction
The current landscape of Retrieval-Augmented Generation (RAG) over Knowledge Graphs (KGs) faces a binary challenge:
1.  **HippoRAG 2** provides high-fidelity associative recall but suffers from $O(E)$ complexity due to global Personalized PageRank (PPR) calculations.
2.  **FastToG** provides high efficiency through community pruning but suffers from "Boundary Loss," where the reasoning path is prematurely severed at community intersections.

The **Teleportation Hybrid** introduces a **Bipartite Leaky PPR** mechanism. It uses structural community detection to partition the search space while allowing "semantic energy" to teleport across partitions through high-centrality bridge entities.

---

## 2. System Architecture

### 2.1 Offline Phase: Structural Memory Partitioning
The offline phase transforms a flat knowledge graph into a partitioned, community-aware memory.

**Step 1: Heterogeneous Graph Construction**
We build a bipartite graph $G = (V, E)$ consisting of:
* **Entity Nodes ($P$):** Representing specific concepts and phrases.
* **Chunk Nodes ($D$):** Representing source text passages.
* **Co-occurrence Edges:** Connecting entities found within the same chunk.
* **Synonym Edges:** Connecting entities with high embedding similarity.

**Step 2: Entity-Only Projection & Community Detection**
To keep partitions thematically coherent, we project $G$ into an entity-only graph $G'$. We apply the **Leiden Algorithm** on $G'$ to assign a `community_id` to every entity.

**Step 3: Bridge Identification**
Entities that connect two different communities or have high **Betweenness Centrality** are tagged as **Bridge Entities**. Chunks that contain multiple bridge entities are tagged as **Bridge Chunks**.



---

### 2.2 Online Phase: Threshold-Triggered Teleportation
The retrieval phase avoids global computation by dynamically loading sub-graphs as the search evolves.

**Step 1: Anchor Selection**
The user query is embedded and matched against chunks and entities. The system identifies the "Home Communities"—the partitions containing the highest density of initial seed nodes.

**Step 2: Localized Bipartite PPR**
Instead of a global matrix, the system loads a sub-matrix $M_{local}$ for the active communities. It runs a power iteration using the **Leaky Activation Formula**:

$$r^{(t+1)} = \alpha v + (1 - \alpha) [ (1 - \gamma) M_{local} + \gamma M_{bridge} ] r^{(t)}$$

* **$\gamma$ (Leakage Coefficient):** Controls the percentage of probability mass allowed to flow toward bridge nodes.

**Step 3: Teleportation Trigger**
If the activation score $r_i$ for a Bridge Entity $i$ exceeds a predefined threshold $\tau$, the system "Teleports" by dynamically loading the adjacent community's sub-matrix into the active search space.



---

## 3. Mathematical Formulation

### 3.1 Transition Matrix Separation
We decompose the full adjacency matrix into internal and external components:
* **$M_{local}$:** Contains edges where both nodes belong to the currently loaded communities.
* **$M_{bridge}$:** Contains edges where the source is in an active community but the target is in a dormant community.

### 3.2 Activation Scoring
The final score for a document (chunk) is defined by its steady-state probability mass in the PPR vector $r$. This score reflects both its direct semantic similarity to the query and its structural importance in the reasoning chain.

---

## 4. Performance Benchmarking (2WikiMultiHopQA)

This architecture is optimized to beat existing SOTA on multi-hop datasets by addressing the following metrics:

| Metric | HippoRAG 2 | FastToG | Teleportation Hybrid |
| :--- | :--- | :--- | :--- |
| **Recall@K** | High | Medium (drops at boundaries) | **High** (recovers boundary hops) |
| **Latency** | $O(E)$ - High | $O(1)$ - Very Low | **Medium-Low** (Dynamic scaling) |
| **Token Cost** | Medium | High (LLM-based pruning) | **Low** (Structural-first pruning) |
| **Multi-hop Path** | Associative | Community-based | **Cross-Community Associative** |

---

## 5. Implementation Roadmap
1.  **Graph Ingestion:** Use `neo4j-admin` to ingest the 2WikiMultiHopQA corpus.
2.  **Partitioning:** Use the `igraph` Leiden implementation for offline clustering.
3.  **Power Iteration:** Replace the standard iGraph `pagerank()` with a custom SciPy sparse matrix loop in the `HippoRAG.py` retrieval module.
4.  **Verification:** Compare retrieval accuracy on the `2WikiMultiHopQA` dev set specifically for questions requiring "Compositional Reasoning."

---

## 6. Conclusion
The **Teleportation Hybrid** represents a shift from static retrieval to **dynamic graph reasoning**. By using math (PPR) to guide the search across structural boundaries (Communities), it provides a scalable path for LLMs to reason over massive knowledge graphs without the computational overhead of global graph algorithms.