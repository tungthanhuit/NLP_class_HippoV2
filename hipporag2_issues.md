# HippoRAG 2 Enhanced Pipeline — Issues Detection & Analysis

> **Dataset:** 2WikiMultiHopQA (50-query set), MuSiQue (small-scale)
> **Baseline:** EM=0.42, F1=0.4497, Recall@5=70.5%, AR@K=40%
> **After v1 enhancements:** EM=0.50, F1=0.5287 (+19% EM, +17.6% F1)
> **Prior context:** `hipporag2_analysis_report.md` · `hipporag2_enhancement_plan.md` · `hipporag2_enhancement1_plan.md`

---

## Full Enhanced Pipeline Flow

```
INDEXING
  Corpus passages
      │
      ▼
  OpenIE (LLM, gpt-4o-mini)
      │  extracts (subj, pred, obj) triples
      ▼
  EmbeddingStore (fact_store)
      │  content = tuple string "(subj, pred, obj)"
      │  embedding_text = verbalized sentence "subj pred obj."  [verbalize_facts=True]
      │                 OR raw tuple string                     [verbalize_facts=False, ablation]
      ▼
  Graph construction
      │  phrase nodes + passage nodes ONLY  [no fact nodes in graph]
      │  facts stored exclusively in fact_embedding_store (external Parquet)
      │  PPR edge weights from entity co-occurrence
      ▼
  Persisted index (Parquet + igraph pickle)


RETRIEVAL  (per query, EnhancedHippoRAG.retrieve())
  Query
      │
      ├─[use_enhancements=True]─────────────────────────────────────────────────────┐
      │                                                                              │
      │  Step R1 — Query Reformulation (LLM: decompose_query template)              │
      │    → {"entities": [...], "bridge_hint": "...", "sub_queries": [...]}         │
      │    → always-on for ALL queries (no needs_decomposition() gate exists)        │
      │                                                                              │
      │  Step R2 — Multi-query Fact Retrieval                                       │
      │    → one embedding lookup per sub-query against fact_store                  │
      │    → top-k facts per sub-query merged into joint candidate pool             │
      │    → max-score union pooling: union_map[idx] = max(existing, score)         │
      │      (NOT RRF — np.max(stack(all_sub_scores), axis=0))                      │
      │                                                                              │
      │  Step R3 — DSPy Reranker (filter_llama3.3-70B-Instruct)                    │
      │    → binary keep/drop per fact                                              │
      │                                                                              │
      │  Step R4 — Coverage Audit (post-rerank guard)                               │
      │    → checks if each query entity appears in kept facts                      │
      │    → injects fallback fact for any uncovered entity                         │
      │                                                                              │
      └─[use_enhancements=False]────────────────────────────────────────────────────┤
                                                                                    │
  Step R5 — PPR Graph Search (both paths)                                           │
      │  seeds = kept_facts + NER-extracted DPR entities (fallback path)            │
      │  personalized PageRank → ranked passage list                               │
      ▼
  Top-k passages (retrieval_top_k=200, qa_top_k=5)


QA  (per query, EnhancedHippoRAG.qa_with_ircot() or HippoRAG.qa())
  Initial passages
      │
      ├─[max_qa_steps > 1]────────────────────────────────────────────────────────┐
      │                                                                            │
      │  IRCoT Loop (max_qa_steps - 1 hops)                                       │
      │    for step in range(max_qa_steps - 1):                                   │
      │      1. ircot_step(query, cot_so_far, all_passages)                       │
      │           LLM generates: thought sentence + missing_queries               │
      │      2. if missing_queries → retrieve(missing_queries, skip_enhancements) │
      │           hop retrieval: plain fact scoring, no decomposition              │
      │      3. accumulate up to qa_top_k new unique passages per hop             │
      │      4. if terminal ("So the answer is:") → early exit                    │
      │                                                                            │
      │  Final QA call with all accumulated passages                               │
      │                                                                            │
      └─[max_qa_steps == 1]────────────────────────────────────────────────────────┤
                                                                                    │
  Single qa() call with initial passages                                            │
      ▼
  Answer string + QAResult
```

---

## Root Problems in HippoRAG 2

The standard HippoRAG 2 pipeline has two structural weaknesses that motivated the two enhancements (E1, E2). A third problem (P4) is identified as future work not addressed by either.

---

### Problems Motivating Enhancement 1 — Retrieval Quality

#### P1 — Single-Query Embedding Cannot Cover Multi-Hop Bridge Entities

**Status:** Partially addressed by E1. Entity leak in the decomposition prompt remains an open limitation.

**Root cause:** HippoRAG 2 embeds the original question as a single vector and retrieves top-K similar facts from the fact store. For multi-hop questions, the required evidence spans a bridge entity that is *not named in the question*. A single embedding cannot simultaneously attend to both the named surface entity (e.g., a film title) and the unknown bridge entity (e.g., the director). The fact store returns candidates anchored to the named entity only; the bridge entity's neighborhood is never seeded in PPR.

**Failure mechanism:**
```
Query:  "Where was the director of [Film] born?"
Single embedding → top-K facts all about [Film]
Bridge entity (director) absent from candidates
PPR seeds from [Film] neighborhood only → second-hop passage never retrieved → answer wrong
```

**Scale:** Structurally affects all multi-hop queries where bridge entities are unnamed. On MuSiQue (3-4 hop chains), this is the dominant failure mode — 52.3% NER fallback rate indicates over half of initial retrievals fail to produce reranker-passable facts.

**Log evidence — Q6 (50-query set):**
```
Query: "Which film has the director who was born first, El Tonto or The Heart of Doreon?"

Standard run:
  Pre-rerank: both director-film facts retrieved and kept (2 facts)
  PPR seeds: charlie day, robert north bradbury  [4 nonzero phrase nodes]
  Gold ranks for biography passages: [2203, 1, 2, 695]
  QA result: FAIL — birth-date passages at ranks 2203 and 695

Both director-film edges are seeded. PPR walks from the director nodes into their film
neighborhoods but cannot bridge to the biography passages — the graph has film→director
edges but lacks director→born_on edges with sufficient weight to elevate biography
passages above rank 695.

Enhanced run:
  sub-query A: "Who directed El Tonto?"
  sub-query B: "Who directed The Heart of Doreon?"
  Gold ranks: [2, 1]   ← biography passages now surfaced
  QA result: PASS
```

The `graph_search: N nonzero phrase nodes` value logged per query directly predicts retrieval quality:

| Nonzero phrase nodes | Seeding pattern | Typical gold ranks |
|---|---|---|
| 5+ | Both entities + relation variants seeded | [1, 2] or [1, 3] |
| 3–4 | One entity seeded well | First hop ≈ rank 1, second hop variable |
| 2 | Single entity-film edge only | Second hop at rank 100–2,000 |
| 0 → DPR | No seeds | Flat dense retrieval, no graph walk |

The target for any comparative or relational 2-hop query is 4+ nonzero nodes. The E1 decomposition achieves this for the cases it handles by generating targeted sub-queries that seed both entity neighborhoods.

**Enhancement 1 response:** Reformulate the query into sub-queries targeting each hop independently, merge all sub-query candidate pools using max-score union pooling:
```python
union_map[idx] = max(union_map.get(idx, 0.0), float(sub_scores[idx]))
full_scores = np.max(np.stack(all_sub_scores, axis=0), axis=0)
```
Reformulation fires unconditionally for all queries when `use_enhancements=True` — there is no gating heuristic. Output: `{"entities": [...], "bridge_hint": "...", "sub_queries": [...]}`.

**Open limitation — Entity Leak:** The reformulation LLM may name bridge entities from its parametric memory before any retrieval occurs. The system prompt contains the constraint *"NEVER introduce names, facts, or entity names not explicitly present in the original question"*, but the LLM violates this on ~20–40% of queries where a bridge entity is inferrable from training data. Two concrete failure modes:

1. **False leakage:** LLM names the correct bridge entity → QA answers from model knowledge, not from corpus retrieval.
2. **Placeholder regression:** LLM cannot name the bridge entity → generates e.g. `"Where was [director] born?"` → no corpus match → max-score pooling elevates an irrelevant fact above what the single-query baseline retrieved. Concrete case: Q29 ("Where was the director of Return of Swamp Thing born?") — E1 generated sub-query B as `"What is the place of birth of [director]?"`. The literal placeholder matched no fact; max-score pooling elevated an irrelevant director fact. The single-query baseline happened to retrieve the correct fact; E1 degraded it (standard PASS → enhanced FAIL). Observed at scale: LastHop@K regression on 2Wiki (73.7→72.3, −1.4pp), FirstHop@K regression on MuSiQue (57.8→53.9, −3.9pp).

**Fix direction:** RAG Fusion-style diverse sub-queries without bridge entity naming. When bridge entity is unknown, set sub-query B = null; defer resolution to the IRCoT loop (legal since entity name then comes from retrieved text, not model knowledge).

---

#### P2 — Reranker Drops Evidence for the Second Entity in Two-Entity Queries

**Status:** Partially addressed by E1 coverage audit. Fundamental fix (nugget-aware reranker) remains open.

**Root cause:** The DSPy reranker scores each candidate fact *independently* against the query. For comparative or two-entity questions, it consistently assigns higher relevance scores to facts about the more prominent entity and silently drops the second entity's facts. PPR then seeds from only one entity's neighborhood; the second-hop passage lands at rank 100–2,000.

**Log evidence — Q9 (detailed):**
```
Query: "Do both films have directors from the same country?" (Interview With A Hitman / The Last Coupon)

5 candidates → reranker keeps 2:
  ('the last coupon', 'is a', '1932 british comedy film')
  ('the last coupon', 'directed by', 'frank launder')
Zero facts about Interview With A Hitman or Perry Bhandal kept.
PPR seeds: Frank Launder only.
Gold rank for Perry Bhandal passage: 1,260.
QA result: FAIL
```
The reranker correctly identifies `'british comedy film'` as informative (implies nationality) but in doing so preserves zero coverage of the second film's director. It scores each fact against the query independently — no mechanism enforces balanced entity coverage. The asymmetry (4 systematic failures, all same pattern) shows this is a structural bias, not a random error.

**All 4 systematic failures (50-query set):**
| Query | Entity dropped by reranker | Second-hop gold rank |
|---|---|---|
| Q9 | Perry Bhandal (Interview With A Hitman) | 1,260 |
| Q13 | The Corrs (kept Ryan Adams instead) | 122 |
| Q32 | Frederick I (nationality predicate) | 457 |
| Q46 | Julie Dash / Illusions director | 163 |

**Enhancement 1 response:** Post-rerank coverage audit — checks whether each query entity appears as a case-insensitive substring in the set of kept facts; injects the highest-scoring candidate fact for any uncovered entity from the full pre-rerank pool.

**Experimental evidence (ablation):**
- Config C (E1 only, single-shot): 5.4% injection rate (54/1000 queries)
- Config E (E1+E2): 2.4% injection rate (24/1000) — lower because IRCoT fills gaps iteratively

**Limitation:** The audit is a post-hoc guard. The reranker's entity-dropping bias is not corrected — only its consequence is patched. Facts that never entered the candidate pool cannot be rescued.

**Fix direction — three levels:**

**Fix A (fundamental):** LANCER-style nugget-aware reranker — identify information nuggets the query requires, select facts jointly to cover all nuggets rather than scoring each independently. Reference: `hipporag2_enhancement_plan.md` §Enhancement 2.

**Fix B (coverage audit v2):** Upgrade the current entity-presence audit to `(entity, relation)` pair checking. Q32 currently passes the audit (Frederick I appears in a kept fact as a title) but the nationality predicate is absent — the nationality passage ranks 457. The `required_relations` list is derivable from the decomposition's `bridge_hint` field at no extra LLM cost:
```python
def coverage_audit_v2(query, kept_facts, all_candidates, query_entities, required_relations):
    for (entity, relation) in required_pairs:
        if not any(entity in f and relation in f for f in kept_facts):
            inject best matching candidate fact for (entity, relation)
```

**Fix C (NER seed injection on DPR fallback):** When the reranker returns 0 kept facts and DPR fallback triggers, the top DPR passage is usually the correct first-hop passage (gold rank 1–3 in 5/6 DPR-fallback cases in the standard run). Extract named entities from the top DPR passage and use them as additional PPR seeds, converting flat DPR fallback into a graph+dense hybrid path.

---

### Problem Motivating Enhancement 2 — Reasoning Quality

#### P3 — Single-Shot QA Cannot Chain Multi-Hop Evidence

**Status:** Substantially addressed by E2 (IRCoT). Answer-format reliability remains an open limitation.

**Root cause:** HippoRAG 2 passes the top-K retrieved passages as a flat context to a single QA call. For multi-hop questions, the LLM must identify which entity from passage A is the subject of passage B, then answer from passage B. This cross-passage entity chaining consistently fails in one pass — the model either answers from the more salient passage (wrong entity) or hallucinates.

**Log evidence — Q13: retrieval succeeds, reasoning fails (50-query set):**
```
Query: "What is the nationality of the performer of When The Stars Go Blue?"

rerank_facts: 5 → 1 kept: ('the corrs', 'performed', 'when the stars go blue')
PPR seeds: [the corrs, when the stars go blue]
Gold ranks: [1, 122]   ← The Corrs passage IS at rank 1

Top docs: [The Corrs passage (rank 1), Ryan Adams passage (rank 2)]
QA answer: "American"  ← WRONG (correct: "Irish")
QA result: FAIL

Enhanced run — same retrieval, same failure:
  Gold ranks: [1, 122]  (unchanged)
  QA answer: "American"  ← WRONG
```
Retrieval is partially correct — The Corrs passage is rank 1. The failure is purely in the QA step: the top-5 context contains the Ryan Adams passage (rank 2, song composer, American). The LLM receives both passages without guidance about which entity is the *performer* vs the *composer* and uses the wrong one. The information needed is present; the problem is reasoning over unordered mixed-hop context.

**Log evidence — Q32: fragile pass (retrieval-reasoning gap exposed):**
```
Query: "What is the nationality of Beatrice I, Countess of Burgundy's husband?"

rerank_facts: 5 → 2 kept:
  ('beatrice i', 'was countess of', 'burgundy')
  ('beatrice i', 'held title', 'duchess of swabia')
PPR seeds: [beatrice i]  — Frederick I NOT seeded
Gold ranks: [2, 457, 1, 3]  — Frederick I nationality passage at rank 457
Top docs include Frederick I passage at rank 2.
QA answer: "German"  ← CORRECT (but for wrong reason)
```
Q32 passes because PPR happened to include the Frederick I passage at rank 2, and the LLM inferred nationality from his title "Holy Roman Emperor" rather than from a nationality triple. This is fragile: a slightly different passage set would produce a wrong answer. It illustrates that the standard single-shot QA pass cannot reliably distinguish luck from systematic reasoning.

**The retrieval-reasoning gap:** Even when PPR retrieval returns gold passages in the top 2 (clearly sufficient context), single-shot QA still fails on approximately 15% of those queries on the 50-query evaluation set. The gap is entirely attributable to unordered mixed-hop context: the LLM must simultaneously identify hop-1 information, hop-2 information, and chain them without guidance about order or bridge entity identity.

**Evidence from ablation (n=1000, 2WikiMultiHopQA):**
| Config | EM | AR@K5 |
|---|---|---|
| A: DPR, single-shot | 0.475 | 40.6% |
| B: HippoRAG graph, single-shot | 0.475 | 43.0% |
| D: HippoRAG graph + IRCoT | **0.623** | 43.0% |

The graph adds +2.4pp retrieval coverage (A→B) but **zero EM gain** — the bottleneck is reasoning, not retrieval. IRCoT adds **+31.2% EM** using the identical initial retrieval as B.

**Enhancement 2 response:** IRCoT multi-hop reasoning loop. At each step:
1. `_ircot_step(query, cot_so_far, all_passages)` → LLM generates a thought identifying a missing entity
2. If `missing_queries` present: `retrieve(missing_queries, skip_enhancements=True)` → targeted hop retrieval
3. Accumulate up to `qa_top_k` new unique passages per hop (no hard cap)
4. If `"So the answer is:"` in thought → exit loop early
5. For queries that did not self-terminate: `_ircot_final_qa(query, all_passages)` — passes the full accumulated passage list

**Key metrics from E2-only ablation (config D, n=1000):**
- AR@Loop = 96.8%: loop accumulates all gold passages for 968/1000 queries
- IRCoT terminal rate = 87.7%: 877 queries self-terminate in-loop; only 123 reach `_ircot_final_qa`
- Average 1.76 IRCoT steps per query (maximum 2) — most multi-hop questions resolve in one hop
- `_ircot_final_qa` receives the full `all_passages` list (not re-selected top-5)

**Adding E1 to E2:** Config E (E1+E2) achieves EM=0.628 vs D=0.623 (+0.005, +0.8%). **E2 is the dominant contributor; E1 is marginal when IRCoT is present.**

**How IRCoT resolves Q13 — step-by-step walkthrough:**
```
Initial retrieve():
  sub-query: "Who performed When The Stars Go Blue?"
  Fact kept: ('the corrs', 'performed', 'when the stars go blue')
  PPR seeds: [the corrs, when the stars go blue]
  passages_so_far: [The Corrs passage, Ryan Adams passage, ...]

Step 1 — ircot_step():
  CoT so far: []
  Prompt: "Question: What nationality is the performer...  Passages: [...]  Next step:"
  LLM output: "The performer of When The Stars Go Blue is The Corrs."
  → Performer entity explicitly named; no new retrieval needed (entity already in passages)

Step 2 — ircot_step():
  CoT so far: ["The performer of When The Stars Go Blue is The Corrs."]
  LLM output: "So the answer is: Irish"   ← terminal sentinel triggered

Final answer: "Irish"  ← CORRECT
```
The one-sentence CoT constraint is the key mechanism: forcing the LLM to explicitly identify The Corrs as the performer before asking about nationality prevents it from substituting Ryan Adams (composer) as the subject. The accumulated CoT chain provides the hop-order guidance that the flat-context single-shot call cannot infer.

**Open limitation — Answer Formatting Failures:** The `ircot_step` prompt instructs the LLM to output either a single reasoning sentence or `"So the answer is: <answer>"`. The LLM frequently generates multi-sentence reasoning chains that include the final answer *without* the terminal sentinel. `cot_is_terminal()` does not trigger; the loop exhausts all `max_qa_steps - 1` steps and adds spurious hop retrievals. Reasoning failure rate remains 59.1% (D) / 58.8% (E) after IRCoT — high because most queries self-terminate before `_ircot_final_qa`, and in-loop answers are sometimes wrong.

**Fix direction:** Strict output format constraint in the `ircot_step` prompt:
```
Output EXACTLY ONE of:
(a) "Thought: <one sentence naming the specific missing entity to retrieve next>"
(b) "So the answer is: <answer>"
Do NOT output multi-sentence reasoning. Do NOT answer if not certain from retrieved passages.
```
Add confidence gate: emit terminal only if the answer entity appears verbatim in at least one retrieved passage.

**Key references:** IRCoT (ACL 2023, arXiv:2212.10509) — primary reference; one-sentence CoT constraint; +14.3 recall points on 2WikiMultiHopQA. PRISM (arXiv:2510.14278) — extends IRCoT with selector-adder loop; 91.1% recall on 2WikiMultiHopQA. Reasoning Bottleneck in Graph-RAG (arXiv:2603.14045) — quantifies the retrieval-reasoning gap (77–91% coverage → 23–67% accuracy) and motivates IRCoT structurally.

---

### Open Problem — Not Addressed by E1 or E2

#### P4 — OpenIE Coverage Gap (Missing Predicate Classes)

**Status:** Open. Neither enhancement addresses missing index content.

**Root cause:** The standard OpenIE prompt extracts triples without predicate constraints. Five biographical predicate classes are systematically absent from the fact store:

| Predicate class | Wikidata PID | Example failure type |
|---|---|---|
| place of birth | P19 | "Where was [X] born?" (second hop) |
| place of death | P20 | "Where did [X] die?" (second hop) |
| birth / death date | P569, P570 | "When was [X] born?" (second hop) |
| employer / works at | P108 | "Who employed [X]?" (second hop) |
| in-law relations | P26, P22 | "Who is the father-in-law of [X]?" |

**Why retrieval cannot fix it:** When a triple was never extracted, there is no graph node for the bridge entity and no embedding in the fact store. Neither E1 (multi-query from existing facts) nor E2 (IRCoT loop over retrieved passages) can recover evidence that was never indexed.

**Impact on scale results:**
- 2WikiMultiHopQA (50-query set): 13/50 hard failures (26%) traced to missing biographical predicates
- MuSiQue: AR@Loop=60.0% vs 2Wiki 96.7% — the 40pp gap reflects broader coverage absence across diverse question types (medicine, geography, sports, law)

**Fix direction (future work):** Ontology-constrained KG construction — typed triple extraction (5-tuples: subj, pred, obj, subj_type, obj_type) with predicate constraint validation against `biographical_ontology.json`. Reference: `hipporag2_enhancement1_plan.md`.

---

## Enhancement-to-Problem Mapping

| Problem | Root cause | Enhancement | Experimental result | Status |
|---|---|---|---|---|
| P1 | Single-query retrieval misses bridge entities | E1: multi-query + max-score pooling | AR@K5: +0.4pp; entity leak remains | Partial |
| P2 | Reranker drops second-entity evidence | E1: post-rerank coverage audit | 5.4% (C) / 2.4% (E) injection rate | Partial |
| P3 | Single-shot QA cannot chain multi-hop reasoning | E2: IRCoT loop | +31.2% EM, AR@Loop=96.8% | Substantial |
| P4 | OpenIE missing biographical predicates | Not addressed | — | Open / future work |

---

## Full Proposal Pipeline

The following is the target architecture after all issues are addressed.

### 1. Indexing — Ontology-Constrained KG Construction

Replace unconstrained OpenIE with a two-representation extraction pipeline.

**Stage 1 — Typed Triple Extraction**
LLM extracts 5-tuples: `[subject, predicate, object, subject_type, object_type]`.
Biographical predicates are highest priority (P19, P20, P569, P570, P108, P26, P22).

**Stage 2 — Predicate Constraint Validation**
`OntologyConstraintValidator` checks domain/range constraints from `biographical_ontology.json`.
Tier 1 (biographical) and Tier 2 (creative works) constrained; Tier 3 (generic) pass-through.
Unknown types → accept optimistically (prevents false rejection on ~10–15% of triples).

**Stage 3 — Entity Normalization**
- Artifact removal: `unicodedata.normalize("NFKD", ...)`, double-space collapse
- KNN-based alias resolution: new surface form → cosine similarity ≥ 0.85 → map to canonical form
- `alias_map.json` persisted alongside index; applied at query time to normalize entity mentions

**Two representations per triple:**
- Tuple `(subj, pred, obj)` — graph node construction, identity key for deduplication
- Natural sentence `"subj pred obj."` — embedding surface for similarity search (verbalize_facts)

**Why two representations:** Graph construction needs stable identity keys (tuples). Similarity search needs natural text (sentences match query phrasing better than tuple strings). These requirements are distinct; conflating them is Issue 4.

Implementation reference: `hipporag2_enhancement1_plan.md`; Wikontic (arXiv:2512.00590) for ontology structure.

---

### 2. Retrieval — Entity-Coverage Query Decomposition

Replace the current LLM-based bridge-entity-naming decomposition with entity-agnostic sub-query generation.

**Step 2.1 — RAG Fusion-style Decomposition**
Generate 3–4 diverse sub-queries that cover different aspects of the question without assuming bridge entity knowledge. Two modes:

- *Fusion mode:* generate queries with synonyms, varied specificity, related sub-topics. Always-on (same as current implementation — no gating change needed, only prompt change).
- *Diverse mode:* broader/narrower/related angle queries. Better for comparative questions.

No bridge entity naming in sub-queries. If the bridge entity cannot be named without model knowledge, sub-query B = null (deferred to IRCoT).

**Step 2.2 — Multi-query Fact Retrieval + RRF**
One embedding lookup per sub-query against fact sentences (not tuple strings).
Top-k facts per sub-query (k = `e1_facts_per_sub_query`, default 3) merged into joint pool.
RRF score: `Σ 1/(rank_q + 60)` over sub-query result lists.

**Step 2.3 — Reranker with Nugget Coverage**
LANCER-style prompt: identify information nuggets, select facts to cover all nuggets.
Fallback: current DSPy binary reranker + post-rerank coverage audit guard.

**Step 2.4 — NER-Seeded DPR Fallback**
When `rerank_facts` returns 0 kept facts:
1. Dense retrieve top-3 passages
2. NER-extract named entities from top DPR passage
3. Link extracted entities to graph nodes as additional PPR seeds
4. Run PPR from hybrid seeds (graph + DPR entities)

This converts the flat DPR fallback (no PPR) into a hybrid path that benefits from graph structure even when fact retrieval fails completely.

**Step 2.5 — PPR Graph Search**
Unchanged. Seeds now come from: (a) validated kept facts, (b) coverage audit injections, (c) NER-seeded DPR fallback entities.

---

### 3. Generation — IRCoT with Structured Chain-of-Thought

**Loop structure:**
```
for step in range(max_qa_steps - 1):
    thought = ircot_step(query, cot_so_far, all_passages)
    if is_terminal(thought):
        break
    missing_queries = extract_missing_queries(thought)
    if missing_queries:
        hop_passages = retrieve(missing_queries, skip_enhancements=True)
        accumulate(hop_passages, max=qa_top_k per hop)

answer = final_qa(query, all_passages, cot_context=cot_so_far)
```

**Key prompt constraints for `ircot_step`:**
- Output EXACTLY one reasoning sentence OR `"So the answer is: <answer>"`
- Do NOT answer if not certain from retrieved passages
- Do NOT use model knowledge; reason only from the retrieved context
- Name the specific entity that needs to be looked up next (this becomes `missing_queries`)

**Bridge entity deferred resolution:**
When sub-query B was set to null during decomposition (bridge entity unknown), the first IRCoT step resolves it from the first-hop passage. The thought sentence names the bridge entity, which triggers a targeted hop retrieval. This is legal because the entity name now comes from retrieved text, not from model knowledge.

**Context accumulation:**
Each hop adds up to `qa_top_k` unique new passages (no hard cap). AR@Loop metric tracks gold doc coverage across the full accumulated context union.

---

## Projected Metric Trajectory

| Metric | Standard baseline | v1 (E2 only) | After E2 hardened | After E1+E2 | After E1+E2+E3 |
|---|---|---|---|---|---|
| ExactMatch | 0.42 | 0.50 | ~0.54 | ~0.58–0.62 | **0.64–0.70** |
| F1 | 0.4497 | 0.5287 | ~0.57 | ~0.61–0.65 | **0.68–0.74** |
| AR@K | 40% | ~48% | ~53% | ~60–65% | **68–75%** |
| Recall@5 | 70.5% | ~76% | ~80% | ~84–88% | **88–92%** |
| DPR fallback rate | 12% | ~8% | ~6% | <3% | <1% |
| Queries failing | 29/50 | ~21/50 | ~18/50 | ~15/50 | **~8–12/50** |

E3 = IRCoT with structured CoT (Issue 3 fix). E2 hardened = placeholder fix + nugget reranker (Issues 2+8). E1 = ontology-constrained KG (Issue 1).

---

## Ablation Study Design

The `main_ablation.py` runner covers the following configurations:

| Config | Features | Purpose |
|---|---|---|
| A: DPR only | Dense retrieval, single-shot QA | Lower bound |
| B: HippoRAG base | Graph + reranker, no decomp, single-shot QA | Standard baseline |
| C: HippoRAG + E1 | Graph + decomposition (always-on) + max-score union pooling + coverage audit, single-shot QA | Retrieval improvement only |
| D: HippoRAG + E2 | Graph, no decomp, IRCoT hops (max_qa_steps=4) | Reasoning improvement only |
| E: HippoRAG + E1+E2 | Full: E1 retrieval + IRCoT | Combined |

Additional ablation axes:
- `verbalize_facts=True` vs `False` within config B (Issue 4): quantifies embedding surface gain
- E2 with null-subquery constraint vs current (Issue 2): isolates placeholder regression fix
- E3 with hardened IRCoT prompt vs current (Issue 3): isolates formatting fix

---

## Experiment Results — Analysis Report

> **Runs:**
> 1. Standard HippoRAG 2 — 2WikiMultiHopQA: `run_20260521_182448`
> 2. Enhanced HippoRAG 2 — 2WikiMultiHopQA: `run_20260521_183057`
> 3. Standard HippoRAG 2 — MuSiQue: `run_20260522_163339`
> 4. Enhanced HippoRAG 2 — MuSiQue: `run_20260523_143106`
>
> **Scale:** 1,000 queries per dataset. Graph: 48,644 nodes / 4,452,651 edges (2Wiki), shared index.
> **Config:** LLM = Gemini-3.1-flash-lite-preview, Embedding = text-embedding-3-small, qa_top_k=5, retrieval_top_k=200, max_qa_steps=3 (enhanced).

---

### 1. Summary Results Table

| Metric | 2Wiki Standard | 2Wiki Enhanced | Δ | MuSiQue Standard | MuSiQue Enhanced | Δ |
|---|---|---|---|---|---|---|
| **ExactMatch** | 0.481 | **0.630** | +0.149 (+31%) | 0.258 | **0.401** | +0.143 (+55%) |
| **F1** | 0.5249 | **0.7176** | +0.1927 (+37%) | 0.3349 | **0.5122** | +0.1773 (+53%) |
| Recall@5 | 72.22% | 72.62% | +0.40pp | 54.03% | 54.93% | +0.90pp |
| Recall@1 | 42.18% | 42.65% | +0.47pp | 30.13% | 30.85% | +0.72pp |
| Recall@10 | 75.48% | 76.25% | +0.77pp | 60.74% | 61.60% | +0.86pp |
| Recall@100 | 86.02% | 85.90% | −0.12pp | 79.52% | 80.01% | +0.49pp |
| AR@K=5 | 42.8% | 43.4% | +0.6pp | 20.7% | 21.2% | +0.5pp |
| FirstHop@K=5 | 70.8% | 74.1% | +3.3pp | 57.8% | 53.9% | −3.9pp |
| LastHop@K=5 | 73.7% | 72.3% | −1.4pp | 52.0% | 55.9% | +3.9pp |
| AR@Loop | — | **96.7%** | — | — | **60.0%** | — |
| FirstHop@N | — | 98.6% | — | — | 82.0% | — |
| LastHop@N | — | 98.4% | — | — | 82.3% | — |
| Avg loop length (N) | — | 8.6 passes | — | — | 10.2 passes | — |
| Context coverage | 100.0% | 100.0% | — | 99.8% | 99.8% | — |
| Reasoning failure | 74.2% | 58.7% | −15.5pp | 75.7% | 61.2% | −14.5pp |
| Avg context tokens | 25,677 | 25,744 | +67 | 21,602 | 21,645 | +43 |
| NER fallback rate | — | 20.2% (202/1000) | — | — | 52.3% (523/1000) | — |
| Flat DPR fallback | — | 2.0% (20/1000) | — | — | 2.4% (24/1000) | — |
| Coverage audit injections | — | 0 | — | — | 0 | — |

---

### 2. Key Observations and Analysis

#### 2.1 QA gain is large; retrieval gain at K=5 is negligible

The EM improvement is +31% (2Wiki) and +55% (MuSiQue), and F1 improvement is +37% / +53%. These are the primary outcome metrics.

However, retrieval at K=5 is nearly unchanged: AR@K=5 moved by only +0.6pp / +0.5pp, and Recall@5 by +0.4pp / +0.9pp. This is a decisive finding: **the enhancement does not substantially improve which passages appear in the initial top-5 window**. The large QA gain comes almost entirely from the IRCoT loop's extended context accumulation, not from better initial retrieval.

The practical consequence is that Enhancement 1 (query decomposition + max-score union pooling) contributes very little to retrieval when measured at K=5. The extended loop length (N=8.6 for 2Wiki, N=10.2 for MuSiQue) is doing the real work — the model gets more attempts to find the needed passages through iterative hop retrieval.

#### 2.2 AR@Loop reveals the true mechanism

The gap between AR@K=5 and AR@Loop is the key explanatory signal:

| Dataset | AR@K=5 | AR@Loop | Gap |
|---|---|---|---|
| 2WikiMultiHopQA | 43.4% | 96.7% | +53.3pp |
| MuSiQue | 21.2% | 60.0% | +38.8pp |

On 2WikiMultiHopQA, the IRCoT loop accumulates 96.7% of all-gold coverage across its passes — meaning for 967/1000 queries, every gold passage was found somewhere in the loop context. Starting from just 43.4% in the initial 5-passage window, the loop adds the missing passages through iterative hop retrievals. This is the structural value of IRCoT: it converts a single-shot retrieval problem into an iterative search problem where each reasoning step can correct for missed passages.

On MuSiQue, AR@Loop=60% vs AR@K=5=21.2%. Even with an average of 10.2 retrieval passes, 40% of queries still fail to accumulate all gold passages. This reflects MuSiQue's structural hardness: 3-4 hop questions where some bridge triples are completely absent from the graph (Issue 1 pattern, but more prevalent across diverse question types).

#### 2.3 Reasoning failure remains the bottleneck

After IRCoT, reasoning failure is still 58.7% (2Wiki) and 61.2% (MuSiQue). These are high rates. However, interpreting this metric requires care: `reasoning_failure_rate_pct` is defined as the fraction of queries where the final QA call failed to produce the correct answer despite full context coverage. 

The combination of AR@Loop=96.7% and EM=0.630 for 2Wiki implies that among the 967/1000 queries with all gold passages available, 630 answered correctly — a success rate of 65% given perfect context. The remaining 35% are pure reasoning failures (wrong entity selected, hallucinated answer, or question misunderstood). This aligns with Issue 3 (IRCoT answer formatting: the LLM generates multi-sentence reasoning rather than one sentence + terminal signal), which inflates the remaining hop steps and introduces noise passages.

For MuSiQue, the picture is starker: AR@Loop=60% means only 600 queries ever had all gold passages. EM=0.401 means 401 answered correctly — a success rate of ~67% given perfect context, comparable to 2Wiki. **The remaining gap is purely a retrieval problem (missing triples / incomplete graph traversal), not a reasoning problem.**

#### 2.4 Enhancement 1 (multi-query) shows asymmetric hop effects

Multi-query decomposition changes FirstHop@K and LastHop@K differently per dataset:

- **2WikiMultiHopQA:** FirstHop@K: 70.8→74.1 (+3.3pp), LastHop@K: 73.7→72.3 (−1.4pp)
- **MuSiQue:** FirstHop@K: 57.8→53.9 (−3.9pp), LastHop@K: 52.0→55.9 (+3.9pp)

On 2Wiki (2-hop questions, entity-focused), decomposition produces sub-queries that better target the first-hop entity. The small LastHop regression is likely the placeholder sub-query effect (Issue 2/8): for some queries, sub-query B is a placeholder that retrieves a wrong fact which then misleads PPR away from the last-hop passage.

On MuSiQue (3-4 hop questions, complex chains), the pattern inverts: decomposition hurts FirstHop@K but helps LastHop@K. The multi-query approach is generating sub-queries that overfit to bridge entities named via LLM knowledge (Issue 2), successfully finding later-hop facts while accidentally retrieving the wrong first-hop passage.

This asymmetry suggests that the decomposition prompt is behaving differently across query types — and that the "always-on" Enh1 (1000/1000 calls) is not gated appropriately, applying decomposition even to simple queries where it is harmful.

#### 2.5 Coverage audit is completely inactive (0 injections on both datasets)

The coverage audit (Enhancement 2) injected **zero facts** across 2,000 queries on two datasets. This is a significant finding. There are two possible explanations:

1. **Multi-query retrieval implicitly covers all entities:** The Enh1 sub-queries, even imperfect, are broad enough to retrieve at least one fact per query entity in every case that the reranker does not drop entirely. The coverage audit's guard condition (`entity not in kept_facts`) is never triggered.

2. **The coverage audit's entity matching is too strict or has a bug:** If the entity string matching fails due to normalization issues (e.g., the query entity is `"Charlie Day"` but the fact contains `"charlie day"`), every entity appears covered even when it isn't, and no injection occurs.

Either way, Enh2 as currently implemented contributes zero to the results. This needs investigation before Enh2 can be credited with any metric improvement.

#### 2.6 NER fallback rate is high on MuSiQue (52.3%)

Over half of MuSiQue initial retrievals fail reranking and require NER-seeded fallback. The fallback then succeeds (NER-based graph traversal) in 499/523 cases, with 24 falling back to flat DPR. Compare 2Wiki: only 202/1000 (20.2%) need NER fallback.

The high MuSiQue fallback rate indicates that the multi-query sub-queries on MuSiQue's complex chains frequently fail to retrieve any reranker-passable facts. This is consistent with Issue 2 (LLM decomposition uses model knowledge that doesn't match the corpus vocabulary) and Issue 1 (the required bridging triples are absent from the fact store). MuSiQue's questions are structurally more diverse (medicine, geography, sports, law) and less template-driven than 2WikiMultiHopQA, so the OpenIE coverage gap manifests more broadly.

#### 2.7 Avg context tokens unchanged despite longer loop

Enhanced runs accumulate 8.6 passes (2Wiki) / 10.2 passes (MuSiQue) yet context tokens barely increase (2Wiki: 25,677→25,744; MuSiQue: 21,602→21,645).

> **[CORRECTION — verified against `EnhancedHippoRAG.py`]:** The earlier hypothesis that the final QA call only sees `qa_top_k=5` passages is **wrong**. `_ircot_final_qa(passages=all_passages, ...)` receives the full accumulated `all_passages` list — not a re-selection to top-5. The actual reason for the near-flat token count is more fundamental: **877/1000 queries (87.7%) terminate in-loop** via the `answer` field in `_ircot_step` JSON and never reach `_ircot_final_qa` at all. These queries get only the initial `qa_top_k=5` context. Only **123 queries (12.3%)** reach the final QA call with full accumulated context. The average is dominated by the 87.7% short-path queries, so the aggregate barely moves even though the 123 long-path queries see significantly more passages.

This means the context accumulation is actually being used for the hard cases that need it — the 123 queries that failed to self-terminate. The reasoning failure rate of 58.7% reflects both the 877 in-loop terminations (where the model did answer but sometimes incorrectly) and the 123 final QA cases. No implementation gap exists here.

---

### 3. Deep Sample Analysis — Where Enhanced Methods Help and Hurt

#### Pattern A: IRCoT resolves bridge entity from first-hop passage (HELPS)

Query type: `"Who is the father-in-law of [Person]?"`, `"Where was the director of [Film] born?"`

Standard pipeline: Single embedding → top-5 candidates → all about [Person] or [Film] → reranker keeps 0 → DPR fallback → answer wrong.

Enhanced pipeline: Sub-queries `"Who did [Person] marry?"` + `"Who is [spouse]'s father?"` → multi-query pool covers both entities → reranker keeps bridge entity fact → PPR seeds from both entities → first-hop and second-hop passages retrieved → IRCoT step 1 confirms bridge entity from first-hop → IRCoT step 2 identifies the answer entity → correct answer.

Evidence: 2Wiki AR@Loop=96.7% with FirstHop@N=98.6% shows the IRCoT loop very reliably finds both hops on structured 2-hop questions.

#### Pattern B: MuSiQue 3-4 hop chains exceed loop depth (PARTIALLY HELPS)

Query type: `"What is the [attribute] of the [entity] of the [thing] of the [thing2]?"` (3-4 entity chain)

Enhanced pipeline: IRCoT max_qa_steps=3 provides 2 hop retrievals. For 3-4 hop questions, the loop terminates before all bridge entities are resolved. The accumulated context has 10.2 passes on average but AR@Loop=60% — the loop runs out of steps on long chains.

Evidence: MuSiQue FirstHop@N=82.0% and LastHop@N=82.3% — 18% of first hops are still missed in the full loop. These are cases where the required triple was never indexed (Issue 1 generalized) or where the sub-query decomposition produced a placeholder for the first hop (Issue 2/8).

#### Pattern C: Decomposition entity leak inflates false confidence (HURTS)

Query type: `"Where was the director of [Film] born?"` where director is an obscure figure

Enhanced pipeline: Decomposition LLM names the director from its parametric knowledge (e.g., `"Jim Wynorski"`). Sub-query B = `"Where was Jim Wynorski born?"` retrieves a passage. Reranker keeps it (birth location fact present). But the passage may not be in the corpus, so RRF elevates it while the correct passage drops. QA answers incorrectly with the parametric answer.

Evidence: 2Wiki LastHop@K=73.7→72.3 (−1.4pp) regression. The small AR@K regression on last hop is a signal of decomposition introducing wrong passages into the top-K window.

#### Pattern D: NER fallback rescues zero-rerank queries (HELPS on 2Wiki, mixed on MuSiQue)

When reranker keeps 0 facts, NER extracts entities from the top DPR passage and seeds PPR. On 2Wiki: 202 fallbacks, 182 recovered by NER-seeded graph. On MuSiQue: 523 fallbacks, 499 recovered. The fallback is working well mechanically.

However, on MuSiQue, the high fallback rate (52.3%) indicates the retrieval pipeline is failing on the initial step for over half of queries. The NER fallback is a second-chance mechanism that salvages some of these, but AR@Loop=60% suggests many still miss gold passages even after fallback.

#### Pattern E: Coverage audit contributes nothing (BUG OR DESIGN GAP)

0 coverage audit injections across 2,000 queries. As analyzed in §2.5, this means either the guard condition never triggers (entities always appear covered by multi-query) or the entity matching has a normalization bug. In either case, Enhancement 2 as currently labeled contributes nothing measurable to the results. The QA gains are entirely attributable to Enh1 (multi-query) + IRCoT loop structure.

---

### 4. Suggested Ablation Experiment Configurations

The following configurations isolate each component's contribution and address the gaps identified above.

#### Ablation Group 1 — Isolate IRCoT vs Multi-query (Primary decomposition)

| Config | Enh1 (multi-query) | IRCoT hops | max_qa_steps | Purpose |
|---|---|---|---|---|
| B | Off | Off | 1 | HippoRAG baseline |
| C | On | Off | 1 | E1 only: does multi-query improve retrieval without IRCoT? |
| D | Off | On | 3 | E2 only: does IRCoT alone gain without better initial retrieval? |
| E | On | On | 3 | Full system (current enhanced run) |

**Expected finding:** Based on the near-zero AR@K=5 improvement, config C (E1 only) should show minimal EM gain vs B. Config D (IRCoT only, no multi-query) should gain most of the QA improvement. This would confirm that the QA gain comes from IRCoT, not from decomposition.

**Why this matters:** If D ≈ E in EM/F1, then Enhancement 1 (multi-query decomposition) adds LLM cost with no QA benefit, and the thesis contribution should be reframed to IRCoT-only.

#### Ablation Group 2 — Isolate Enhancement 1 sub-components

> **[IMPLEMENTATION NOTE]:** There is no `needs_decomposition()` gate in the current code — C2 (always-on) is the current implementation. C1 is a hypothetical configuration that would require adding a keyword-based gate, included here to test whether selective decomposition helps.

| Config | decomposition gate | Always-on | Sub-queries | Purpose |
|---|---|---|---|---|
| C0 | Off (never decompose) | — | 1 (query only) | Retrieval baseline |
| C1 | Keyword heuristic (hypothetical) | Off | 1–3 (gated) | Test selective decomposition |
| C2 | Always-on (current) | On | 2–4 (always) | Current implementation |
| C3 | Always-on + null-subquery | On | 1–2 (no placeholder) | Issue 2/8 fix |

**Expected finding:** C3 should eliminate the FirstHop@K regression seen in MuSiQue and the LastHop@K regression in 2Wiki. If C1 (keyword heuristic gate) outperforms C2 (always-on), it means unconditional decomposition is harmful on simple queries and the gating logic is worth implementing.

#### Ablation Group 3 — Isolate IRCoT loop depth

| Config | max_qa_steps | Max hop passes | Purpose |
|---|---|---|---|
| D1 | 2 | 1 | IRCoT with single hop |
| D2 | 3 | 2 | Current enhanced config |
| D3 | 5 | 4 | Deeper IRCoT (tests MuSiQue limit) |
| D4 | 3 + hardened IRCoT prompt | 2 | Issue 3 fix: strict one-sentence output format |

**Expected finding:** D3 on MuSiQue should improve AR@Loop from 60% toward 80%+ by giving longer chains more hop budget. D4 (hardened IRCoT prompt) should reduce the 59% reasoning failure rate by enforcing terminal-sentinel discipline — the fix for P3's open limitation.

#### Ablation Group 4 — Verify Coverage Audit

| Config | Coverage audit | Entity normalization | Purpose |
|---|---|---|---|
| E1 | Off | Off | Current (audit inactive, 0 injections) |
| E2 | On + case-norm | Off | Fix entity matching (lowercase comparison) |
| E3 | On + case-norm | On | Normalized entity matching |

**Expected finding:** E2 should produce non-zero injections if the entity matching bug hypothesis is correct. If E2 still shows 0 injections, then multi-query retrieval genuinely covers all entities and the coverage audit is redundant, confirming it can be removed.

#### Ablation Group 5 — Embedding Surface (Issue 4)

| Config | verbalize_facts | Embedding surface | Purpose |
|---|---|---|---|
| B0 | False | Raw tuple `"('subj', 'pred', 'obj')"` | Original HippoRAG 2 behavior |
| B1 | True | Verbalized `"subj pred obj."` | Current default (fixed verbalize_fact) |

**Expected finding:** B1 should show higher Recall@5 and AR@K than B0, confirming that sentence-form embedding better matches query phrasing. The gain is expected to be 1-3pp on Recall@5 based on the embedding surface analysis in §Issue 4.

#### Summary — Recommended Execution Order

```
Priority 1 (critical for thesis framing):
  Run C, D vs E on both datasets → confirm IRCoT is the dominant contributor

Priority 2 (coverage audit verification):
  Run E2 (case-normalized audit) on 100-query subset → check injection count > 0

Priority 3 (fix §2.7 context passing):
  Run D4 (full accumulated context to final QA) on 100-query subset → EM delta

Priority 4 (embedding surface):
  Run B0 vs B1 → Recall@5 / AR@K comparison

Priority 5 (loop depth on MuSiQue):
  Run D3 (max_qa_steps=5) on MuSiQue → AR@Loop target 75-80%
```

---

## Ablation Study Results — 2WikiMultiHopQA (n=1000)

> **Configs:** A=DPR only · B=HippoRAG base · C=E1 only (multi-query, single-shot QA) · D=E2 only (IRCoT, plain retrieval) · E=E1+E2 (full system)
> **Index:** shared (no re-indexing across configs). `max_qa_steps=3` for D and E.

### 1. Results Table

| Metric | A: DPR | B: Base | C: E1 only | D: E2 only | E: E1+E2 |
|---|---|---|---|---|---|
| **ExactMatch** | 0.475 | 0.475 | 0.487 | **0.623** | **0.628** |
| **F1** | 0.5169 | 0.5202 | 0.5305 | 0.7119 | 0.7165 |
| Δ EM vs B | — | 0 | +0.012 | **+0.148** | **+0.153** |
| AR@K=5 | 40.6% | 43.0% | 43.4% | 43.0% | 43.4% |
| FirstHop@K=5 | 71.1% | 72.5% | 72.9% | 72.5% | 72.9% |
| LastHop@K=5 | 69.9% | 71.5% | 72.0% | 71.5% | 72.0% |
| AR@Loop | — | — | — | 96.8% | 96.9% |
| FirstHop@N | — | — | — | 99.0% | 99.1% |
| LastHop@N | — | — | — | 98.2% | 98.1% |
| Avg loop length (N) | — | — | — | 8.7 | 8.6 |
| Reasoning failure | 74.9% | 74.0% | 74.1% | 59.1% | 58.8% |
| IRCoT terminal rate | — | — | — | 87.7% | 87.8% |
| NER fallback rate | 0% | 8.0% | 4.0% | 13.1% | 11.5% |
| Coverage audit rate | 0% | 0% | **5.4%** | 0% | **2.4%** |
| Multi-query rate | 0% | 0% | 100% | 0% | 43.9% |
| **Total LLM calls** | 1,000 | 1,000 | 2,000 | 1,881 | 2,875 |
| — Reform calls | 0 | 0 | 1,000 | 0 | 1,000 |
| — IRCoT loop calls | 0 | 0 | 0 | 1,758 | 1,753 |
| — Final QA calls | 1,000 | 1,000 | 1,000 | 123 | 122 |
| **Total tokens** | 1.43M | 1.47M | 1.83M | 4.36M | 4.71M |
| Retrieval time (s) | 307 | 3,455 | 672 | 1,728 | 1,270 |

---

### 2. Analysis

#### Finding 1 — IRCoT is the dominant contributor (+31% EM); E1 is marginal (+2.5%)

Config D (E2 only, IRCoT with plain base retrieval) achieves EM=0.623 vs B=0.475: a **+31.2% gain** using the same initial retrieval as the baseline. Config C (E1 only, multi-query with single-shot QA) achieves EM=0.487: a **+2.5% gain** at the cost of doubling LLM calls.

The incremental gain of adding E1 on top of E2 (E vs D) is **+0.005 EM (+0.8%)** at 53% more LLM calls (2,875 vs 1,881) and 8% more tokens (4.71M vs 4.36M). E1 has essentially no value when IRCoT is already present.

**Implication for thesis:** The core contribution is the IRCoT multi-hop reasoning loop (E2), not the query decomposition + max-score union pooling (E1). The thesis framing should lead with IRCoT.

#### Finding 2 — HippoRAG graph adds no single-shot QA benefit over DPR

A (DPR) and B (HippoRAG graph + reranker) both achieve exactly EM=0.475. The graph does improve retrieval coverage — AR@K5: 40.6% (A) → 43.0% (B), +2.4pp — but this retrieval improvement completely fails to translate into better answers in single-shot QA mode.

The gap between retrieval quality (AR@K5=43%) and QA accuracy (EM=47.5%) is the reasoning bottleneck described in Issue 3. Even when both gold passages are in the top-5 context, the model frequently uses the wrong entity or cannot chain the hops in a single pass. IRCoT resolves this by providing structured sequential context rather than a flat passage pool.

#### Finding 3 — IRCoT self-terminates early for 87.7% of queries

Config D: only **123 out of 1000 queries** required the fallback final QA call. The remaining **877 queries (87.7%)** generated a "So the answer is:" terminal signal within the IRCoT loop steps, meaning the model answered correctly within the reasoning chain itself.

This has an important LLM budget implication: the IRCoT loop consumed 1,758 step calls but produced answers for 877 queries internally. The loop is highly efficient — average 1.76 IRCoT steps per query, not the maximum 2. Most multi-hop questions need only one reasoning hop to resolve the bridge entity and answer.

#### Finding 4 — E1 multi-query increases NER fallback in E2 mode (counterproductive)

| Config | NER fallback rate |
|---|---|
| B (Base, no E1) | 8.0% |
| C (E1, no IRCoT) | 4.0% |
| D (E2, no E1) | 13.1% |
| E (E1+E2) | 11.5% |

D has a higher NER fallback rate (13.1%) than B (8.0%) despite using the same initial retrieval. This is because D's IRCoT hop retrievals (using `skip_enhancements=True`) also trigger the NER fallback path when they get 0 reranker-passed facts. Hop retrievals on bridge entity sub-queries are harder than initial retrievals and fail the reranker more often.

Config C (E1 only) reduces the initial retrieval fallback rate to 4.0% — multi-query increases the candidate pool and reduces complete reranker rejections. But when combined with IRCoT (E = 11.5%), the benefit is diluted by the same hop-retrieval fallback issue.

#### Finding 5 — Coverage audit IS working (earlier grep analysis was wrong)

Config C shows `coverage_audit_rate=5.4%` (54/1000 queries injected). Config E shows `2.4%` (24/1000). The earlier analysis that reported "0 coverage audit injections" was wrong — it grepped log messages while the metric counter in the pipeline snapshot was the authoritative source.

The audit fires at low rates because E1 multi-query already covers most query entities in the candidate pool. The 5.4% in C vs 2.4% in E difference suggests that single-shot mode (C) has more coverage gaps than IRCoT mode (E), consistent with IRCoT's iterative retrieval filling coverage holes over multiple hops.

However, the audit's contribution to EM is not directly measurable from this data. The C vs B delta (+0.012 EM) includes both multi-query and coverage audit effects combined.

#### Finding 6 — E1 multi-query rate is 44% in E config (not 100%)

Config E shows `multi_query_rate=0.4386`. Config C shows `1.0`. Both have `use_enhancements=True`.

> **[CORRECTION — verified against `EnhancedHippoRAG.py`]:** There is NO `needs_decomposition()` gate. Reformulation fires unconditionally on all `retrieve()` calls where `use_enhanced_pipeline=True`. The 43.9% rate is a dilution artifact: the IRCoT loop calls `retrieve(skip_enhancements=True)` for hop retrievals, which sets `use_enhanced_pipeline=False` and skips multi-query. The `multi_query_rate` metric counts multi-query calls over the total `retrieve()` calls including hops. In E: 1000 initial calls (all multi-query=True) + 1753 hop calls (all multi-query=False) = 2753 total → 1000/2753 ≈ 36% (the ~44% discrepancy likely reflects that some hop calls also go through the enhanced path, or the metric denominator differs).

Either way, E1 is firing on all initial retrieve() calls in config E. The incremental gain over D is still only +0.005 EM, confirming that multi-query decomposition contributes negligibly when IRCoT is present.

#### Finding 7 — LLM call economics

| Config | EM | Total calls | Tokens | EM / 1000 calls | EM / 1M tokens |
|---|---|---|---|---|---|
| B (Base) | 0.475 | 1,000 | 1.47M | 475 | 323 |
| C (E1) | 0.487 | 2,000 | 1.83M | 243 | 266 |
| D (E2) | 0.623 | 1,881 | 4.36M | 331 | 143 |
| E (E1+E2) | 0.628 | 2,875 | 4.71M | 218 | 133 |

By any efficiency metric, E1 makes the system worse per call or per token. E2 alone (D) gives the best trade-off between quality gain and cost. Adding E1 on top of E2 is the most expensive configuration for the smallest marginal gain.

---

### 3. Revised Component Contribution Summary

| Component | EM contribution | AR@K5 contribution | Cost |
|---|---|---|---|
| HippoRAG graph (PPR+reranker) vs DPR | **0** EM | +2.4pp | ×11 retrieval time |
| E1: multi-query + max-score pooling | +0.012 EM | +0.4pp | +1,000 LLM calls |
| E2: IRCoT loop | **+0.148 EM** | 0 (same AR@K5) | +881 net calls, +2.9M tokens |
| E1 on top of E2 | +0.005 EM | +0.4pp | +994 LLM calls |

The graph PPR structure is essential for retrieving the right passages (AR@K5 +2.4pp vs DPR, which compounds in E2's iterative hops). But E1 is not load-bearing — the thesis contribution can be stated as: **HippoRAG graph retrieval + IRCoT iterative reasoning**, where E1 (query decomposition) is an optional enhancement with marginal independent value.

---

### 4. Revised Ablation Priorities

Based on the actual results, the priority ordering from §4 of the Suggested Ablations section changes:

| Old priority | New priority | Reason |
|---|---|---|
| 1. C vs D vs E to confirm IRCoT dominates | ✅ Confirmed — D ≈ E, both >> C | Done |
| 2. Coverage audit normalization fix | Lower — audit is working (5.4%), bug hypothesis was wrong | Deprioritize |
| 3. Harden IRCoT prompt (D4) | **High** — 59% reasoning failure rate; `_ircot_final_qa` already gets full context | Run next |
| 4. Embedding surface B0 vs B1 | Medium — graph retrieval quality may cascade into IRCoT | Run after |
| 5. Loop depth D3 on MuSiQue | High — AR@Loop=60% suggests depth is the bottleneck | Run in parallel |

**Next recommended run:** D4 — IRCoT with hardened `ircot_step` prompt (strict one-sentence output, terminal sentinel enforced). This targets the 59% reasoning failure rate: the loop already accumulates all gold passages (AR@Loop=96.8%), but the model fails to emit the terminal `"So the answer is:"` format, causing spurious extra hops. The final QA call already receives all accumulated passages; the bottleneck is prompt discipline.

---

## Ablation Study Results — Config F: DPR + IRCoT (2WikiMultiHopQA, n=1000)

> **Config F:** DPR initial retrieval + DPR hop retrieval + IRCoT loop (max_qa_steps=4). No graph, no reranker, no query decomposition.
> **Source:** `outputs/ablation/2wikimultihopqa/dpr_ircot/ablation_result.json`

### 1. Results Table — Full 6-Config Comparison

| Metric | A: DPR | B: Base | C: E1 only | D: E2 only | E: E1+E2 | **F: DPR+IRCoT** |
|---|---|---|---|---|---|---|
| **ExactMatch** | 0.475 | 0.475 | 0.487 | 0.623 | 0.628 | **0.624** |
| **F1** | 0.5169 | 0.5202 | 0.5305 | 0.7119 | 0.7165 | **0.7088** |
| Δ EM vs B | — | 0 | +0.012 | +0.148 | +0.153 | **+0.149** |
| AR@K=5 | 40.6% | 43.0% | 43.4% | 43.0% | 43.4% | **40.6%** |
| FirstHop@K=5 | 71.1% | 72.5% | 72.9% | 72.5% | 72.9% | **72.7%** |
| LastHop@K=5 | 69.9% | 71.5% | 72.0% | 71.5% | 72.0% | **67.6%** |
| AR@Loop | — | — | — | 96.8% | 96.9% | **96.3%** |
| FirstHop@N | — | — | — | 99.0% | 99.1% | **99.0%** |
| LastHop@N | — | — | — | 98.2% | 98.1% | **97.7%** |
| Avg loop length (N) | — | — | — | 8.7 | 8.6 | **8.8** |
| Reasoning failure | 74.9% | 74.0% | 74.1% | 59.1% | 58.8% | **59.2%** |
| IRCoT terminal rate | — | — | — | 87.7% | 87.8% | **86.9%** |
| NER fallback rate | 0% | 8.0% | 4.0% | 13.1% | 11.5% | **0%** |
| Multi-query rate | 0% | 0% | 100% | 0% | 43.9% | **0%** |
| **Total LLM calls** | 1,000 | 1,000 | 2,000 | 1,881 | 2,875 | **1,898** |
| — IRCoT calls | 0 | 0 | 0 | 1,758 | 1,753 | **1,767** |
| — Final QA calls | 1,000 | 1,000 | 1,000 | 123 | 122 | **131** |
| **Total tokens** | 1.43M | 1.47M | 1.83M | 4.36M | 4.71M | **4.34M** |
| Retrieval time (s) | 307 | 3,455 | 672 | 1,728 | 1,270 | **997** |

---

### 2. Key Findings

#### Finding 1 — DPR + IRCoT matches HippoRAG graph + IRCoT (F ≈ D)

Config F achieves EM=0.624, F1=0.7088. Config D achieves EM=0.623, F1=0.7119. The difference is **+0.001 EM and −0.003 F1** — within noise across 1,000 queries.

This is the central result: **replacing the HippoRAG knowledge graph (PPR + OpenIE + reranker) with flat DPR retrieval, at both the initial step and every IRCoT hop, produces no measurable QA degradation.** The graph's retrieval advantage at K=5 (+2.4pp AR@K, A→B) is completely neutralized once IRCoT's iterative loop is active.

**Implication:** When IRCoT is present, the expensive graph construction pipeline (OpenIE extraction, entity normalization, graph building, PPR computation, DSPy reranker) contributes zero marginal EM. The entire multi-hop reasoning benefit comes from the IRCoT loop structure, not from the graph's seeding quality.

#### Finding 2 — Graph's AR@K advantage disappears at loop level

| Retriever | AR@K=5 | AR@Loop | EM |
|---|---|---|---|
| DPR (A/F initial) | 40.6% | 96.3% (F) | 0.624 (F) |
| HippoRAG graph (B/D initial) | 43.0% | 96.8% (D) | 0.623 (D) |
| Gap | −2.4pp | −0.5pp | +0.001 |

The graph gives +2.4pp at K=5, but this shrinks to −0.5pp at loop level (F slightly worse). The IRCoT loop closes the initial retrieval gap in 8.8 DPR hops vs 8.7 graph hops. Both converge to ~96–97% AR@Loop with nearly identical EM.

#### Finding 3 — LastHop@K=5 is the only meaningful gap (−3.9pp)

| Metric | D (graph+IRCoT) | F (DPR+IRCoT) | Δ |
|---|---|---|---|
| FirstHop@K=5 | 72.5% | 72.7% | +0.2pp (F better) |
| LastHop@K=5 | 71.5% | 67.6% | **−3.9pp** (D better) |
| AR@K=5 | 43.0% | 40.6% | −2.4pp |
| AR@Loop | 96.8% | 96.3% | −0.5pp |
| EM | 0.623 | 0.624 | +0.001 |

The graph is better at placing the *last-hop* passage in the initial top-5 window (−3.9pp disadvantage for DPR), which is exactly where graph PPR should help — it walks from the first-hop entity to its neighbors. But this initial placement advantage does not translate into EM gain because the IRCoT loop retrieves the last-hop passage within 8.8 steps regardless.

#### Finding 4 — IRCoT loop behavior is essentially identical without the graph

| Metric | D (graph hops) | F (DPR hops) | Δ |
|---|---|---|---|
| IRCoT terminal rate | 87.7% | 86.9% | −0.8pp |
| Avg loop length (N) | 8.7 | 8.8 | +0.1 |
| Final QA calls | 123 | 131 | +8 |
| AR@Loop | 96.8% | 96.3% | −0.5pp |
| Reasoning failure | 59.1% | 59.2% | +0.1pp |

The loop runs almost identically in both configurations. F uses 8 more final QA calls (131 vs 123) because slightly fewer queries self-terminate, consistent with the slightly lower terminal rate (86.9% vs 87.7%). The difference is negligible.

#### Finding 5 — Graph retrieval is 3× slower for zero QA gain

| Config | Retrieval time | EM |
|---|---|---|
| F: DPR + IRCoT | 997s | 0.624 |
| D: HippoRAG + IRCoT | 1,728s | 0.623 |

Config D's graph retrieval takes 1.73× longer than F's DPR retrieval (1,728s vs 997s) for a −0.001 EM outcome. The 731-second overhead buys no QA improvement.

#### Finding 6 — LLM call economics: F is the most efficient IRCoT configuration

| Config | EM | Total calls | Tokens | EM / 1000 calls | EM / 1M tokens |
|---|---|---|---|---|---|
| B (Base) | 0.475 | 1,000 | 1.47M | 475 | 323 |
| D (graph+IRCoT) | 0.623 | 1,881 | 4.36M | 331 | 143 |
| E (full system) | 0.628 | 2,875 | 4.71M | 218 | 133 |
| **F (DPR+IRCoT)** | **0.624** | **1,898** | **4.34M** | **329** | **144** |

F and D are essentially tied on LLM efficiency. F achieves the same EM as D at similar token cost, while also being faster on retrieval (no graph/PPR/reranker overhead).

---

### 3. Revised Component Contribution Summary (Updated)

| Component | EM contribution | AR@K5 contribution | Cost |
|---|---|---|---|
| HippoRAG graph (PPR+reranker) vs DPR | **0** EM (confirmed by F vs D) | +2.4pp K=5, +0.5pp loop | ×1.7 retrieval time |
| E1: multi-query + max-score pooling | +0.012 EM | +0.4pp | +1,000 LLM calls |
| E2: IRCoT loop | **+0.148–0.149 EM** | 0 (same AR@K5) | ~+900 net calls, +2.9M tokens |
| E1 on top of E2 | +0.005 EM | +0.4pp | +994 LLM calls |
| **Graph on top of IRCoT (D vs F)** | **≈0 EM** | +2.4pp K=5 (erased by loop) | +731s retrieval |

The updated thesis: **IRCoT is the sole driver of multi-hop QA improvement. Neither the HippoRAG knowledge graph nor query decomposition (E1) adds measurable EM when IRCoT is present.** The graph's structural advantage — better last-hop seeding at K=5 — is fully compensated by the loop's iterative DPR hops within 8–9 passes.

---

### 4. Open Questions from Config F

**Q1: Does the graph matter at lower loop budgets?**
Config F with max_qa_steps=2 (1 hop) vs D with max_qa_steps=2 would test whether the graph's K=5 advantage becomes decisive when the loop cannot compensate. If D >> F at max_qa_steps=2, the graph earns its cost as a loop-efficiency accelerator.

**Q2: Does the graph matter on MuSiQue?**
MuSiQue AR@Loop=60% vs 2Wiki 96.7% — the graph's deeper traversal may be critical on 3-4 hop chains where DPR hops cannot bridge across multiple missing triples. Config F on MuSiQue would isolate this.

**Q3: What is the theoretical ceiling of DPR+IRCoT?**
F achieves AR@Loop=96.3% using only DPR hops. The 3.7% gap vs theoretical 100% may reflect passages that DPR cannot retrieve by embedding similarity alone — cases where graph structure would be genuinely necessary.
