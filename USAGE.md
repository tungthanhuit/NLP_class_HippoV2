# HippoRAG 2 — Usage Guide

This guide covers environment setup, the LiteLLM gateway, running the pipeline, direct HippoRAG comparison, and the full ablation study.

---

## 1. Environment Setup

### Option A — pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — uv (recommended, much faster)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins CPU-only PyTorch. If you have a GPU, replace the
> `--index-url` block at the top of that file with the appropriate CUDA wheel URL before installing.

---

## 2. Environment Variables

Create a `.env` file at the repo root or inside `litellm_gateway/`:

```dotenv
WOKU_API_KEY=your_woku_key_here
FPT_API_KEY=your_fpt_key_here

# Optional: protect the local gateway with a master key
# LITELLM_MASTER_KEY=my-local-key
```

The gateway startup script auto-loads this file — no manual `export` needed.

---

## 3. LiteLLM Gateway

The gateway proxies both LLM and embedding requests through `http://localhost:4000/v1`.
All pipeline scripts point to this single URL regardless of the upstream provider.

**Start the gateway (keep this terminal open):**

```bash
./litellm_gateway/start_gateway.sh
```

Verify it is up:

```bash
curl http://localhost:4000/v1/models
```

Available model aliases (defined in `litellm_gateway/config.yaml`):

| Alias | Type | Upstream |
|---|---|---|
| `gpt-4o-mini` | LLM | `llm.wokushop.com` |
| `gemini-3.1-flash-lite-preview` | LLM | `llm.wokushop.com` |
| `text-embedding-3-small` | Embeddings | `mkp-api.fptcloud.com` |
| `multilingual-e5-large` | Embeddings | `mkp-api.fptcloud.com` |

To add or rename a model, edit `litellm_gateway/config.yaml` and restart the gateway.

---

## 4. Datasets

Pre-sampled files are in `reproduce/dataset/`. Available splits:

| File pair | Dataset |
|---|---|
| `musique.json` / `musique_corpus.json` | MuSiQue (full) |
| `hotpotqa.json` / `hotpotqa_corpus.json` | HotpotQA |
| `2wikimultihopqa.json` / `2wikimultihopqa_corpus.json` | 2WikiMultiHopQA |
| `musique_small_scale.json` / `…_corpus.json` | MuSiQue (small) |
| `2wikimultihopqa_first10.json` / `…_corpus.json` | First-10 smoke test |

---

## 5. Scripts Overview

| Script | Model | Purpose |
|---|---|---|
| `main.py` | `HippoRAG` | **Standard run** — graph-based retrieval + QA, toggleable enhancements |
| `main_dpr.py` | `StandardRAG` | DPR baseline — plain dense retrieval, no graph |
| `main_ablation.py` | Both | Ablation study — 5 configs, side-by-side comparison table |

---

## 6. Standard Run (`main.py`)

`main.py` uses `HippoRAG` — the full graph-based pipeline. Toggle the enhancements with `--use_enhancements`.

**HippoRAG standard (no enhancements):**

```bash
# pip
python main.py \
  --dataset musique \
  --llm_name gpt-4o-mini \
  --llm_base_url http://localhost:4000/v1 \
  --embedding_name text-embedding-3-small

# uv
uv run main.py \
  --dataset musique \
  --llm_name gpt-4o-mini \
  --llm_base_url http://localhost:4000/v1 \
  --embedding_name text-embedding-3-small
```

**HippoRAG with all enhancements (E1 + E2):**

```bash
python main.py --dataset musique --use_enhancements
uv run main.py --dataset musique --use_enhancements
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `musique` | `musique`, `hotpotqa`, `2wikimultihopqa` |
| `--llm_name` | `gpt-4o-mini` | Model alias from the gateway |
| `--llm_base_url` | `http://localhost:4000/v1` | Gateway base URL |
| `--embedding_name` | `text-embedding-3-small` | Embedding model alias |
| `--save_dir` | `outputs` | Root directory for index and results |
| `--use_enhancements` | off | Enable E1 (RAG Fusion + NER fallback) + E2 (IRCoT) |
| `--mode` | `all` | `all` / `index` (build only) / `qa` (run QA on existing index) |
| `--force_index_from_scratch` | `false` | Rebuild graph even if cached |
| `--log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## 7. Direct Comparison: HippoRAG Standard vs. Enhanced

Run both configurations back-to-back on the same dataset to see the direct impact
of the enhancements. The index is built once and reused.

```bash
# Step 1 — Build index and run standard HippoRAG
python main.py --dataset musique --save_dir outputs/compare --mode all

# Step 2 — Run enhanced HippoRAG on the same index (skip re-indexing)
python main.py --dataset musique --save_dir outputs/compare --mode qa --use_enhancements
```

```bash
# uv equivalents
uv run main.py --dataset musique --save_dir outputs/compare --mode all
uv run main.py --dataset musique --save_dir outputs/compare --mode qa --use_enhancements
```

Results for each run are written to:
```
outputs/compare_musique/<llm>_<embed>/
  pipeline_metrics_snapshot.json   ← full counter snapshot
  qa_metrics.json                  ← EM / F1
  retrieval_metrics_k5.json        ← Recall@5
  run_<timestamp>.log              ← full run log
```

Primary retrieval metric is **R@5 / AR@5** (what the LLM actually reads).
Full-pipeline recall at K=200 is stored as a supplementary field.

---

## 8. DPR Baseline (`main_dpr.py`)

`main_dpr.py` uses `StandardRAG` — plain dense retrieval with no graph or enhancements.
Use it as the lowest baseline before comparing with HippoRAG.

```bash
python main_dpr.py --dataset musique --llm_name gpt-4o-mini
uv run main_dpr.py --dataset musique --llm_name gpt-4o-mini
```

---

## 9. Ablation Study (`main_ablation.py`)

The ablation study runs five configurations on the **same index** and prints a
side-by-side comparison table.

**Enhancement definitions:**

| Label | Includes |
|---|---|
| **E1** | Query decomposition (RAG Fusion) + RRF + NER-seeded DPR fallback |
| **E2** | IRCoT multi-hop reasoning loop |

**Five configs:**

| ID | Name | `use_enhancements` | `max_qa_steps` | What it isolates |
|---|---|---|---|---|
| A | DPR only | — | 1 | Dense retrieval baseline |
| B | HippoRAG base | `False` | 1 | Graph + reranker, no LLM in retrieval |
| C | HippoRAG + E1 | `True` | 1 | Retrieval-side improvements only |
| D | HippoRAG + E2 | `False` | 4 | IRCoT hops only, plain initial retrieval |
| E | HippoRAG + E1+E2 | `True` | 4 | Full system |

**Run all five configs:**

```bash
python main_ablation.py \
  --dataset musique \
  --llm_name gpt-4o-mini \
  --llm_base_url http://localhost:4000/v1 \
  --embedding_name text-embedding-3-small \
  --save_dir outputs/ablation

uv run main_ablation.py \
  --dataset musique \
  --llm_name gpt-4o-mini \
  --llm_base_url http://localhost:4000/v1 \
  --embedding_name text-embedding-3-small \
  --save_dir outputs/ablation
```

**Run a subset:**

```bash
# Only graph base vs. full system
python main_ablation.py --dataset musique --configs B,E

# Only the IRCoT ablation (with and without E1)
python main_ablation.py --dataset musique --configs D,E
```

**Override IRCoT max steps:**

```bash
python main_ablation.py --dataset musique --max_qa_steps_override 3
```

Per-config results are saved to `outputs/ablation/<dataset>/<config_id>/ablation_result.json`.

---

## 10. Metrics Reference

### Retrieval (primary: K=5)

| Metric | Description |
|---|---|
| `R@5` | Any gold passage found in top-5 retrieved |
| `AR@5` | All gold passages found in top-5 |
| `R@N` / `AR@N` | IRCoT context recall — any / all gold in full accumulated context |
| `fallback_rate` | % queries where reranker kept 0 facts → DPR/NER fallback triggered |
| `reranker_keep_rate` | Fraction of candidate facts kept after recognition memory |

### QA

| Metric | Description |
|---|---|
| `exact_match` (EM) | Exact-match accuracy |
| `f1` | Token-level F1 |
| `context_coverage_pct` | % queries where a gold passage is in the LLM's reading context |
| `reasoning_failure_rate_pct` | % queries where gold evidence present but answer is wrong |

### LLM Budget (per phase)

| Metric | Description |
|---|---|
| `reformulation.calls` / `.tokens` | E1 query reformulation cost |
| `ircot.calls` / `.tokens` | IRCoT reasoning step cost |
| `qa.calls` / `.tokens` | Final answer generation cost |
| `total_calls` / `total_tokens` | Aggregate across all phases |

### Enhancement Effectiveness

| Metric | Description |
|---|---|
| `multi_query_rate` | % queries where E1 generated >1 sub-query (RRF triggered) |
| `coverage_audit_rate` | % queries where coverage audit injected a missing-entity fact |
| `ircot_terminal_count` | Queries where IRCoT found the answer before `max_qa_steps` |
| `avg_hop_passages` | Average new passages added per IRCoT retrieval hop |

---

## 11. Tips

**Reuse a cached index** — the graph and embeddings persist on disk. On subsequent runs
omit `--force_index_from_scratch` (default `false`) to skip re-indexing entirely.

**Quick smoke test** — validate the full pipeline with the first-10 split:

```bash
python main.py --dataset 2wikimultihopqa_first10
uv run main.py --dataset 2wikimultihopqa_first10
```

**Index only, QA later:**

```bash
python main.py --dataset musique --mode index
# ... later ...
python main.py --dataset musique --mode qa --use_enhancements
```

**Debug logging** — see per-query fact scores, IRCoT steps, and reranker decisions:

```bash
python main.py --dataset musique --log_level DEBUG
```
