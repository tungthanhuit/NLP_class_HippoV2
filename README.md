<h1 align="center">HippoRAG 2: From RAG to Memory</h1>
<p align="center">
    <img src="https://github.com/OSU-NLP-Group/HippoRAG/raw/main/images/hippo_brain.png" width="55%" style="max-width: 300px;">
</p>

[<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1nuelysWsXL8F5xH6q4JYJI8mvtlmeM9O#scrollTo=TjHdNe2KC81K)

[<img align="center" src="https://img.shields.io/badge/arXiv-2502.14802 HippoRAG 2-b31b1b" />](https://arxiv.org/abs/2502.14802)
[<img align="center" src="https://img.shields.io/badge/🤗 Dataset-HippoRAG 2-yellow" />](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main)
[<img align="center" src="https://img.shields.io/badge/arXiv-2405.14831 HippoRAG 1-b31b1b" />](https://arxiv.org/abs/2405.14831)
[<img align="center" src="https://img.shields.io/badge/GitHub-HippoRAG 1-blue" />](https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy)

### HippoRAG 2 is a powerful memory framework for LLMs that enhances their ability to recognize and utilize connections in new knowledge—mirroring a key function of human long-term memory.

Our experiments show that HippoRAG 2 improves associativity (multi-hop retrieval) and sense-making (the process of integrating large and complex contexts) in even the most advanced RAG systems, without sacrificing their performance on simpler tasks.

Like its predecessor, HippoRAG 2 remains cost and latency efficient in online processes, while using significantly fewer resources for offline indexing compared to other graph-based solutions such as GraphRAG, RAPTOR, and LightRAG.



**Figure 1:** Evaluation of continual learning capabilities across three key dimensions: factual memory (NaturalQuestions, PopQA), sense-making (NarrativeQA), and associativity (MuSiQue, 2Wiki, HotpotQA, and LV-Eval). HippoRAG 2 surpasses other methods across all categories, bringing it one step closer to true long-term memory.



**Figure 2:** HippoRAG 2 methodology.

#### Check out our papers to learn more:

- **[HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)** [NeurIPS '24].
- **[From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](https://arxiv.org/abs/2502.14802)** [ICML '25].

---

## Installation

HippoRAG can be installed as a package, or run directly from this repository.

### Option A: Install from source (recommended for this repo)

```sh
conda create -n hipporag python=3.10
conda activate hipporag

# Install dependencies (CPU-only PyTorch by default; see note below)
pip install -r requirements.txt

# Install this repo in editable mode
pip install -e .
```

**PyTorch note:** `requirements.txt` pins CPU-only PyTorch wheels via `--index-url https://download.pytorch.org/whl/cpu`.
If you want CUDA wheels, install PyTorch separately for your CUDA version (then install the rest of the requirements), or adjust the PyTorch lines.

### Option B: Install from PyPI

```sh
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag
```

Initialize the environmental variables and activate the environment:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your api key (or any non-empty string for many local proxies)>

conda activate hipporag
```

## Quick Start

This section shows how to run HippoRAG as a Python library. If you want to run the full indexing + retrieval + QA pipeline shipped in this repository, see **Reproducing our Experiments** below.

### OpenAI Models

This simple example will illustrate how to use `hipporag` with any OpenAI model:

```python
from hipporag import HippoRAG

# Prepare datasets and evaluation
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County."
]

save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'gpt-4o-mini' # Any OpenAI model name
embedding_model_name = 'text-embedding-3-small'# Embedding model name (OpenAI embeddings or local Sentence-Transformers)

#Startup a HippoRAG instance
hipporag = HippoRAG(save_dir=save_dir, 
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name) 

#Run indexing
hipporag.index(docs=docs)

#Separate Retrieval & QA
queries = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?"
]

retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
qa_results = hipporag.rag_qa(retrieval_results)

#Combined Retrieval & QA
rag_results = hipporag.rag_qa(queries=queries)

```

### Neo4j KB Backend

By default, HippoRAG persists its KB locally (Parquet embeddings + `graph.pickle` + OpenIE JSON). You can switch persistence to Neo4j for both indexing and inference by setting `kb_backend="neo4j"` in `BaseConfig`.

Prereqs:

- Neo4j 5.26.24 running (for example, via local Docker) and reachable at `neo4j_uri`.
- Set `NEO4J_PASSWORD` (or pass `neo4j_password` in config).

#### Same KB, different LLM (Neo4j)

Namespace behavior depends on how you construct `BaseConfig`:

- The library default (`BaseConfig.neo4j_namespace_include_llm=False`) reuses the same Neo4j KB across different `llm_name` values (embedding model still remains part of the namespace).
- This repo's [main.py](main.py) defaults `--neo4j_namespace_include_llm true` to avoid accidental reuse across experiments.

In all cases, the embedding model name is included in the effective namespace.
If you want to run **the same stored KB** (same extracted OpenIE graph + stored embeddings) with **different QA/rerank LLMs**, set:

- `neo4j_namespace_include_llm=False`

and keep these stable across runs:

- `neo4j_namespace` (the prefix you choose)
- `embedding_model_name` (embedding dimensionality must match)

Notes:

- The KB contents (e.g., OpenIE triples) come from the LLM used during indexing. If you change LLM later while reusing the KB, you are changing *inference behavior* (QA/rerank), not rebuilding the KB.
- If you use Milvus (`chunk_vector_backend="milvus"`), the Milvus collection name follows the same effective namespace.

If you are starting Neo4j with Docker, use the 5.26.24 image:

```bash
docker run --name neo4j-hipporag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> \
  neo4j:5.26.24
```

#### Clearing a namespace (Neo4j)

To delete **all** nodes/relationships for a namespace (including stored embeddings), use:

```bash
# Dry-run (prints what would be deleted)
python scripts/clear_kb_namespace.py \
--neo4j-password "$NEO4J_PASSWORD" \
--namespace <EFFECTIVE_NAMESPACE>
```

# Execute deletion

```bash
python scripts/clear_kb_namespace.py \
--neo4j-password "$NEO4J_PASSWORD" \
--namespace <EFFECTIVE_NAMESPACE> \
--yes
```

If you don't know the *effective* namespace string, the script can compute it using the same rule as `HippoRAG`:

```bash
python scripts/clear_kb_namespace.py \
  --neo4j-password "$NEO4J_PASSWORD" \
  --neo4j-namespace hipporag \
  --llm-name gpt-4o-mini \
  --embedding-name text-embedding-3-small \
  --include-llm true \
  --yes
```

Optional: uv usage

```
uv run main.py --dataset=2wikimultihopqa_first10 --retrieval_recall_k_list '[1, 2, 5, 10, 20, 50]'
```

### Teleportation Hybrid Retrieval (experimental)

This repository includes an optional retrieval backend that combines:

- offline entity-community partitioning (Leiden),
- bridge entity/chunk tagging, and
- online leaky local PPR with threshold-triggered cross-community teleportation.

Enable it from the CLI:

```bash
uv run main.py \
  --dataset=2wikimultihopqa_first10 \
  --ppr_mode teleportation_hybrid \
  --teleportation_leakage_gamma 0.15 \
  --teleportation_trigger_threshold 0.003 \
  --teleportation_home_communities_top_k 2 \
  --teleportation_max_teleport_steps 3
```

Tuning guidance:

- Increase `--teleportation_leakage_gamma` to explore cross-community bridges more aggressively.
- Lower `--teleportation_trigger_threshold` to allow earlier teleports.
- Increase `--teleportation_home_communities_top_k` if queries have multiple strong semantic anchors.

Optional: if you are using Milvus chunk vectors, you can also drop the corresponding Milvus collection:

```bash
python scripts/clear_kb_namespace.py \
  --neo4j-password "$NEO4J_PASSWORD" \
  --namespace <EFFECTIVE_NAMESPACE> \
  --drop-milvus true \
  --milvus-uri http://localhost:19530 \
  --yes
```

### Milvus Chunk Vector Backend (optional)

When using the Neo4j KB backend, you can store/search *chunk (passage) embeddings* in Milvus for scalable dense retrieval.

- Neo4j remains the source of truth for chunk content, OpenIE metadata, and graph edges.
- Entity/fact embeddings remain stored in Neo4j in this implementation.

To enable it:

```bash
export HIPPORAG_CHUNK_VECTOR_BACKEND=milvus
export MILVUS_URI=http://localhost:19530
```

Then run the Neo4j test as usual:

```bash
python tests_neo4j.py
```

Requirements:

- Install `pymilvus` (see `requirements.txt`).
- Run Milvus locally and make it reachable at `MILVUS_URI`. For Docker on Linux, Milvus provides a standalone launch script that starts a container named `milvus` on port `19530`:

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Example:

```python
import os
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

cfg = BaseConfig(
  kb_backend="neo4j",
  save_dir="outputs",
  llm_name="gpt-4o-mini",
  embedding_model_name="text-embedding-3-small",
  neo4j_uri="bolt://localhost:7687",
  neo4j_user="neo4j",
  neo4j_password=os.environ["NEO4J_PASSWORD"],
)

hipporag = HippoRAG(global_config=cfg)
hipporag.index(docs=docs)
results = hipporag.rag_qa(queries=queries)
```

For evaluation:

```python
answers = [
    ["Politician"],
    ["By going to the ball."],
    ["Rockland County"]
]

gold_docs = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.",
    "Montebello is a part of Rockland County."]
]

rag_results = hipporag.rag_qa(queries=queries, 
                              gold_docs=gold_docs,
                              gold_answers=answers)
```

#### Example (Local Sentence-Transformers Embeddings)

To use a local embedding model via Sentence-Transformers, set `embedding_model_name` to `Transformers/<hf-id>`.
For example, you can use BGE-M3 as:

```python
from hipporag import HippoRAG

hipporag = HippoRAG(
  save_dir='outputs',
  llm_model_name='gpt-4o-mini',
  embedding_model_name='Transformers/BAAI/bge-m3',
  # Some HF models require this to load correctly:
  embedding_trust_remote_code=True,
)
```

#### Example (OpenAI Compatible Embeddings)

If you want to use LLMs and Embeddings Compatible to OpenAI, please use the following methods.

```python
hipporag = HippoRAG(save_dir=save_dir, 
    llm_model_name='Your LLM Model name',
    llm_base_url='Your LLM Model url',
    embedding_model_name='Your Embedding model name',  
    embedding_base_url='Your Embedding model url')
```

### Local Deployment (OpenAI-compatible server)

This example illustrates how to use `hipporag` with any locally deployed **OpenAI-compatible** server (or proxy) for chat and/or embeddings.

1. Start your server/proxy so it exposes OpenAI-style routes under `/v1` (e.g., `http://localhost:6578/v1` or `http://localhost:4000/v1`).
2. Point HippoRAG to it via `llm_base_url` (and optionally `embedding_base_url`):

```python
save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = # Any OpenAI model name
embedding_model_name = # Embedding model name (OpenAI embeddings or local Sentence-Transformers)
llm_base_url= # Base url for your deployed LLM (i.e. http://localhost:8000/v1)

hipporag = HippoRAG(save_dir=save_dir,
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model_name,
                    llm_base_url=llm_base_url)

# Same Indexing, Retrieval and QA as running OpenAI models above
```

## Testing

When making a contribution to HippoRAG, please run the scripts below to ensure that your changes do not result in unexpected behavior from our core modules. 

These scripts test for indexing, graph loading, document deletion and incremental updates to a HippoRAG object.

### OpenAI Test

To test HippoRAG with an OpenAI LLM and embedding model, simply run the following. 
The cost of this test will be negligible.

```sh
export OPENAI_API_KEY=<your openai api key> 

conda activate hipporag

python tests_openai.py
```

### Local Test

To test locally, start any OpenAI-compatible server (or proxy), then point the tests to it via env vars:

```sh
export OPENAI_API_KEY=dummy
export HIPPORAG_LLM_BASE_URL=http://localhost:4000/v1
export HIPPORAG_EMBEDDING_BASE_URL=http://localhost:4000/v1
```

Then run:

```sh
CUDA_VISIBLE=1 python tests_local.py
```

## Reproducing our Experiments

To use our code to run experiments we recommend you clone this repository and follow the structure of the `main.py` script.

### Data for Reproducibility

We evaluated several sampled datasets in our paper, some of which are already included in the `reproduce/dataset` directory of this repo. For the complete set of datasets, please visit
our [HuggingFace dataset](https://huggingface.co/datasets/osunlp/HippoRAG_v2) and place them under `reproduce/dataset`. We also provide the OpenIE results for both `gpt-4o-mini` and `Llama-3.3-70B-Instruct` for our `musique` sample under `outputs/musique`.

To test your environment is properly set up, you can use the small dataset `reproduce/dataset/sample.json` for debugging as shown below.

### Running Indexing & QA

#### 1) Start an OpenAI-compatible LLM + embeddings endpoint

You need an OpenAI-compatible `/v1` endpoint for **chat** and **embeddings**.

**Option A: OpenAI**

```sh
export OPENAI_API_KEY=<your openai api key>
```

Then pass `--llm_base_url https://api.openai.com/v1` (and optionally `--embedding_base_url https://api.openai.com/v1`).

**Option B: Local proxy (LiteLLM gateway in this repo)**

```sh
# One-time install if you haven't already
pip install 'litellm[proxy]'

# Configure upstream key for the gateway
cp litellm_gateway/.env.example litellm_gateway/.env
# Edit litellm_gateway/.env and set FPT_API_KEY=...

# Start gateway (listens on http://localhost:4000)
./litellm_gateway/start_gateway.sh
```

Then use `--llm_base_url http://localhost:4000/v1`. If your embeddings are served from a different endpoint, also pass `--embedding_base_url`.

#### 2) (Optional) Start Neo4j + Milvus

By default, this repo's [main.py](main.py) uses `--kb_backend neo4j` and `--chunk_vector_backend milvus`.
If you don't want to run external services, see the **Local (parquet) backend** example below.

Neo4j (Docker example):

```bash
docker run --name neo4j-hipporag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> \
  neo4j:5.26.24
```

Milvus (Docker script example):

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Then set:

```sh
export NEO4J_PASSWORD=<your_password>
export MILVUS_URI=http://localhost:19530
```

#### 3) Run indexing + retrieval + QA

`main.py` expects two files for a dataset named `<dataset>`:

- `reproduce/dataset/<dataset>_corpus.json`
- `reproduce/dataset/<dataset>.json`

The repo includes several small datasets (e.g., `sample`, `musique`, `hotpotqa`, `2wikimultihopqa_first10`).

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model

conda activate hipporag
```

### Run with OpenAI Model

```sh
dataset=sample  # or any other dataset under reproduce/dataset

python main.py \
  --dataset "$dataset" \
  --llm_base_url https://api.openai.com/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

### Run with local OpenAI-compatible server

Start any OpenAI-compatible server and point `--llm_base_url` to it:

```sh
dataset=sample
python main.py \
  --dataset "$dataset" \
  --llm_base_url http://localhost:4000/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

If your embeddings endpoint differs from your chat endpoint:

```sh
python main.py \
  --dataset "$dataset" \
  --llm_base_url http://localhost:4000/v1 \
  --embedding_base_url http://localhost:4001/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

### Local (parquet) backend (no Neo4j/Milvus)

To run without external services, switch backends to local persistence:

```sh
dataset=sample
python main.py \
  --dataset "$dataset" \
  --kb_backend parquet \
  --chunk_vector_backend default \
  --llm_base_url http://localhost:4000/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

### OpenIE mode

In this codebase, OpenIE currently runs in **online** mode via the configured OpenAI-compatible LLM API.
`--openie_mode offline` is not supported in the current implementation and will raise an error.

### Split offline indexing from online retrieval/QA

`main.py` now supports running KG construction separately from retrieval/generation via `--pipeline_mode`.

#### A) Build/update KG once (offline indexing only)

```sh
dataset=sample
python main.py \
  --dataset "$dataset" \
  --pipeline_mode index_only \
  --llm_base_url http://localhost:4000/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

#### B) Serve requests with existing KG (no rebuild)

Single query:

```sh
dataset=sample
python main.py \
  --dataset "$dataset" \
  --pipeline_mode retrieve_qa_only \
  --query_text "Who wrote The Hobbit?" \
  --llm_base_url http://localhost:4000/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

Batch queries from JSON (must be a JSON list of strings):

```sh
dataset=sample
python main.py \
  --dataset "$dataset" \
  --pipeline_mode retrieve_qa_only \
  --queries_json_path reproduce/dataset/test_queries.json \
  --llm_base_url http://localhost:4000/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name text-embedding-3-small
```

```sh
# Build KG once:
uv run main.py --dataset musique --pipeline_mode index_only

# Serve/query without rebuilding:
uv run main.py --dataset musique --pipeline_mode retrieve_qa_only --query_text "Who wrote The Hobbit?"

# Batch request-time queries:
uv run main.py --dataset musique --pipeline_mode retrieve_qa_only --queries_json_path queries.json
```

Notes:

- Keep namespace/backend settings stable across runs to reuse the same stored KG (`--neo4j_namespace`, `--neo4j_namespace_include_llm`, `--embedding_name`, backend flags).
- `--ppr_mode global` and `--ppr_mode teleportation_hybrid` can both run against the same constructed KG.
- Use `--force_index_from_scratch true` only when you intentionally want to rebuild.

## Debugging Note

- `/reproduce/dataset/sample.json` is a small dataset specifically for debugging.
- If you want to rerun a particular experiment, remember to clear the saved files, including OpenIE results and knowledge graph, e.g.,

```sh
rm reproduce/dataset/openie_results/openie_sample_results_ner_meta-llama_Llama-3.3-70B-Instruct_3.json
rm -rf outputs/sample/sample_meta-llama_Llama-3.3-70B-Instruct_Transformers_sentence-transformers_all-MiniLM-L6-v2
```

### Useful `main.py` flags

- `--pipeline_mode full|index_only|retrieve_qa_only`: run full pipeline, only KG construction, or retrieval+QA only
- `--query_text "<question>"`: run retrieval+QA for one ad-hoc query (no dataset query file needed)
- `--queries_json_path <path>`: run retrieval+QA for a JSON list of query strings
- `--force_index_from_scratch true`: rebuild everything (clears Neo4j namespace when using `--kb_backend neo4j`)
- `--force_openie_from_scratch true`: regenerate OpenIE extractions
- `--retrieval_recall_k_list '1,2,5,10,20,50'` or `--retrieval_recall_k_list '[1,2,5,10,20,50]'`: control Recall@k evaluation points
- `--neo4j_namespace <prefix>` and `--neo4j_namespace_include_llm false`: control KB reuse/isolation across runs

### Custom Datasets

To setup your own custom dataset for evaluation, follow the format and naming convention shown in `reproduce/dataset/sample_corpus.json` (your dataset's name should be followed by `_corpus.json`). If running an experiment with pre-defined questions, organize your query corpus according to the query file `reproduce/dataset/sample.json`, be sure to also follow our naming convention.

The corpus and optional query JSON files should have the following format:

#### Retrieval Corpus JSON

```json
[
  {
    "title": "FIRST PASSAGE TITLE",
    "text": "FIRST PASSAGE TEXT",
    "idx": 0
  },
  {
    "title": "SECOND PASSAGE TITLE",
    "text": "SECOND PASSAGE TEXT",
    "idx": 1
  }
]
```

#### (Optional) Query JSON

```json

[
  {
    "id": "sample/question_1.json",
    "question": "QUESTION",
    "answer": [
      "ANSWER"
    ],
    "answerable": true,
    "paragraphs": [
      {
        "title": "{FIRST SUPPORTING PASSAGE TITLE}",
        "text": "{FIRST SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 0
      },
      {
        "title": "{SECOND SUPPORTING PASSAGE TITLE}",
        "text": "{SECOND SUPPORTING PASSAGE TEXT}",
        "is_supporting": true,
        "idx": 1
      }
    ]
  }
]
```

#### (Optional) Chunking Corpus

When preparing your data, you may need to chunk each passage, as longer passage may be too complex for the OpenIE process.

## Code Structure

```
📦 .
│-- 📂 src/hipporag
│   ├── 📂 embedding_model          # Implementation of all embedding models
│   │   ├── __init__.py             # Getter function for get specific embedding model classes
|   |   ├── base.py                 # Base embedding model class `BaseEmbeddingModel` to inherit and `EmbeddingConfig`
|   |   ├── ...
│   ├── 📂 evaluation               # Implementation of all evaluation metrics
│   │   ├── __init__.py
|   |   ├── base.py                 # Base evaluation metric class `BaseMetric` to inherit
│   │   ├── qa_eval.py              # Eval metrics for QA
│   │   ├── retrieval_eval.py       # Eval metrics for retrieval
│   ├── 📂 information_extraction  # Implementation of all information extraction models
│   │   ├── __init__.py
|   |   ├── openie_openai_gpt.py    # Model for OpenIE with OpenAI GPT
|   |   ├── openie_transformers_offline.py  # Model for OpenIE with local Transformers models
│   ├── 📂 llm                      # Classes for inference with large language models
│   │   ├── __init__.py             # Getter function
|   |   ├── base.py                 # Config class for LLM inference and base LLM inference class to inherit
|   |   ├── openai_gpt.py           # Class for inference with OpenAI GPT
|   |   ├── transformers_offline.py # Class for inference using local Transformers models
│   ├── 📂 prompts                  # Prompt templates and prompt template manager class
|   │   ├── 📂 dspy_prompts         # Prompts for filtering
|   │   │   ├── ...
|   │   ├── 📂 templates            # All prompt templates for template manager to load
|   │   │   ├── README.md           # Documentations of usage of prompte template manager and prompt template files
|   │   │   ├── __init__.py
|   │   │   ├── triple_extraction.py
|   │   │   ├── ...
│   │   ├── __init__.py
|   |   ├── linking.py              # Instruction for linking
|   |   ├── prompt_template_manager.py  # Implementation of prompt template manager
│   ├── 📂 utils                    # All utility functions used across this repo (the file name indicates its relevant usage)
│   │   ├── config_utils.py         # We use only one config across all modules and its setup is specified here
|   |   ├── ...
│   ├── __init__.py
│   ├── HippoRAG.py          # Highest level class for initiating retrieval, question answering, and evaluations
│   ├── embedding_store.py   # Storage database to load, manage and save embeddings for passages, entities and facts.
│   ├── rerank.py            # Reranking and filtering methods
│-- 📂 examples
│   ├── ...
│   ├── ...
│-- 📜 README.md
│-- 📜 requirements.txt   # Dependencies list
│-- 📜 .gitignore         # Files to exclude from Git


```

## Contact

Questions or issues? File an issue or contact 
[Bernal Jiménez Gutiérrez](mailto:jimenezgutierrez.1@osu.edu),
[Yiheng Shu](mailto:shu.251@osu.edu),
[Yu Su](mailto:su.809@osu.edu),
The Ohio State University

## Citation

If you find this work useful, please consider citing our papers:

### HippoRAG 2

```
@misc{gutiérrez2025ragmemorynonparametriccontinual,
      title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
      year={2025},
      eprint={2502.14802},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14802}, 
}
```

### HippoRAG

```
@inproceedings{gutiérrez2024hipporag,
      title={HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models}, 
      author={Bernal Jiménez Gutiérrez and Yiheng Shu and Yu Gu and Michihiro Yasunaga and Yu Su},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=hkujvAPVsg}
```

## TODO:

- Add support for more embedding models
- Add support for embedding endpoints
- Add support for vector database integration

Please feel free to open an issue or PR if you have any questions or suggestions.