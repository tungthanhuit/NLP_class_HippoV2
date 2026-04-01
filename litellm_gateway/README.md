# LiteLLM Gateway (local)

This folder runs a **local LiteLLM Proxy** that forwards **LLM (chat)** and **embeddings** requests to:

- Upstream `BASE_URL`: `https://mkp-api.fptcloud.com`

It exposes **friendly local model names** so your client code can simply use:

- LLM: `fpt-llm`
- Embeddings: `fpt-embed`

## Prereqs

From repo root:

- `pip install 'litellm[proxy]'`

## Run

1) Put your upstream API key in a `.env` file (recommended):

- Create `litellm_gateway/.env` with:
  - `FPT_API_KEY="..."`

Alternatively, you can still use shell exports:

- `export FPT_API_KEY="..."`

2) Start the gateway:

- `./litellm_gateway/start_gateway.sh`

Auth note:

- If `general_settings.master_key` is NOT set in `litellm_gateway/config.yaml`, the launcher ignores `LITELLM_MASTER_KEY` from env to avoid accidental DB-backed key validation errors.
- If you DO enable `general_settings.master_key`, clients must send the same key (for OpenAI SDK clients, set `OPENAI_API_KEY` to that master key).

By default it listens on:

- `http://localhost:4000`

If your upstream expects OpenAI-style routes under `/v1`, update `api_base` in `litellm_gateway/config.yaml` to `https://mkp-api.fptcloud.com/v1`.

Override host/port (optional):

- `LITELLM_HOST=127.0.0.1 LITELLM_PORT=4000 ./litellm_gateway/start_gateway.sh`

If you keep your `.env` somewhere else:

- `LITELLM_DOTENV_PATH=/path/to/.env ./litellm_gateway/start_gateway.sh`

## Use (OpenAI-compatible)

### Chat completions

- `curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fpt-llm","messages":[{"role":"user","content":"Say hi"}]}'`

### Embeddings

- `curl http://localhost:4000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"fpt-embed","input":"hello world"}'`

## Add/rename models

Edit `litellm_gateway/config.yaml` and add another entry under `model_list`:

```yaml
- model_name: my-local-name
  litellm_params:
    model: openai/<UPSTREAM_MODEL_NAME>
    api_base: https://mkp-api.fptcloud.com
    api_key: os.environ/FPT_API_KEY
```

Then call the gateway using `model: "my-local-name"`.

## Use from HippoRAG

Point your client to the local gateway and use the friendly model names:

- Local base URL: `http://localhost:4000`
- LLM model: `fpt-llm`
- Embedding model: `fpt-embed`
