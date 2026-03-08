# convertdocx2md

Convert SSAF well reports and related CSV context into GraphRAG-ready markdown, build an index, and query via CLI or a FastAPI service.

## What is in this repo

- `ssaf_docx_to_md.py`: Converts Word reports (`.docx`) to normalized markdown.
- `preprocess_headers.py`: Adds well context to headings to improve retrieval after chunking.
- `csv_to_md_profiles.py`: Converts selected profile CSV data into markdown-friendly inputs.
- `settings.yaml`: GraphRAG configuration (Azure OpenAI, Blob storage, Azure AI Search).
- `app/main.py`: FastAPI wrapper exposing `/query/global`, `/query/local`, `/query/drift`.
- `.github/workflows/deploy-aca.yml`: Build and deploy workflow for Azure Container Apps.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

`settings.yaml` requires:

```bash
export GRAPHRAG_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-xx-xx"
export AZURE_OPENAI_CHAT_DEPLOYMENT="<chat-deployment>"
export AZURE_OPENAI_EMBED_DEPLOYMENT="<embed-deployment>"

export AZURE_STORAGE_CONNECTION_STRING="..."
export AZURE_SEARCH_ENDPOINT="https://<search-service>.search.windows.net"
export AZURE_SEARCH_ADMIN_KEY="..."
```

API and deployment also use:

```bash
export SERVICE_API_KEY="..."   # required by app/main.py x-api-key auth
export AZURE_API_KEY="..."     # standardized Azure key name in deployment/runtime env
export AZURE_API_BASE="$AZURE_OPENAI_ENDPOINT"
export AZURE_API_VERSION="$AZURE_OPENAI_API_VERSION"
```

## Data preparation workflow

1. Convert reports to markdown.

```bash
python ssaf_docx_to_md.py ./raw_reports --out-dir ./md --media-root ./md_media
```

2. Move or copy prepared markdown into `input/`.

3. Preprocess headers.

```bash
python preprocess_headers.py
```

4. Build GraphRAG index.

```bash
python -m graphrag index --root .
```

## Query via CLI

```bash
python -m graphrag query "List all anomalies detected for CEDAR-W14" --root . --method local --streaming
python -m graphrag query "Give a field-wide summary of major production issues." --root . --method global --streaming
python -m graphrag query "What likely explains decline in well performance and what should be checked next?" --root . --method drift --streaming
```

Helpful flags:

- `--community-level 2`
- `--response-type "Multiple Paragraphs"`
- `--data ./output`

## Run the API locally

From repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --workers 1
```

Key endpoints:

- `GET /health`
- `POST /query/global`
- `POST /query/local`
- `POST /query/drift`
- `GET /openapi-3.0.json`

Protected endpoints require `x-api-key: $SERVICE_API_KEY`.

Example:

```bash
curl -s -X POST "http://localhost:8000/query/local" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $SERVICE_API_KEY" \
  -d '{"question":"hello","community_level":2,"response_type":"Multiple Paragraphs"}'
```

## GitHub Actions deployment notes

`deploy-aca.yml` now standardizes on `AZURE_API_KEY` and keeps `GRAPHRAG_API_KEY`.

Required repository variables include:

- `ACR_NAME`
- `RESOURCE_GROUP`
- `CONTAINERAPP_NAME`
- `CONTAINERAPP_ENV`
- `GRAPHRAG_ROOT`
- `GRAPHRAG_ARTIFACTS_CONTAINER`
- `GRAPHRAG_ARTIFACTS_PREFIX`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBED_DEPLOYMENT`
- `AZURE_SEARCH_ENDPOINT`

Required repository secrets include:

- `GRAPHRAG_API_KEY`
- `AZURE_API_KEY`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_SEARCH_ADMIN_KEY`
- `SERVICE_API_KEY`
- `ACR_USERNAME`
- `ACR_PASSWORD`
- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

If you previously used `azure-openai-key`, it may persist in ACA until manually removed:

```bash
az containerapp secret remove -g "$RG" -n "$APP" --secret-names azure-openai-key
```
