# convertdocx2md

Convert SSAF Well Health Reports from Word (`.docx`), Master Table and 4D Seismic CSVs into GraphRAG-ready Markdown, index them, and run GraphRAG queries.

## What this repository includes

- `ssaf_docx_to_md.py`: Converts report files to normalized Markdown.
- `preprocess_headers.py`: Injects well context into headers to reduce identity loss during chunking.
- `settings.yaml`: GraphRAG config (Azure OpenAI, Blob storage, Azure AI Search).
- `app/main.py`: Optional FastAPI wrapper for `/query/global`, `/query/local`, and `/query/drift`.

## Quick Start

### 1. Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure required environment variables

`settings.yaml` expects these values:

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

### 3. Convert DOCX reports to Markdown

Single file:

```bash
python ssaf_docx_to_md.py ssaf_report_Alder-W02.docx \
  -o ssaf_report_Alder-W02.md \
  --media-dir ssaf_report_Alder-W02_media
```

Batch folder:

```bash
python ssaf_docx_to_md.py ./reports \
  --out-dir ./md \
  --media-root ./md_media
```

### 4. Prepare GraphRAG input and preprocess headers

Place your `.md` files in `input/`, then run:

```bash
python preprocess_headers.py
```

### 5. Run indexing

```bash
python -m graphrag index --root .
```

## Run Test Prompts (Query)

Use the same shell/venv where indexing worked:

```bash
python -m graphrag query "List all anomalies detected for CEDAR-W14" --root . --method local --streaming
python -m graphrag query "Give a field-wide summary of major production issues." --root . --method global --streaming
python -m graphrag query "What likely explains decline in well performance and what should be checked next?" --root . --method drift --streaming
```

Helpful options:

```bash
python -m graphrag query --help
```

Common flags:

- `--community-level 2`
- `--response-type "3 bullet points"`
- `--data ./output` (force local parquet output directory)

## Do I need Azure Cloud Shell for queries?

No. If indexing and querying work in your current venv/shell, keep using it.

Cloud Shell is optional and only needed if your current environment cannot reach Azure services or does not have the required credentials/secrets.

## Output structure

For each converted report:

- `md/<report>.md` (normalized, GraphRAG-ready)
- `md_media/<report>/...` (extracted plots)
