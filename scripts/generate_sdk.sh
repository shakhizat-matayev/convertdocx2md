#!/usr/bin/env bash
set -euo pipefail

PACKAGE_NAME="${PACKAGE_NAME:-graphrag_sdk_client}"
PROJECT_NAME="${PROJECT_NAME:-graphrag-sdk-client}"
OUTPUT_DIR="${OUTPUT_DIR:-./sdk_client}"
CONFIG_PATH="${CONFIG_PATH:-./openapi-python-client-config.yaml}"

if ! command -v openapi-python-client >/dev/null 2>&1; then
  echo "openapi-python-client is not installed."
  echo "Install with: pip install openapi-python-client"
  exit 1
fi

TMP_SPEC="$(mktemp --suffix=.json)"
trap 'rm -f "$TMP_SPEC"' EXIT

python - <<'PY' "$TMP_SPEC"
import json
import sys
from pathlib import Path

from app.main import app

out = Path(sys.argv[1])
out.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
print(f"Wrote OpenAPI schema to {out}")
PY

openapi-python-client generate \
  --path "$TMP_SPEC" \
  --config "$CONFIG_PATH" \
  --meta none \
  --output-path "$OUTPUT_DIR" \
  --overwrite

echo "SDK generated in: $OUTPUT_DIR"
