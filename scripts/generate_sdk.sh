#!/usr/bin/env bash
set -euo pipefail

OPENAPI_URL="${OPENAPI_URL:-http://localhost:8000/openapi-3.0.json}"
PACKAGE_NAME="${PACKAGE_NAME:-graphrag_sdk_client}"
PROJECT_NAME="${PROJECT_NAME:-graphrag-sdk-client}"
OUTPUT_DIR="${OUTPUT_DIR:-./sdk_client}"

if ! command -v openapi-python-client >/dev/null 2>&1; then
  echo "openapi-python-client is not installed."
  echo "Install with: pip install openapi-python-client"
  exit 1
fi

TMP_SPEC="$(mktemp)"
trap 'rm -f "$TMP_SPEC"' EXIT

curl -fsSL "$OPENAPI_URL" -o "$TMP_SPEC"

openapi-python-client generate \
  --path "$TMP_SPEC" \
  --meta none \
  --package-name "$PACKAGE_NAME" \
  --project-name "$PROJECT_NAME" \
  --output-path "$OUTPUT_DIR" \
  --overwrite

echo "SDK generated in: $OUTPUT_DIR"
