import asyncio
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel, Field, field_validator

import graphrag.api as api
from graphrag.config.load_config import load_config

from azure.storage.blob import BlobServiceClient


API_KEY = os.getenv("SERVICE_API_KEY", "")

def require_api_key(x_api_key: str | None = Header(default=None)):
    if not API_KEY:
        # fail closed in prod; for POC you can allow missing locally if you want,
        # but recommended is to require it when deployed
        raise HTTPException(status_code=500, detail="SERVICE_API_KEY not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------
# Configuration (env vars)
# -------------------------
GRAPHRAG_ROOT = Path(os.getenv("GRAPHRAG_ROOT", ".")).resolve()

# Keep OUTPUT_DIR but do NOT require it. Useful for temp files, debugging, compatibility.
OUTPUT_DIR = Path(os.getenv("GRAPHRAG_OUTPUT_DIR", "/data/output")).resolve()

REQUEST_TIMEOUT_SECONDS = float(os.getenv("GRAPHRAG_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("GRAPHRAG_MAX_CONCURRENT_REQUESTS", "4"))
MAX_QUERY_CHARS = int(os.getenv("GRAPHRAG_MAX_QUERY_CHARS", "1500"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_RESPONSE_TYPE = os.getenv("GRAPHRAG_RESPONSE_TYPE", "Multiple Paragraphs")
DEFAULT_COMMUNITY_LEVEL = int(os.getenv("GRAPHRAG_COMMUNITY_LEVEL", "2"))

# Blob artifacts configuration (recommended path)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
ARTIFACTS_CONTAINER = os.getenv("GRAPHRAG_ARTIFACTS_CONTAINER", "graphrag-artifacts")
ARTIFACTS_PREFIX = os.getenv("GRAPHRAG_ARTIFACTS_PREFIX", "runs/poc").strip("/")

# Cache controls
ARTIFACT_CACHE_TTL_SECONDS = int(os.getenv("GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS", "600"))

# -------------------------
# Structured JSON logging
# -------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time_ms": int(time.time() * 1000),
        }
        for k in ("request_id", "path", "http_method", "status_code", "latency_ms", "graphrag_method"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("graphrag_api")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.propagate = False


# -------------------------
# FastAPI app + globals
# -------------------------
app = FastAPI(
    title="GraphRAG API (Global + Local + DRIFT)",
    version="0.1.0",
    openapi_version="3.0.3"
    )
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

graphrag_config = None

# Artifacts (loaded lazily from Blob)
entities_df: Optional[pd.DataFrame] = None
communities_df: Optional[pd.DataFrame] = None
community_reports_df: Optional[pd.DataFrame] = None
text_units_df: Optional[pd.DataFrame] = None
relationships_df: Optional[pd.DataFrame] = None
covariates_df: Optional[pd.DataFrame] = None  # optional

# Blob client + cache state
_blob_service: Optional[BlobServiceClient] = None
_artifact_lock = asyncio.Lock()
_artifacts_loaded_at: float = 0.0


# -------------------------
# Input sanitization
# -------------------------
_whitespace_re = re.compile(r"\s+")
_control_chars_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_query(q: str) -> str:
    q = q.strip()
    q = _control_chars_re.sub("", q)
    q = _whitespace_re.sub(" ", q)
    return q


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    community_level: int = Field(DEFAULT_COMMUNITY_LEVEL, ge=0, le=10)
    response_type: str = Field(DEFAULT_RESPONSE_TYPE, min_length=1, max_length=128)
    verbose: bool = False

    # Global-only knobs
    dynamic_community_selection: bool = False

    @field_validator("question")
    @classmethod
    def _sanitize(cls, v: str) -> str:
        v = sanitize_query(v)
        if not v:
            raise ValueError("question cannot be empty after sanitization")
        return v


# -------------------------
# Provenance helpers
# -------------------------
def _debug_summary(context: Any) -> Dict[str, Any]:
    if isinstance(context, dict):
        return {"context_type": "dict", "keys": list(context.keys())[:50]}
    if isinstance(context, list):
        return {"context_type": "list", "len": len(context)}
    if isinstance(context, pd.DataFrame):
        return {"context_type": "dataframe", "shape": list(context.shape), "columns": list(context.columns)[:50]}
    return {"context_type": type(context).__name__}


def _try_extract_citations(context: Any) -> List[Dict[str, Any]]:
    """
    Best-effort extraction for docs-friendly provenance.
    GraphRAG context can vary by method/version; keep it safe and small.
    """
    citations: List[Dict[str, Any]] = []

    if isinstance(context, dict):
        for key in (
            "citations",
            "sources",
            "source_text_units",
            "text_units",
            "used_text_units",
            "retrieved_text_units",
        ):
            if key in context:
                val = context.get(key)
                if isinstance(val, pd.DataFrame):
                    citations = val.head(10).to_dict(orient="records")
                elif isinstance(val, list):
                    citations = [{"value": x} if not isinstance(x, dict) else x for x in val[:20]]
                else:
                    citations = [{"key": key, "value": str(val)[:300]}]
                break

        if not citations:
            for k, v in context.items():
                if isinstance(v, pd.DataFrame) and ("text" in v.columns or "id" in v.columns):
                    citations = [{"dataframe": k, "sample": v.head(10).to_dict(orient="records")}]
                    break

    return citations


# -------------------------
# Helpers: output dir + blob artifacts
# -------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Output path exists but is not a directory: {path}")


def _blob_key(filename: str) -> str:
    # artifacts are stored as: runs/poc/<filename>
    return f"{ARTIFACTS_PREFIX}/{filename}" if ARTIFACTS_PREFIX else filename


def _get_blob_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "This service is configured to load GraphRAG artifacts from Azure Blob."
            )
        _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return _blob_service


def _read_parquet_from_blob(filename: str) -> pd.DataFrame:
    """
    Download a parquet blob into memory and load into a pandas DataFrame.
    """
    svc = _get_blob_service()
    blob_client = svc.get_blob_client(container=ARTIFACTS_CONTAINER, blob=_blob_key(filename))
    data = blob_client.download_blob().readall()
    return pd.read_parquet(io.BytesIO(data))


async def _ensure_artifacts_loaded(force: bool = False) -> None:
    """
    Lazily load required parquet artifacts from Blob with TTL caching.
    This runs per worker; it avoids doing heavy downloads in startup.
    """
    global entities_df, communities_df, community_reports_df, text_units_df, relationships_df, covariates_df
    global _artifacts_loaded_at

    now = time.time()
    cache_valid = (now - _artifacts_loaded_at) < ARTIFACT_CACHE_TTL_SECONDS
    already_loaded = all(
        x is not None
        for x in (entities_df, communities_df, community_reports_df, text_units_df, relationships_df)
    )

    if not force and already_loaded and cache_valid:
        return

    async with _artifact_lock:
        # Re-check after acquiring lock
        now = time.time()
        cache_valid = (now - _artifacts_loaded_at) < ARTIFACT_CACHE_TTL_SECONDS
        already_loaded = all(
            x is not None
            for x in (entities_df, communities_df, community_reports_df, text_units_df, relationships_df)
        )
        if not force and already_loaded and cache_valid:
            return

        start = time.perf_counter()

        # Required by methods
        entities_df = await asyncio.to_thread(_read_parquet_from_blob, "entities.parquet")
        communities_df = await asyncio.to_thread(_read_parquet_from_blob, "communities.parquet")
        community_reports_df = await asyncio.to_thread(_read_parquet_from_blob, "community_reports.parquet")
        text_units_df = await asyncio.to_thread(_read_parquet_from_blob, "text_units.parquet")
        relationships_df = await asyncio.to_thread(_read_parquet_from_blob, "relationships.parquet")

        # Optional
        try:
            covariates_df = await asyncio.to_thread(_read_parquet_from_blob, "covariates.parquet")
        except Exception:
            covariates_df = None

        _artifacts_loaded_at = time.time()
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"Loaded GraphRAG artifacts from blob://{ARTIFACTS_CONTAINER}/{ARTIFACTS_PREFIX} in {latency_ms}ms")


# -------------------------
# Startup: load config only (NO artifact reads here)
# -------------------------
@app.on_event("startup")
def startup() -> None:
    global graphrag_config

    settings_path = GRAPHRAG_ROOT / "settings.yaml"
    if not settings_path.exists():
        raise RuntimeError(f"settings.yaml not found under GRAPHRAG_ROOT={GRAPHRAG_ROOT}")

    # Ensure output dir exists, but do NOT require /app/output or local parquets
    _ensure_dir(OUTPUT_DIR)

    graphrag_config = load_config(GRAPHRAG_ROOT)
    logger.info(f"GraphRAG config loaded from {GRAPHRAG_ROOT}. OUTPUT_DIR={OUTPUT_DIR} (not required for artifacts).")


# -------------------------
# Middleware: request logging
# -------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        rec = logging.LogRecord("graphrag_api", logging.INFO, __file__, 0, "request", (), None)
        rec.request_id = request_id
        rec.path = request.url.path
        rec.http_method = request.method
        rec.status_code = status_code
        rec.latency_ms = latency_ms
        logger.handle(rec)


# -------------------------
# Health endpoint (FAST, reliable)
# -------------------------
@app.get("/health", operation_id="health_check", tags=["health"])
def health():
    artifacts_loaded = all(
        x is not None
        for x in (entities_df, communities_df, community_reports_df, text_units_df, relationships_df)
    )
    cache_age_s = int(time.time() - _artifacts_loaded_at) if _artifacts_loaded_at else None

    return {
        "status": "ok",
        "graphrag_root": str(GRAPHRAG_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "max_query_chars": MAX_QUERY_CHARS,
        "blob_configured": bool(AZURE_STORAGE_CONNECTION_STRING),
        "artifacts_container": ARTIFACTS_CONTAINER,
        "artifacts_prefix": ARTIFACTS_PREFIX,
        "artifact_cache_ttl_seconds": ARTIFACT_CACHE_TTL_SECONDS,
        "artifacts_loaded": artifacts_loaded,
        "artifact_cache_age_seconds": cache_age_s,
        "covariates_loaded": covariates_df is not None,
    }


# -------------------------
# Shared executor wrapper
# -------------------------
async def _run_with_limits(method_name: str, coro) -> Dict[str, Any]:
    request_id = str(uuid4())
    start = time.perf_counter()

    async with _sem:
        try:
            response, context = await asyncio.wait_for(coro, timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"{method_name} timed out after {REQUEST_TIMEOUT_SECONDS}s")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{method_name} failed: {e}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    rec = logging.LogRecord("graphrag_api", logging.INFO, __file__, 0, "graphrag_query", (), None)
    rec.request_id = request_id
    rec.path = f"/query/{method_name}"
    rec.http_method = "POST"
    rec.status_code = 200
    rec.latency_ms = latency_ms
    rec.graphrag_method = method_name
    logger.handle(rec)

    return {
        "answer": response,
        "citations": _try_extract_citations(context),
        "debug": {
            "request_id": request_id,
            "method": method_name,
            "latency_ms": latency_ms,
            **_debug_summary(context),
        },
    }


# -------------------------
# Global / Local / DRIFT endpoints
# -------------------------
@app.post("/query/global", operation_id="graphrag_global_search")
async def query_global(req: QueryRequest, _: None = Depends(require_api_key)):
    if graphrag_config is None:
        raise HTTPException(status_code=500, detail="GraphRAG config not loaded")

    await _ensure_artifacts_loaded()

    coro = api.global_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        community_level=req.community_level,
        dynamic_community_selection=req.dynamic_community_selection,
        response_type=req.response_type,
        query=req.question,
        verbose=req.verbose,
    )
    return await _run_with_limits("global", coro)


@app.post("/query/local", operation_id="graphrag_local_search")
async def query_local(req: QueryRequest, _: None = Depends(require_api_key)):
    if graphrag_config is None:
        raise HTTPException(status_code=500, detail="GraphRAG config not loaded")

    await _ensure_artifacts_loaded()

    coro = api.local_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        text_units=text_units_df,
        relationships=relationships_df,
        covariates=covariates_df,
        community_level=req.community_level,
        response_type=req.response_type,
        query=req.question,
        verbose=req.verbose,
    )
    return await _run_with_limits("local", coro)


@app.post("/query/drift", operation_id="graphrag_drift_search")
async def query_drift(req: QueryRequest, _: None = Depends(require_api_key)):
    if graphrag_config is None:
        raise HTTPException(status_code=500, detail="GraphRAG config not loaded")

    await _ensure_artifacts_loaded()

    coro = api.drift_search(
        config=graphrag_config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        text_units=text_units_df,
        relationships=relationships_df,
        community_level=req.community_level,
        response_type=req.response_type,
        query=req.question,
        verbose=req.verbose,
    )
    return await _run_with_limits("drift", coro)