import asyncio
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from hmac import compare_digest
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import uuid4

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

import graphrag.api as graphrag_api
from graphrag.config.load_config import load_config


# -------------------------
# Environment configuration
# -------------------------
GRAPHRAG_ROOT = Path(os.getenv("GRAPHRAG_ROOT", ".")).resolve()
OUTPUT_DIR = Path(os.getenv("GRAPHRAG_OUTPUT_DIR", "/data/output")).resolve()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

REQUEST_TIMEOUT_SECONDS = float(os.getenv("GRAPHRAG_REQUEST_TIMEOUT_SECONDS", "120"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("GRAPHRAG_MAX_CONCURRENT_REQUESTS", "4"))
MAX_QUERY_CHARS = int(os.getenv("GRAPHRAG_MAX_QUERY_CHARS", "1500"))
DEFAULT_RESPONSE_TYPE = os.getenv("GRAPHRAG_RESPONSE_TYPE", "Multiple Paragraphs")
DEFAULT_COMMUNITY_LEVEL = int(os.getenv("GRAPHRAG_COMMUNITY_LEVEL", "2"))

ARTIFACTS_CONTAINER = os.getenv("GRAPHRAG_ARTIFACTS_CONTAINER", "graphrag-artifacts")
ARTIFACTS_PREFIX = os.getenv("GRAPHRAG_ARTIFACTS_PREFIX", "runs/poc").strip("/")
ARTIFACT_CACHE_TTL_SECONDS = int(os.getenv("GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS", "600"))

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "").strip()

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("graphrag_api")


# -------------------------
# Security
# -------------------------
api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False, scheme_name="ApiKeyAuth")


def require_api_key(api_key: str | None = Security(api_key_scheme)) -> None:
    if not SERVICE_API_KEY:
        raise HTTPException(status_code=500, detail="SERVICE_API_KEY is not configured")
    if api_key is None or not compare_digest(api_key, SERVICE_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


# -------------------------
# API models (OpenAPI-first)
# -------------------------
class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    request_id: Optional[str] = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    community_level: int = Field(default=DEFAULT_COMMUNITY_LEVEL, ge=0, le=10)
    response_type: str = Field(default=DEFAULT_RESPONSE_TYPE, min_length=1, max_length=128)
    verbose: bool = Field(default=False)
    dynamic_community_selection: bool = Field(default=False)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        collapsed = re.sub(r"\s+", " ", value)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", collapsed).strip()
        if not cleaned:
            raise ValueError("question cannot be empty after sanitization")
        return cleaned


class QueryMethod(str, Enum):
    global_search = "global"
    local_search = "local"
    drift_search = "drift"


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    method: QueryMethod
    answer: str
    latency_ms: int = Field(ge=0)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    graphrag_root: str
    output_dir: str
    timeout_seconds: float
    max_concurrent_requests: int
    max_query_chars: int
    artifacts_container: str
    artifacts_prefix: str
    artifact_cache_ttl_seconds: int
    artifacts_loaded: bool
    artifact_cache_age_seconds: Optional[int] = None


# -------------------------
# Runtime state
# -------------------------
@dataclass
class RuntimeState:
    config_loaded: bool
    sem: asyncio.Semaphore
    artifact_lock: asyncio.Lock
    artifacts_loaded_at: float
    blob_service: Optional[BlobServiceClient]
    graphrag_config: object
    entities_df: Optional[pd.DataFrame]
    communities_df: Optional[pd.DataFrame]
    community_reports_df: Optional[pd.DataFrame]
    text_units_df: Optional[pd.DataFrame]
    relationships_df: Optional[pd.DataFrame]
    covariates_df: Optional[pd.DataFrame]


state = RuntimeState(
    config_loaded=False,
    sem=asyncio.Semaphore(MAX_CONCURRENT_REQUESTS),
    artifact_lock=asyncio.Lock(),
    artifacts_loaded_at=0.0,
    blob_service=None,
    graphrag_config=None,
    entities_df=None,
    communities_df=None,
    community_reports_df=None,
    text_units_df=None,
    relationships_df=None,
    covariates_df=None,
)


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="GraphRAG Query API",
    version="1.0.0",
    openapi_version="3.0.3",
)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Output path is not a directory: {path}")


def _blob_key(filename: str) -> str:
    if ARTIFACTS_PREFIX:
        return f"{ARTIFACTS_PREFIX}/{filename}"
    return filename


def _get_blob_service() -> BlobServiceClient:
    if state.blob_service is not None:
        return state.blob_service

    if AZURE_STORAGE_CONNECTION_STRING:
        state.blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        return state.blob_service

    if AZURE_STORAGE_ACCOUNT_URL:
        credential = DefaultAzureCredential()
        state.blob_service = BlobServiceClient(account_url=AZURE_STORAGE_ACCOUNT_URL, credential=credential)
        return state.blob_service

    raise RuntimeError(
        "Blob storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING "
        "or AZURE_STORAGE_ACCOUNT_URL."
    )


def _read_parquet_from_blob(filename: str) -> pd.DataFrame:
    client = _get_blob_service().get_blob_client(container=ARTIFACTS_CONTAINER, blob=_blob_key(filename))
    payload = client.download_blob().readall()
    return pd.read_parquet(io.BytesIO(payload))


async def _ensure_artifacts_loaded(force: bool = False) -> None:
    loaded = all(
        item is not None
        for item in (
            state.entities_df,
            state.communities_df,
            state.community_reports_df,
            state.text_units_df,
            state.relationships_df,
        )
    )
    fresh = (time.time() - state.artifacts_loaded_at) < ARTIFACT_CACHE_TTL_SECONDS
    if loaded and fresh and not force:
        return

    async with state.artifact_lock:
        loaded = all(
            item is not None
            for item in (
                state.entities_df,
                state.communities_df,
                state.community_reports_df,
                state.text_units_df,
                state.relationships_df,
            )
        )
        fresh = (time.time() - state.artifacts_loaded_at) < ARTIFACT_CACHE_TTL_SECONDS
        if loaded and fresh and not force:
            return

        started = time.perf_counter()

        required_names = [
            "entities.parquet",
            "communities.parquet",
            "community_reports.parquet",
            "text_units.parquet",
            "relationships.parquet",
        ]
        required_tasks = [
            asyncio.to_thread(_read_parquet_from_blob, filename)
            for filename in required_names
        ]
        (
            state.entities_df,
            state.communities_df,
            state.community_reports_df,
            state.text_units_df,
            state.relationships_df,
        ) = await asyncio.gather(*required_tasks)

        try:
            state.covariates_df = await asyncio.to_thread(_read_parquet_from_blob, "covariates.parquet")
        except Exception:
            state.covariates_df = None

        state.artifacts_loaded_at = time.time()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "artifacts_loaded_ms=%s container=%s prefix=%s",
            elapsed_ms,
            ARTIFACTS_CONTAINER,
            ARTIFACTS_PREFIX,
        )


@app.on_event("startup")
def startup() -> None:
    settings_path = GRAPHRAG_ROOT / "settings.yaml"
    if not settings_path.exists():
        raise RuntimeError(f"settings.yaml not found under GRAPHRAG_ROOT={GRAPHRAG_ROOT}")

    _ensure_output_dir(OUTPUT_DIR)
    state.graphrag_config = load_config(GRAPHRAG_ROOT)
    state.config_loaded = True
    logger.info("graphrag_config_loaded root=%s", GRAPHRAG_ROOT)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _error_response(status_code: int, message: str, request_id: Optional[str]) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiError(error=message, request_id=request_id).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return _error_response(exc.status_code, detail, request_id)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.warning("validation_failed request_id=%s errors=%s", request_id, len(exc.errors()))
    return _error_response(422, "Validation failed", request_id)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.exception("unhandled_error request_id=%s", request_id)
    return _error_response(500, "Internal server error", request_id)


@app.get(
    "/health",
    response_model=HealthResponse,
    operation_id="get_health",
    tags=["health"],
)
def health() -> HealthResponse:
    artifacts_loaded = all(
        item is not None
        for item in (
            state.entities_df,
            state.communities_df,
            state.community_reports_df,
            state.text_units_df,
            state.relationships_df,
        )
    )
    cache_age = int(time.time() - state.artifacts_loaded_at) if state.artifacts_loaded_at else None

    return HealthResponse(
        status="ok",
        graphrag_root=str(GRAPHRAG_ROOT),
        output_dir=str(OUTPUT_DIR),
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        max_query_chars=MAX_QUERY_CHARS,
        artifacts_container=ARTIFACTS_CONTAINER,
        artifacts_prefix=ARTIFACTS_PREFIX,
        artifact_cache_ttl_seconds=ARTIFACT_CACHE_TTL_SECONDS,
        artifacts_loaded=artifacts_loaded,
        artifact_cache_age_seconds=cache_age,
    )


async def _run_query(method: QueryMethod, req: QueryRequest, request_id: str) -> QueryResponse:
    if not state.config_loaded or state.graphrag_config is None:
        raise HTTPException(status_code=500, detail="GraphRAG config not loaded")

    await _ensure_artifacts_loaded()

    if method == QueryMethod.global_search:
        query_coro = graphrag_api.global_search(
            config=state.graphrag_config,
            entities=state.entities_df,
            communities=state.communities_df,
            community_reports=state.community_reports_df,
            community_level=req.community_level,
            dynamic_community_selection=req.dynamic_community_selection,
            response_type=req.response_type,
            query=req.question,
            verbose=req.verbose,
        )
    elif method == QueryMethod.local_search:
        query_coro = graphrag_api.local_search(
            config=state.graphrag_config,
            entities=state.entities_df,
            communities=state.communities_df,
            community_reports=state.community_reports_df,
            text_units=state.text_units_df,
            relationships=state.relationships_df,
            covariates=state.covariates_df,
            community_level=req.community_level,
            response_type=req.response_type,
            query=req.question,
            verbose=req.verbose,
        )
    else:
        query_coro = graphrag_api.drift_search(
            config=state.graphrag_config,
            entities=state.entities_df,
            communities=state.communities_df,
            community_reports=state.community_reports_df,
            text_units=state.text_units_df,
            relationships=state.relationships_df,
            community_level=req.community_level,
            response_type=req.response_type,
            query=req.question,
            verbose=req.verbose,
        )

    started = time.perf_counter()
    async with state.sem:
        try:
            answer, _context = await asyncio.wait_for(query_coro, timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="GraphRAG query timed out")
        except HTTPException:
            raise
        except Exception:
            logger.exception("graphrag_query_failed method=%s request_id=%s", method.value, request_id)
            raise HTTPException(status_code=502, detail="GraphRAG query failed")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return QueryResponse(
        request_id=request_id,
        method=method,
        answer=str(answer),
        latency_ms=elapsed_ms,
    )


QUERY_RESPONSES = {
    401: {"model": ApiError, "description": "Unauthorized"},
    422: {"model": ApiError, "description": "Validation error"},
    500: {"model": ApiError, "description": "Internal server error"},
    502: {"model": ApiError, "description": "GraphRAG backend error"},
    504: {"model": ApiError, "description": "Backend timeout"},
}

DRIFT_STREAM_RESPONSES = {
    200: {
        "description": "Server-sent event stream with drift answer chunks",
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string"
                }
            }
        },
    },
    401: {"model": ApiError, "description": "Unauthorized"},
    422: {"model": ApiError, "description": "Validation error"},
    500: {"model": ApiError, "description": "Internal server error"},
    502: {"model": ApiError, "description": "GraphRAG backend error"},
    504: {"model": ApiError, "description": "Backend timeout"},
}


def _chunk_text(value: str, size: int = 700) -> list[str]:
    if not value:
        return [""]
    return [value[index:index + size] for index in range(0, len(value), size)]


async def _drift_sse_stream(req: QueryRequest, request: Request) -> AsyncIterator[str]:
    request_id = request.state.request_id
    try:
        result = await _run_query(QueryMethod.drift_search, req, request_id)

        start_payload = {
            "type": "start",
            "request_id": result.request_id,
            "method": result.method.value,
            "latency_ms": result.latency_ms,
        }
        yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\\n\\n"

        for chunk in _chunk_text(result.answer):
            if await request.is_disconnected():
                return
            chunk_payload = {
                "type": "chunk",
                "request_id": result.request_id,
                "delta": chunk,
            }
            yield f"data: {json.dumps(chunk_payload, ensure_ascii=False)}\\n\\n"
            await asyncio.sleep(0)

        end_payload = {
            "type": "end",
            "request_id": result.request_id,
        }
        yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\\n\\n"
    except HTTPException as exc:
        error_payload = {
            "type": "error",
            "request_id": request_id,
            "error": exc.detail if isinstance(exc.detail, str) else "Request failed",
            "status_code": exc.status_code,
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\\n\\n"
    except Exception:
        logger.exception("drift_stream_failed request_id=%s", request_id)
        error_payload = {
            "type": "error",
            "request_id": request_id,
            "error": "Internal server error",
            "status_code": 500,
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\\n\\n"


@app.post(
    "/query/global",
    response_model=QueryResponse,
    operation_id="query_global",
    tags=["query"],
    responses=QUERY_RESPONSES,
)
async def query_global(req: QueryRequest, request: Request, _: None = Depends(require_api_key)) -> QueryResponse:
    return await _run_query(QueryMethod.global_search, req, request.state.request_id)


@app.post(
    "/query/local",
    response_model=QueryResponse,
    operation_id="query_local",
    tags=["query"],
    responses=QUERY_RESPONSES,
)
async def query_local(req: QueryRequest, request: Request, _: None = Depends(require_api_key)) -> QueryResponse:
    return await _run_query(QueryMethod.local_search, req, request.state.request_id)


@app.post(
    "/query/drift",
    operation_id="query_drift",
    tags=["query"],
    responses=DRIFT_STREAM_RESPONSES,
)
async def query_drift(req: QueryRequest, request: Request, _: None = Depends(require_api_key)) -> StreamingResponse:
    return StreamingResponse(
        _drift_sse_stream(req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/openapi-3.0.json", include_in_schema=False)
def openapi_3_0() -> JSONResponse:
    schema = app.openapi()
    if PUBLIC_BASE_URL:
        schema["servers"] = [{"url": PUBLIC_BASE_URL.rstrip("/")}]
    return JSONResponse(content=schema)
