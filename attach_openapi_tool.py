import os
import json
import re
import base64
import jsonref
import requests
from typing import Any
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.projects.models import ConnectionType
from azure.ai.projects.models import (
    PromptAgentDefinition,
    OpenApiTool,
    OpenApiFunctionDefinition,
    OpenApiProjectConnectionAuthDetails,
    OpenApiProjectConnectionSecurityScheme,
)

# ---- REQUIRED ENV VARS ----
PROJECT_ENDPOINT = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

# Connection name as shown in Foundry "Manage connected resources"
OPENAPI_CONNECTION_NAME = os.getenv("OPENAPI_PROJECT_CONNECTION_NAME", "graphrag-api-key-conn")
OPENAPI_CONNECTION_ID = os.getenv("OPENAPI_PROJECT_CONNECTION_ID", "").strip()

# Your OpenAPI 3.0 schema URL (the one you generated)
OPENAPI_SPEC_URL = os.getenv(
    "OPENAPI_SPEC_URL",
    "https://graphrag-api.purpleocean-5db79053.westeurope.azurecontainerapps.io/openapi-3.0.json",
)

# Agent name to create/update (choose a stable name)
AGENT_NAME = os.getenv("AGENT_NAME", "GraphRAG-Agent")


def _decode_jwt_claim(token: str, claim_name: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(decoded.decode("utf-8"))
        value = claims.get(claim_name)
        return str(value) if value else None
    except Exception:
        return None


def _extract_subscription_from_endpoint(endpoint: str) -> str | None:
    match = re.search(r"/subscriptions/([0-9a-fA-F-]{36})", endpoint)
    return match.group(1) if match else None


def _infer_tenant_from_credential(credential: DefaultAzureCredential) -> str | None:
    try:
        token = credential.get_token("https://management.azure.com/.default").token
        return _decode_jwt_claim(token, "tid")
    except Exception:
        return None


def _print_verification_context(credential: DefaultAzureCredential, agent: Any) -> None:
    tenant_id = (
        os.getenv("AZURE_TENANT_ID")
        or os.getenv("ARM_TENANT_ID")
        or _infer_tenant_from_credential(credential)
        or "<unknown>"
    )

    subscription_id = (
        os.getenv("AZURE_SUBSCRIPTION_ID")
        or os.getenv("ARM_SUBSCRIPTION_ID")
        or _extract_subscription_from_endpoint(PROJECT_ENDPOINT)
        or "<unknown>"
    )

    print("Verification context:")
    print(f"  project_endpoint={PROJECT_ENDPOINT}")
    print(f"  tenant_id={tenant_id}")
    print(f"  subscription_id={subscription_id}")
    print(f"  agent_id={getattr(agent, 'id', '<unknown>')}")
    print(f"  agent_name={getattr(agent, 'name', '<unknown>')}")
    print(f"  agent_version={getattr(agent, 'version', '<unknown>')}")


def _load_openapi_spec(url: str) -> dict:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    try:
        openapi_spec = response.json()
    except ValueError as exc:
        raise RuntimeError(f"OpenAPI spec from {url} is not valid JSON") from exc

    try:
        resolved = jsonref.replace_refs(openapi_spec, proxies=False, lazy_load=False)
        return json.loads(json.dumps(resolved))
    except Exception:
        return openapi_spec


def _extract_api_key_header(spec: dict[str, Any]) -> tuple[str, str]:
    security_schemes = spec.get("components", {}).get("securitySchemes", {})
    if not isinstance(security_schemes, dict) or not security_schemes:
        raise RuntimeError("OpenAPI spec has no components.securitySchemes; cannot configure auth")

    for scheme_name, scheme in security_schemes.items():
        if not isinstance(scheme, dict):
            continue
        if scheme.get("type") == "apiKey" and scheme.get("in") == "header":
            header_name = str(scheme.get("name") or "").strip()
            if not header_name:
                raise RuntimeError(f"Security scheme '{scheme_name}' is apiKey/header but has no header name")
            return scheme_name, header_name

    raise RuntimeError(
        "No header-based apiKey security scheme found in OpenAPI spec. "
        "The GraphRAG API expects API key auth."
    )


def _build_connection_security_scheme(conn_id: str):
    attempts = [
        {"project_connection_id": conn_id, "connection_id": conn_id},
        {"project_connection_id": conn_id},
        {"connection_id": conn_id},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return OpenApiProjectConnectionSecurityScheme(**kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not build OpenAPI connection security scheme from SDK model. "
        "Check azure-ai-projects package compatibility."
    ) from last_error


def _resolve_connection_id(project_client: AIProjectClient) -> str:
    if OPENAPI_CONNECTION_ID:
        print("Using project connection ID from OPENAPI_PROJECT_CONNECTION_ID")
        return OPENAPI_CONNECTION_ID

    available: list[tuple[str, str, bool]] = []

    try:
        connection = project_client.connections.get(OPENAPI_CONNECTION_NAME)
        conn_id = getattr(connection, "id", None)
        if conn_id:
            print(f"Using project connection '{OPENAPI_CONNECTION_NAME}' id={conn_id}")
            return conn_id
    except ResourceNotFoundError:
        print(
            f"Connection '{OPENAPI_CONNECTION_NAME}' not found. "
            "Trying fallback resolution using available/default ApiKey connections..."
        )

    for conn in project_client.connections.list():
        name = str(getattr(conn, "name", "") or "")
        conn_type = str(getattr(conn, "connection_type", "") or "")
        is_default = bool(getattr(conn, "is_default", False))
        available.append((name, conn_type, is_default))

    for name, conn_type, _ in available:
        if name.lower() == OPENAPI_CONNECTION_NAME.lower() and name:
            connection = project_client.connections.get(name)
            conn_id = getattr(connection, "id", None)
            if conn_id:
                print(f"Using project connection '{name}' id={conn_id} (case-insensitive match)")
                return conn_id

    api_key_like_names = [
        name for name, conn_type, _ in available if conn_type in {ConnectionType.API_KEY.value, ConnectionType.CUSTOM_KEYS.value}
    ]
    if len(api_key_like_names) == 1:
        name = api_key_like_names[0]
        connection = project_client.connections.get(name)
        conn_id = getattr(connection, "id", None)
        if conn_id:
            print(f"Using only available API key-like connection '{name}' id={conn_id}")
            return conn_id

    for connection_type in (ConnectionType.API_KEY, ConnectionType.CUSTOM_KEYS):
        try:
            default_conn = project_client.connections.get_default(connection_type)
            conn_id = getattr(default_conn, "id", None)
            if conn_id:
                name = getattr(default_conn, "name", "<default>")
                print(f"Using default {connection_type.value} connection '{name}' id={conn_id}")
                return conn_id
        except ResourceNotFoundError:
            continue

    if available:
        listing = ", ".join(
            f"{name or '<unnamed>'} ({conn_type or 'unknown'}, default={is_default})"
            for name, conn_type, is_default in available
        )
    else:
        listing = "<none>"

    raise RuntimeError(
        "No usable Foundry project connection was found for OpenAPI tool authentication. "
        f"Requested name='{OPENAPI_CONNECTION_NAME}'. Available connections: {listing}. "
        "Set OPENAPI_PROJECT_CONNECTION_NAME correctly or provide OPENAPI_PROJECT_CONNECTION_ID."
    )

def main():
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential) as project_client:
        # 1) Resolve Project Connection ID (what the OpenAPI tool needs)
        conn_id = _resolve_connection_id(project_client)

        # 2) Load OpenAPI schema
        openapi_spec = _load_openapi_spec(OPENAPI_SPEC_URL)
        scheme_name, header_name = _extract_api_key_header(openapi_spec)
        print(f"OpenAPI auth scheme detected: {scheme_name} (header '{header_name}')")

        if header_name.lower() != "x-api-key":
            print(
                "⚠️ Warning: OpenAPI header is not 'x-api-key'. "
                "Verify the Foundry connection injects the exact header expected by the API."
            )

        # 3) Create OpenAPI tool using project connection authentication
        tool = OpenApiTool(
            openapi=OpenApiFunctionDefinition(
                name="graphrag_api",
                description="GraphRAG API: global/local/drift queries with citations",
                spec=openapi_spec,
                auth=OpenApiProjectConnectionAuthDetails(
                    security_scheme=_build_connection_security_scheme(conn_id)
                ),
            )
        )

        # 4) Create a new agent version with this tool attached
        agent = project_client.agents.create_version(
            agent_name=AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_DEPLOYMENT,
                instructions=(
                    "You are a GraphRAG assistant. Use the OpenAPI tool for answering questions. "
                    "Use /query/global for broad summaries, /query/local for entity-specific questions, "
                    "and /query/drift for complex questions requiring both."
                ),
                tools=[tool],
            ),
        )

        _print_verification_context(credential, agent)
        print(f"✅ Agent created/updated: id={agent.id}, name={agent.name}, version={agent.version}")
        print(
            "Auth check reminder: if Playground still fails, verify project connection secret matches SERVICE_API_KEY "
            "and that it is sent as header 'x-api-key'."
        )
        print("Now open the Foundry portal -> Agents -> select this agent and test in Playground.")

if __name__ == "__main__":
    main()