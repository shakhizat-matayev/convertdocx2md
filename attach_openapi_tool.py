import os
import json
import jsonref
import requests
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
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

# Your OpenAPI 3.0 schema URL (the one you generated)
OPENAPI_SPEC_URL = os.getenv(
    "OPENAPI_SPEC_URL",
    "https://graphrag-api.purpleocean-5db79053.westeurope.azurecontainerapps.io/openapi-3.0.json",
)

# Agent name to create/update (choose a stable name)
AGENT_NAME = os.getenv("AGENT_NAME", "GraphRAG-Agent")

def main():
    with DefaultAzureCredential() as credential, AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential) as project_client:
        # 1) Resolve Project Connection ID (what the OpenAPI tool needs)
        conn_id = project_client.connections.get(OPENAPI_CONNECTION_NAME).id
        print(f"Using project connection '{OPENAPI_CONNECTION_NAME}' id={conn_id}")

        # 2) Load OpenAPI schema
        spec_text = requests.get(OPENAPI_SPEC_URL, timeout=30).text
        openapi_spec = json.loads(spec_text)

        # 3) Create OpenAPI tool using project connection authentication
        tool = OpenApiTool(
            openapi=OpenApiFunctionDefinition(
                name="graphrag_api",
                description="GraphRAG API: global/local/drift queries with citations",
                spec=openapi_spec,
                auth=OpenApiProjectConnectionAuthDetails(
                    security_scheme=OpenApiProjectConnectionSecurityScheme(
                        project_connection_id=conn_id
                    )
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

        print(f"✅ Agent created/updated: id={agent.id}, name={agent.name}, version={agent.version}")
        print("Now open the Foundry portal -> Agents -> select this agent and test in Playground.")

if __name__ == "__main__":
    main()