from __future__ import annotations

import logging

from fastapi import FastAPI

log = logging.getLogger(__name__)

# Routes tagged with this are exposed as MCP tools. Applied directly on the
# route decorators that should be exposed (see open_webui.routers.knowledge)
# so the exposed surface is visible at the route definition instead of a
# separately-maintained operation-id list here.
KNOWLEDGE_MCP_TAG = 'mcp'


def setup_knowledge_mcp(app: FastAPI, mount_path: str = '/api/v1/mcp') -> None:
    """Mount a FastAPI-MCP server exposing the knowledge routes tagged `mcp`.

    No-op (just a log line) if the optional `fastapi-mcp` dependency isn't
    installed. Any other failure during setup is left to propagate so a real
    misconfiguration isn't mistaken for a missing optional dependency.
    """
    try:
        from fastapi_mcp import FastApiMCP
    except ImportError:
        log.info('fastapi-mcp is not installed; skipping MCP knowledge server setup')
        return

    # FastApiMCP forwards the Authorization header into each tool invocation
    # (via an in-process ASGI transport it builds itself) so the existing
    # get_verified_user / AccessGrants checks apply per caller.
    mcp = FastApiMCP(
        app,
        name='Open WebUI Knowledge',
        description='Read/write access to Open WebUI knowledge bases.',
        include_tags=[KNOWLEDGE_MCP_TAG],
        headers=['authorization'],
    )
    mcp.mount_http(router=app, mount_path=mount_path)
    log.info(f'FastAPI-MCP knowledge server mounted at {mount_path}')
