from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import FastAPI

log = logging.getLogger(__name__)

# Routes tagged with this are exposed as MCP tools. Applied directly on the
# route decorators that should be exposed (see open_webui.routers.knowledge)
# so the exposed surface is visible at the route definition instead of a
# separately-maintained operation-id list here.
KNOWLEDGE_MCP_TAG = 'mcp'

# Ceilings applied to every MCP tool-call response, regardless of which
# endpoint produced it. Endpoints exposed as MCP tools can return
# arbitrarily large text (full file content, RAG query results, ...), and
# MCP clients cap how big a single tool result can be -- an oversized
# response fails outright instead of just being slow. These are enforced at
# the transport layer (see _ResponseCappingTransport) so individual routes
# don't need their own size limits, and direct (non-MCP) callers of the same
# routes are completely unaffected.
MCP_RESPONSE_MAX_CHARS = 2000
MCP_RESPONSE_MAX_ITEMS = 20


def _cap_json_value(value: Any, max_chars: int, max_items: int) -> Any:
    """Recursively truncate long strings and long lists in a JSON-serializable structure."""
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return f'{value[:max_chars]}... [truncated, {len(value) - max_chars} more characters]'
    if isinstance(value, list):
        capped = [_cap_json_value(item, max_chars, max_items) for item in value[:max_items]]
        if len(value) > max_items:
            capped.append(f'... [truncated, {len(value) - max_items} more items omitted]')
        return capped
    if isinstance(value, dict):
        return {key: _cap_json_value(item, max_chars, max_items) for key, item in value.items()}
    return value


class _ResponseCappingTransport(httpx.AsyncBaseTransport):
    """Wraps another transport and caps the size of any JSON response.

    FastApiMCP dispatches each tool call through an httpx.AsyncClient back
    into this same app (see setup_knowledge_mcp). Wrapping that client's
    transport bounds every MCP tool result in one place instead of adding
    size limits to each exposed route -- a direct (non-MCP) call to one of
    those routes doesn't go through this client at all, so it's unaffected.
    """

    def __init__(self, transport: httpx.AsyncBaseTransport, max_chars: int, max_items: int):
        self._transport = transport
        self._max_chars = max_chars
        self._max_items = max_items

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)

        if 'application/json' not in response.headers.get('content-type', ''):
            return response

        await response.aread()
        try:
            data = json.loads(response.content)
        except ValueError:
            return response

        capped_body = json.dumps(_cap_json_value(data, self._max_chars, self._max_items)).encode('utf-8')

        # response.aread() above already transparently decompressed the
        # original body according to its real content-encoding; capped_body
        # is fresh, uncompressed JSON, so the encoding/length headers that
        # described the original compressed bytes no longer apply. Carrying
        # them over verbatim claims e.g. "zstd" for a plain-text body, and
        # httpx's own decoder then tries to zstd-decode it and raises
        # (`ZstdError: Unknown frame descriptor`).
        headers = httpx.Headers(response.headers)
        headers.pop('content-length', None)
        headers.pop('content-encoding', None)
        return httpx.Response(
            status_code=response.status_code,
            headers=headers,
            content=capped_body,
            request=request,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


def _force_stateless_mcp_sessions(mcp) -> None:
    """Work around fastapi-mcp==0.4.0 hardcoding `stateless=False` (no public
    toggle exists) when it builds the SDK's StreamableHTTPSessionManager --
    see fastapi_mcp.transport.http.FastApiHttpSessionManager, which stores
    sessions in a dict private to whichever worker process created them.
    Under UVICORN_WORKERS>1 each worker has its own dict, so a session
    created on one 404s ("Session not found") on every other worker and the
    client disconnects/reconnects in a loop. There's no shared-store option
    to configure instead -- it's an open, unimplemented feature request
    upstream: https://github.com/tadata-org/fastapi_mcp/issues/208

    None of the knowledge tools we expose need multi-turn session state,
    sampling, or server-initiated notifications -- each tool call is just a
    thin wrapper around a stateless REST endpoint -- so running the
    transport stateless (a fresh transport per request; any client-supplied
    session id is simply ignored rather than looked up) is safe here and
    sidesteps the problem entirely instead of requiring session affinity in
    front of the app. `self.stateless` is read fresh on every request
    dispatch (mcp.server.streamable_http_manager), so flipping it after the
    manager has already started still takes effect for every request that
    follows.

    This has to patch a private, underscore-prefixed hook because 0.4.0
    exposes no public one; if a future version changes these internals this
    silently becomes a no-op (logged), rather than raising.
    """
    http_transport = getattr(mcp, '_http_transport', None)
    original_ensure_started = getattr(http_transport, '_ensure_session_manager_started', None)
    if http_transport is None or original_ensure_started is None:
        log.warning(
            'Could not force MCP sessions stateless (fastapi-mcp internals changed); '
            'sessions may break under UVICORN_WORKERS > 1.'
        )
        return

    async def _ensure_started_stateless() -> None:
        await original_ensure_started()
        manager = getattr(http_transport, '_session_manager', None)
        if manager is not None and not manager.stateless:
            manager.stateless = True
            log.info('MCP session manager forced stateless (safe under UVICORN_WORKERS > 1)')

    http_transport._ensure_session_manager_started = _ensure_started_stateless


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
    # (via the ASGI transport below) so the existing get_verified_user /
    # AccessGrants checks apply per caller. The transport is the same ASGI
    # one FastApiMCP would build by default (see its source), wrapped so
    # responses get capped before they come back as a tool result.
    http_client = httpx.AsyncClient(
        transport=_ResponseCappingTransport(
            httpx.ASGITransport(app=app, raise_app_exceptions=False),
            max_chars=MCP_RESPONSE_MAX_CHARS,
            max_items=MCP_RESPONSE_MAX_ITEMS,
        ),
        base_url='http://apiserver',
        timeout=10.0,
    )

    mcp = FastApiMCP(
        app,
        name='Open WebUI Knowledge & Memory',
        description='Read/write access to Open WebUI knowledge bases and user memories.',
        http_client=http_client,
        include_tags=[KNOWLEDGE_MCP_TAG],
        headers=['authorization'],
    )
    mcp.mount_http(router=app, mount_path=mount_path)
    _force_stateless_mcp_sessions(mcp)
    log.info(f'FastAPI-MCP knowledge server mounted at {mount_path}')
