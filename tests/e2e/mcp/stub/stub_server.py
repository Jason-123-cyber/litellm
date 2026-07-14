"""Deterministic MCP upstreams for the mcp e2e suite.

One process, one port, four streamable-http MCP mounts plus a static OAuth2
token endpoint, so the compose stack keeps a single `mcp-stub` service:

- `/mcp` — anonymous. `echo` answers immediately so auth tests can assert an
  exact round-trip; `slow_echo` holds the request open for `sleep_seconds`
  while per-`marker` in-flight and max-in-flight counters track how many calls
  the proxy let through simultaneously (the observable a per-server
  `max_concurrent_requests` cap must bound); `stats` reads those counters back,
  so tests observe upstream concurrency through the proxy itself and the stub
  needs no side-channel port.
- `/second/mcp` — anonymous, with a deliberately disjoint tool set
  (`second_ping`). Aggregate-routing tests register it as a second gateway
  server; a call only this mount can answer proves which upstream served it.
- `/apikey/mcp` — rejects any request whose `X-API-Key` is not exactly
  UPSTREAM_API_KEY, the header the gateway injects for `auth_type: api_key`.
- `/oauth/mcp` — rejects any request whose `Authorization` is not exactly
  `Bearer OAUTH_ACCESS_TOKEN`. That token is only obtainable from
  `/oauth/token`, so a served request proves the gateway ran the
  client_credentials exchange rather than forwarding something it already had.
- `/oauth/token` — the client_credentials token endpoint: validates the
  grant_type/client_id/client_secret/scope form exactly and answers with
  OAUTH_ACCESS_TOKEN.

The guarded mounts record the headers of the most recent authorized request;
their `recorded_headers` tool reads them back through the proxy, so tests can
assert exactly which credentials the gateway attached upstream (and that the
caller's LiteLLM virtual key never left the gateway).

The credential constants are mirrored in tests/e2e/e2e_config.py; keep the two
in sync. Counter updates are plain attribute mutations between awaits, so
asyncio's single-threaded scheduling makes them atomic; markers come from
`unique_marker()` so concurrent test runs never share a counter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

UPSTREAM_API_KEY = "e2e-stub-upstream-api-key"
OAUTH_CLIENT_ID = "e2e-stub-oauth-client-id"
OAUTH_CLIENT_SECRET = "e2e-stub-oauth-client-secret"
OAUTH_SCOPE = "tools:read"
OAUTH_ACCESS_TOKEN = "e2e-stub-minted-access-token"

main_mcp = FastMCP("e2e-stub", host="0.0.0.0", port=8765, stateless_http=True)
second_mcp = FastMCP("e2e-stub-second", host="0.0.0.0", port=8765, stateless_http=True)
apikey_mcp = FastMCP("e2e-stub-apikey", host="0.0.0.0", port=8765, stateless_http=True)
oauth_mcp = FastMCP("e2e-stub-oauth", host="0.0.0.0", port=8765, stateless_http=True)


@dataclass
class _MarkerStats:
    in_flight: int = 0
    max_in_flight: int = 0
    completed: int = 0


_stats: dict[str, _MarkerStats] = {}

_last_authorized_headers: dict[str, dict[str, str]] = {}


@main_mcp.tool()
def echo(text: str) -> str:
    """Return `text` unchanged."""
    return text


@main_mcp.tool()
async def slow_echo(text: str, marker: str, sleep_seconds: float) -> str:
    """Return `text` after `sleep_seconds`, recording concurrency under `marker`."""
    stats = _stats.setdefault(marker, _MarkerStats())
    stats.in_flight += 1
    stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
    try:
        await asyncio.sleep(sleep_seconds)
    finally:
        stats.in_flight -= 1
        stats.completed += 1
    return text


@main_mcp.tool()
def stats(marker: str) -> str:
    """Return the JSON stats recorded for `marker`."""
    recorded = _stats.get(marker, _MarkerStats())
    return json.dumps(
        {
            "marker": marker,
            "max_in_flight": recorded.max_in_flight,
            "completed": recorded.completed,
        }
    )


@second_mcp.tool()
def second_ping() -> str:
    """Identify the /second upstream; no other mount serves this tool."""
    return "pong-from-second"


def _register_guarded_tools(server: FastMCP, mount: str) -> None:
    def echo(text: str) -> str:
        """Return `text` unchanged."""
        return text

    def recorded_headers() -> str:
        """Return the headers of the most recent authorized request as JSON."""
        return json.dumps(_last_authorized_headers.get(mount, {}))

    _ = server.tool()(echo)
    _ = server.tool()(recorded_headers)


_register_guarded_tools(apikey_mcp, "apikey")
_register_guarded_tools(oauth_mcp, "oauth")


def _require_header(app: ASGIApp, *, mount: str, header: str, expected: str) -> ASGIApp:
    """Serve `app` only to requests carrying `header: expected`; 401 otherwise.
    Authorized requests have their full header map recorded under `mount`."""

    async def guard(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if headers.get(header) != expected:
            response = JSONResponse({"error": "unauthorized", "detail": f"missing or wrong {header}"}, status_code=401)
            await response(scope, receive, send)
            return
        _last_authorized_headers[mount] = dict(headers.items())
        await app(scope, receive, send)

    return guard


async def oauth_token(request: Request) -> JSONResponse:
    """The client_credentials token endpoint. Every form field is matched
    exactly so a failure points at the precise field the proxy sent wrong."""
    form = await request.form()
    granted = (
        form.get("grant_type") == "client_credentials"
        and form.get("client_id") == OAUTH_CLIENT_ID
        and form.get("client_secret") == OAUTH_CLIENT_SECRET
        and form.get("scope") == OAUTH_SCOPE
    )
    if not granted:
        return JSONResponse({"error": "invalid_client"}, status_code=401)
    return JSONResponse({"access_token": OAUTH_ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600})


def build_app() -> Starlette:
    servers = (main_mcp, second_mcp, apikey_mcp, oauth_mcp)
    apps = {server.name: server.streamable_http_app() for server in servers}

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for server in servers:
                await stack.enter_async_context(server.session_manager.run())
            yield

    return Starlette(
        routes=[
            Route("/oauth/token", oauth_token, methods=["POST"]),
            Mount(
                "/oauth",
                app=_require_header(
                    apps["e2e-stub-oauth"],
                    mount="oauth",
                    header="authorization",
                    expected=f"Bearer {OAUTH_ACCESS_TOKEN}",
                ),
            ),
            Mount(
                "/apikey",
                app=_require_header(
                    apps["e2e-stub-apikey"],
                    mount="apikey",
                    header="x-api-key",
                    expected=UPSTREAM_API_KEY,
                ),
            ),
            Mount("/second", app=apps["e2e-stub-second"]),
            Mount("/", app=apps["e2e-stub"]),
        ],
        lifespan=lifespan,
    )


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=8765)
