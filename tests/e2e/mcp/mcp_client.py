"""Client for the mcp e2e suite.

Management routes (/v1/mcp/server CRUD) go through the shared Gateway
transport. The MCP protocol itself (initialize, tools/list, tools/call over
streamable HTTP) goes through the official mcp SDK, the same client library
production MCP hosts use, aimed at either of the gateway's two URL namespaces:
the per-server {PROXY}/{alias}/mcp, or the aggregate {PROXY}/mcp where one
session spans every server the key may use and the `x-mcp-servers` header
narrows it to named aliases. The gateway accepts the LiteLLM virtual key as
either `x-litellm-api-key: Bearer sk-...` or `Authorization: Bearer sk-...`
(both Bearer-prefixed on the MCP routes, matching the docs); `McpAuth` models
the two header styles so each test states which one it drives.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal, Mapping, cast

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, RootModel

from e2e_config import PROXY_BASE_URL, REQUEST_TIMEOUT
from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, unwrap
from models import McpServerCreateBody, McpServerInfo

McpHeaderName = Literal["x-litellm-api-key", "Authorization"]

ToolArguments = Mapping[str, str | float]


@dataclass(frozen=True, slots=True)
class McpAuth:
    """One of the two documented ways to present a LiteLLM key to the MCP gateway."""

    header_name: McpHeaderName
    key: str

    def headers(self) -> dict[str, str]:
        return {self.header_name: f"Bearer {self.key}"}


@dataclass(frozen=True, slots=True)
class McpToolNames:
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class McpDenied:
    """A tools/list attempt the gateway rejected before serving the session."""

    status_code: int | None
    message: str


ListToolsOutcome = McpToolNames | McpDenied


@dataclass(frozen=True, slots=True)
class McpToolText:
    """The text content of one tools/call result."""

    text: str
    is_error: bool


CallToolOutcome = McpToolText | McpDenied


class StubToolStats(BaseModel):
    """JSON the stub's `stats` tool returns for one marker (see stub/stub_server.py)."""

    marker: str
    max_in_flight: int
    completed: int


class StubRecordedHeaders(RootModel[dict[str, str]]):
    """JSON the stub's `recorded_headers` tool returns: the (lowercased) header
    map of the most recent request its auth guard let through."""


def _mcp_url(alias: str) -> str:
    return f"{PROXY_BASE_URL}/{alias}/mcp"


def _aggregate_url() -> str:
    return f"{PROXY_BASE_URL}/mcp"


def _aggregate_headers(auth: McpAuth, servers: tuple[str, ...]) -> dict[str, str]:
    return {**auth.headers(), "x-mcp-servers": ",".join(servers)}


def _first_text(result: CallToolResult) -> str:
    first = result.content[0] if result.content else None
    if isinstance(first, TextContent):
        return first.text
    return f"<non-text content: {type(first).__name__}>"


def _http_status(root: BaseException) -> int | None:
    """The HTTP status behind an SDK failure; the SDK wraps transport errors in
    (possibly nested) ExceptionGroups, so walk them without recursing."""
    pending: list[BaseException] = [root]  # mutable-ok: bounded worklist over an exception tree
    while pending:
        exc = pending.pop()
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        if isinstance(exc, BaseExceptionGroup):
            pending.extend(cast("tuple[BaseException, ...]", exc.exceptions))
    return None


def _http_client(headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(REQUEST_TIMEOUT), follow_redirects=True)


async def _list_tool_names(url: str, headers: dict[str, str]) -> tuple[str, ...]:
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                return tuple(sorted(tool.name for tool in listed.tools))


async def _call_tool(url: str, headers: dict[str, str], tool: str, arguments: ToolArguments) -> McpToolText:
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, dict(arguments))
                return McpToolText(text=_first_text(result), is_error=bool(result.isError))


@dataclass(frozen=True, slots=True)
class McpClient:
    gateway: Gateway

    def create_server(self, body: McpServerCreateBody) -> McpServerInfo:
        return unwrap(
            self.gateway.transport.post(
                "/v1/mcp/server",
                headers=self.gateway.transport.master,
                json=body,
                response_type=McpServerInfo,
            )
        )

    def server_info(self, server_id: str) -> McpServerInfo:
        return unwrap(
            self.gateway.transport.get(
                f"/v1/mcp/server/{server_id}",
                headers=self.gateway.transport.master,
                params=NoBody(),
                response_type=McpServerInfo,
            )
        )

    def delete_server(self, server_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def list_tools_once(self, alias: str, auth: McpAuth) -> ListToolsOutcome:
        try:
            return McpToolNames(names=asyncio.run(_list_tool_names(_mcp_url(alias), auth.headers())))
        except Exception as exc:  # noqa: BLE001 - the SDK raises ExceptionGroup-wrapped transport errors; modelled as a value
            return McpDenied(status_code=_http_status(exc), message=str(exc))

    def poll_tool_names(self, alias: str, auth: McpAuth) -> tuple[str, ...]:
        """tools/list to the shared deadline: a just-created server record
        propagates to the gateway asynchronously and a just-created key can lag
        the auth cache, so the first attempts may 401 or list nothing."""
        deadline = time.monotonic() + self.gateway.poll_timeout
        outcome: ListToolsOutcome = McpDenied(status_code=None, message="never attempted")
        while time.monotonic() < deadline:
            outcome = self.list_tools_once(alias, auth)
            match outcome:
                case McpToolNames(names=names) if names:
                    return names
                case _:
                    time.sleep(self.gateway.poll_interval)
        pytest.fail(
            f"MCP tools for {alias!r} never listed within {self.gateway.poll_timeout}s; last outcome: {outcome}"
        )

    def call_tool(self, alias: str, auth: McpAuth, tool: str, arguments: ToolArguments) -> McpToolText:
        """One tools/call over its own fresh MCP session, so concurrent callers
        behave like independent clients."""
        return asyncio.run(_call_tool(_mcp_url(alias), auth.headers(), tool, arguments))

    def call_tool_once(self, alias: str, auth: McpAuth, tool: str, arguments: ToolArguments) -> CallToolOutcome:
        """A tools/call whose rejection is an outcome, not a crash: the gateway
        surfaces permission denials through the protocol stream, which the SDK
        raises client-side; denial tests assert on the McpDenied value."""
        try:
            return self.call_tool(alias, auth, tool, arguments)
        except Exception as exc:  # noqa: BLE001 - the SDK raises ExceptionGroup-wrapped transport errors; modelled as a value
            return McpDenied(status_code=_http_status(exc), message=str(exc))

    def aggregate_list_tools_once(self, auth: McpAuth, servers: tuple[str, ...]) -> ListToolsOutcome:
        """tools/list over one aggregate {PROXY}/mcp session, narrowed to the
        aliases in `servers` via the x-mcp-servers header."""
        try:
            return McpToolNames(names=asyncio.run(_list_tool_names(_aggregate_url(), _aggregate_headers(auth, servers))))
        except Exception as exc:  # noqa: BLE001 - the SDK raises ExceptionGroup-wrapped transport errors; modelled as a value
            return McpDenied(status_code=_http_status(exc), message=str(exc))

    def poll_aggregate_tool_names(
        self, auth: McpAuth, servers: tuple[str, ...], *, until_listed: str
    ) -> tuple[str, ...]:
        """Aggregate tools/list to the shared deadline, until `until_listed`
        appears (the settle signal that the last-created record has propagated
        to the gateway); returns that full listing."""
        deadline = time.monotonic() + self.gateway.poll_timeout
        outcome: ListToolsOutcome = McpDenied(status_code=None, message="never attempted")
        while time.monotonic() < deadline:
            outcome = self.aggregate_list_tools_once(auth, servers)
            match outcome:
                case McpToolNames(names=names) if until_listed in names:
                    return names
                case _:
                    time.sleep(self.gateway.poll_interval)
        pytest.fail(
            f"aggregate MCP listing for {servers!r} never included {until_listed!r} "
            f"within {self.gateway.poll_timeout}s; last outcome: {outcome}"
        )

    def aggregate_call_tool(
        self, auth: McpAuth, servers: tuple[str, ...], tool: str, arguments: ToolArguments
    ) -> McpToolText:
        """One tools/call over its own fresh aggregate {PROXY}/mcp session."""
        return asyncio.run(_call_tool(_aggregate_url(), _aggregate_headers(auth, servers), tool, arguments))

    def stub_stats(self, alias: str, auth: McpAuth, stats_tool: str, marker: str) -> StubToolStats:
        outcome = self.call_tool(alias, auth, stats_tool, {"marker": marker})
        return StubToolStats.model_validate_json(outcome.text)

    def stub_recorded_headers(self, alias: str, auth: McpAuth, headers_tool: str) -> dict[str, str]:
        """The headers the stub's auth guard recorded for its most recent
        authorized request: by construction, the tools/call this method just
        made, i.e. exactly what the gateway sent upstream on the caller's behalf."""
        outcome = self.call_tool(alias, auth, headers_tool, {})
        assert outcome.is_error is False, f"recorded_headers call errored: {outcome.text[:300]}"
        return StubRecordedHeaders.model_validate_json(outcome.text).root


def build_client() -> McpClient:
    return McpClient(gateway=build_gateway())
