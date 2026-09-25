from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .contracts import Contracts

# The competition gateway is rate limited: pace every tools/call across all cases.
DEFAULT_MAX_INFLIGHT = 1
DEFAULT_MIN_INTERVAL_SECONDS = 0.5
DEFAULT_CALL_TIMEOUT_SECONDS = 60.0
HTTP_429_BACKOFF_SECONDS = (2.0, 5.0, 10.0)
MAX_RETRY_AFTER_SECONDS = 30.0


class _Pacer:
    """Global limiter: at most ``max_inflight`` calls, starts spaced by ``min_interval``."""

    def __init__(self, max_inflight: int, min_interval: float) -> None:
        self._slots = asyncio.Semaphore(max(1, max_inflight))
        self._lock = asyncio.Lock()
        self._min_interval = max(0.0, min_interval)
        self._next_start = 0.0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._slots:
            async with self._lock:
                loop = asyncio.get_running_loop()
                wait = self._next_start - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next_start = loop.time() + self._min_interval
            yield


class _GuardedTransport(httpx2.AsyncBaseTransport):
    """Keep transport faults per-request instead of crashing the MCP session.

    mcp 2.x sends each POST from the transport's own task group, so a raw httpx error there
    tears down the whole session (and every case in flight). This wrapper turns transport
    errors into a synthetic 503 (surfaced by mcp as a per-request MCPError that the ledger
    retries) and absorbs HTTP 429 rate limiting with bounded, Retry-After-aware backoff.
    """

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()  # buffered body so a rate-limited request can be re-sent
        for attempt in range(len(HTTP_429_BACKOFF_SECONDS) + 1):
            try:
                response = await self._inner.handle_async_request(request)
            except httpx2.TransportError as exc:
                return httpx2.Response(
                    503, text=f"transport error: {type(exc).__name__}", request=request
                )
            if response.status_code != 429 or attempt == len(HTTP_429_BACKOFF_SECONDS):
                return response
            delay = _retry_after(response) or HTTP_429_BACKOFF_SECONDS[attempt]
            await response.aclose()
            await asyncio.sleep(min(delay, MAX_RETRY_AFTER_SECONDS))
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self._inner.aclose()


def _retry_after(response: httpx2.Response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None


class EvidenceGateway:
    def __init__(
        self,
        session: ClientSession,
        contracts: Contracts,
        *,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._contracts = contracts
        self._tools: list[str] | None = None
        self._pacer = _Pacer(max_inflight, min_interval)
        self._call_timeout = call_timeout

    async def list_tools(self) -> list[str]:
        if self._tools is None:
            response = await self._session.list_tools()
            self._tools = sorted(tool.name for tool in response.tools)
        return list(self._tools)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        payload = {"case_id": case_id, **arguments}
        async with self._pacer.slot():
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments=payload), timeout=self._call_timeout
            )
        # mcp>=2 exposes snake_case fields; keep the camelCase fallback for older clients.
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            message = " ".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            raise RuntimeError(f"MCP tool {tool_name} failed: {message or 'unknown error'}")
        evidence = getattr(result, "structuredContent", None)
        if evidence is None:
            evidence = getattr(result, "structured_content", None)
        if evidence is None:
            text_blocks = [block.text for block in result.content if getattr(block, "text", None)]
            if len(text_blocks) != 1:
                raise ValueError(f"MCP tool {tool_name} did not return one evidence object")
            evidence = json.loads(text_blocks[0])
        self._contracts.validate_evidence(evidence, f"MCP tool {tool_name}")
        return evidence


@asynccontextmanager
async def connect_gateway(
    endpoint: str, team_api_key: str, contracts: Contracts
) -> AsyncIterator[EvidenceGateway]:
    headers = {"Authorization": f"Bearer {team_api_key}"}
    call_timeout = float(os.getenv("DAY09_MCP_CALL_TIMEOUT", DEFAULT_CALL_TIMEOUT_SECONDS))
    # Read timeout slightly above the per-call timeout so abandoned POSTs end quickly.
    timeout = httpx2.Timeout(call_timeout + 30.0, connect=30.0, write=30.0, pool=30.0)
    transport = _GuardedTransport(httpx2.AsyncHTTPTransport(retries=2))
    async with (
        httpx2.AsyncClient(headers=headers, timeout=timeout, transport=transport) as http_client,
        streamable_http_client(endpoint, http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield EvidenceGateway(
            session,
            contracts,
            max_inflight=int(os.getenv("DAY09_MCP_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)),
            min_interval=float(os.getenv("DAY09_MCP_MIN_INTERVAL", DEFAULT_MIN_INTERVAL_SECONDS)),
            call_timeout=call_timeout,
        )
