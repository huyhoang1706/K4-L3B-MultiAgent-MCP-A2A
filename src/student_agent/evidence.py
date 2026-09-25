"""Case-scoped evidence ledger: the only path from an agent to the MCP Evidence Gateway.

Responsibilities:
- least privilege: each actor may only call the tools listed in TOOL_PERMISSIONS;
- case scope: every call carries this case's case_id and results are never shared across cases;
- efficiency: identical (tool, arguments) requests within a case are served from cache;
- bounded retry: only transient failures (timeouts, dropped streams, HTTP 5xx/429 surfaced
  by mcp as MCPError) are retried; tool errors are final, missing data is never guessed;
- provenance: evidence_ref values are stored exactly as returned and every consumed result
  is recorded with a ``tool_result_consumed`` trace event.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp.shared.exceptions import MCPError

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

logger = logging.getLogger(__name__)

TOOL_PERMISSIONS: dict[str, frozenset[str]] = {
    "coordinator": frozenset({"get_policy"}),
    "entity-agent": frozenset({"get_customer_history", "get_order"}),
    "order-agent": frozenset({"get_order", "get_order_items"}),
    "payment-agent": frozenset(
        {"get_payment_timeline", "get_refund_timeline", "get_order_payments"}
    ),
    "shipment-agent": frozenset({"get_shipment_summary"}),
    "policy-agent": frozenset(),
    "verifier-agent": frozenset(),
}

MAX_TRANSIENT_RETRIES = 2
MAX_TIMEOUT_RETRIES = 1  # a timed-out call was probably already audited server-side
RETRY_BACKOFF_SECONDS = (2.0, 5.0)  # generous: the gateway is rate limited
# mcp_types.jsonrpc codes: CONNECTION_CLOSED, REQUEST_TIMEOUT, INTERNAL_ERROR (HTTP >= 400).
TRANSIENT_MCP_CODES = frozenset({-32000, -32001, -32603})
RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "too many requests", "429")


@dataclass(frozen=True)
class Evidence:
    tool: str
    domain: str
    ref: str
    result_hash: str
    data: Any
    warnings: tuple[str, ...]


@dataclass
class ToolFailure:
    actor: str
    tool: str
    reason: str
    attempts: int

    @property
    def transient(self) -> bool:
        """The data may exist: the call failed for infrastructure reasons, not a tool answer."""
        return self.reason.startswith(("transient:", "timeout", "unexpected:"))


@dataclass
class EvidenceLedger:
    case_id: str
    gateway: EvidenceGateway
    trace: TraceWriter
    available_tools: frozenset[str] | None = None
    calls: int = 0
    failures: list[ToolFailure] = field(default_factory=list)
    _cache: dict[tuple[str, tuple[tuple[str, str], ...]], Evidence | None] = field(
        default_factory=dict
    )
    _locks: dict[tuple[str, tuple[tuple[str, str], ...]], asyncio.Lock] = field(
        default_factory=dict
    )
    _by_tool: dict[str, Evidence] = field(default_factory=dict)
    _consumed_refs: set[str] = field(default_factory=set)

    def evidence(self, tool: str) -> Evidence | None:
        return self._by_tool.get(tool)

    def ref(self, tool: str) -> str | None:
        found = self._by_tool.get(tool)
        return found.ref if found else None

    @property
    def consumed_refs(self) -> frozenset[str]:
        return frozenset(self._consumed_refs)

    def failed_transiently(self, tool: str) -> bool:
        return self.ref(tool) is None and any(
            failure.tool == tool and failure.transient for failure in self.failures
        )

    async def fetch(self, actor: str, tool: str, **arguments: str) -> Evidence | None:
        if tool not in TOOL_PERMISSIONS.get(actor, frozenset()):
            raise PermissionError(f"{actor} is not permitted to call {tool}")
        if self.available_tools is not None and tool not in self.available_tools:
            self.failures.append(ToolFailure(actor, tool, "tool_not_discovered", 0))
            return None
        key = (tool, tuple(sorted(arguments.items())))
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._cache:
                cached = self._cache[key]
                if cached is not None:
                    self._emit_consumed(actor, cached, cached_hit=True)
                return cached
            result = await self._call_with_retry(actor, tool, arguments)
            self._cache[key] = result
            if result is not None:
                self._by_tool.setdefault(tool, result)
                self._emit_consumed(actor, result, cached_hit=False)
            return result

    async def _call_with_retry(
        self, actor: str, tool: str, arguments: dict[str, str]
    ) -> Evidence | None:
        attempts = 0
        while True:
            attempts += 1
            self.calls += 1
            try:
                envelope = await self.gateway.call(tool, case_id=self.case_id, **arguments)
            except TimeoutError:
                if attempts > MAX_TIMEOUT_RETRIES:
                    self.failures.append(ToolFailure(actor, tool, "timeout", attempts))
                    return None
                await asyncio.sleep(_backoff(attempts))
                continue
            except MCPError as exc:
                if not _is_transient_mcp_error(exc):
                    self.failures.append(ToolFailure(actor, tool, f"mcp:{exc.code}", attempts))
                    return None
                if attempts > MAX_TRANSIENT_RETRIES:
                    self.failures.append(
                        ToolFailure(actor, tool, f"transient:mcp:{exc.code}", attempts)
                    )
                    return None
                logger.info("%s %s transient MCP error, retrying: %s", self.case_id, tool, exc)
                await asyncio.sleep(_backoff(attempts))
                continue
            except (RuntimeError, ValueError) as exc:
                # Tool-level error (e.g. no refund lifecycle for this order) or contract
                # violation: deterministic, so retrying would only waste audited calls.
                reason = "tool_error" if isinstance(exc, RuntimeError) else "contract_violation"
                self.failures.append(ToolFailure(actor, tool, reason, attempts))
                return None
            except Exception as exc:  # noqa: BLE001 - never let one tool break the case
                logger.warning("%s %s unexpected MCP failure: %s", self.case_id, tool, exc)
                self.failures.append(
                    ToolFailure(actor, tool, f"unexpected:{type(exc).__name__}", attempts)
                )
                return None
            evidence = self._accept(tool, arguments, envelope)
            if evidence is None:
                self.failures.append(ToolFailure(actor, tool, "scope_mismatch", attempts))
            return evidence

    def _accept(
        self, tool: str, arguments: dict[str, str], envelope: dict[str, Any]
    ) -> Evidence | None:
        data = envelope.get("data")
        if not _in_scope(data, arguments):
            return None
        return Evidence(
            tool=tool,
            domain=str(envelope["domain"]),
            ref=str(envelope["evidence_ref"]),
            result_hash=str(envelope["result_hash"]),
            data=data,
            warnings=tuple(envelope.get("warnings") or ()),
        )

    def _emit_consumed(self, actor: str, evidence: Evidence, *, cached_hit: bool) -> None:
        self._consumed_refs.add(evidence.ref)
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=evidence.tool,
            evidence_refs=[evidence.ref],
            attributes={
                "domain": evidence.domain,
                "result_hash": evidence.result_hash,
                "cache_hit": cached_hit,
                "warning_count": len(evidence.warnings),
            },
        )


def _backoff(attempt: int) -> float:
    return RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]


def _is_transient_mcp_error(exc: MCPError) -> bool:
    message = str(getattr(exc, "message", "") or exc).lower()
    return exc.code in TRANSIENT_MCP_CODES or any(m in message for m in RATE_LIMIT_MARKERS)


def _in_scope(data: Any, arguments: dict[str, str]) -> bool:
    """Reject evidence whose embedded identifiers do not match what was requested."""
    order_id = arguments.get("order_id")
    customer = arguments.get("customer_unique_id")
    rows: list[Any] = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        if order_id and "order_id" in row and row["order_id"] != order_id:
            return False
        if customer and "customer_unique_id" in row and row["customer_unique_id"] != customer:
            return False
    return True
