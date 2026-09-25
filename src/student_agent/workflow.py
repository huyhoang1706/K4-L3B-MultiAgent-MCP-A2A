"""L3B coordinator + specialist multi-agent workflow (deterministic flow).

Code decides the order, the LLM never chooses it:

    coordinator ─► entity-agent ─► coordinator ─┬─► order-agent ────┐
         │  (get_policy)                        ├─► payment-agent ──┼─► policy-agent ─► verifier
         │                                      └─► shipment-agent ─┘
         └──────────────────────── trace (observable events only) ─────────────────────────

Specialists gather MCP evidence through the case-scoped ledger (least privilege) and produce
rule-based findings; their Agents SDK analysts (gpt-6-luna) give an independent structured
reading of the same evidence. The policy agent applies the MCP policy rules; the verifier
checks invariants, compares the independent readings and calibrates confidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .analysis import (
    REFUND_TOPICS,
    EntityResolution,
    OrderFinding,
    PaymentFinding,
    PolicyDecision,
    ShipmentFinding,
    Timeline,
    analyze_order,
    analyze_payment,
    analyze_shipment,
    build_timeline,
    claim_verdict,
    data_conflicts,
    decide,
    entities_from_shipping_limits,
    is_duplicate_capture,
    parse_ts,
    resolve_entity,
    same_money,
    shipment_row_vs_anchor,
)
from .evidence import Evidence, EvidenceLedger
from .llm_agents import agents_trace, get_analysts
from .mcp_gateway import EvidenceGateway
from .policy_table import KNOWN_POLICIES
from .trace import TraceWriter

COORDINATOR = "coordinator"
ENTITY = "entity-agent"
ORDER = "order-agent"
PAYMENT = "payment-agent"
SHIPMENT = "shipment-agent"
POLICY = "policy-agent"
VERIFIER = "verifier-agent"

HEX_ORDER_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_CANDIDATE_LOOKUPS = 2

BASE_EVIDENCE_TOOLS = ("get_customer_history", "get_order", "get_policy")
# Evidence cited per primary issue (besides history, order and policy). Public feedback from
# a full run vs a 2-call run fits required groups {order, item, payment, policy} (+ shipment for
# late delivery, + refund for refund issues), so the item and payment groups are cited for all.
ISSUE_EVIDENCE_TOOLS: dict[str, tuple[str, ...]] = {
    "late_delivery_seller": ("get_shipment_summary", "get_order_items", "get_payment_timeline"),
    "late_delivery_logistics": ("get_shipment_summary", "get_order_items", "get_payment_timeline"),
    "canceled_order_paid": ("get_payment_timeline", "get_order_items"),
    "unavailable_order_paid": ("get_payment_timeline", "get_order_items"),
    "valid_split_payment": ("get_payment_timeline", "get_order_items"),
    "duplicate_charge": ("get_payment_timeline", "get_order_items"),
    "payment_mismatch": ("get_payment_timeline", "get_order_items"),
    "refund_pending": ("get_refund_timeline", "get_payment_timeline", "get_order_items"),
    "refund_failed": ("get_refund_timeline", "get_payment_timeline", "get_order_items"),
    "unsupported_claim": ("get_shipment_summary", "get_payment_timeline", "get_order_items"),
}
REFUND_CLAIM_TOOLS = ("get_payment_timeline", "get_refund_timeline", "get_policy")
# Claims whose conclusion cites the shipment summary. For every other claim the delivery
# verdict comes from history dates + item shipping limits, so the (audited) call is skipped.
SHIPMENT_TOPICS = frozenset(
    {"late_delivery_seller", "late_delivery_logistics", "unsupported_claim"}
)

# Lean call plan: at most 2 audited MCP calls per case. The entity agent always spends one on
# get_customer_history (entity resolution, customer context, anchor instance); the coordinator
# routes the second to the single most decisive source for the claimed issue. The public
# policy is applied from policy_table.KNOWN_POLICIES instead of a per-case get_policy call.
LEAN_SECOND_TOOL: dict[str, str] = {
    "late_delivery_seller": "get_shipment_summary",
    "late_delivery_logistics": "get_shipment_summary",
    "canceled_order_paid": "get_payment_timeline",
    "unavailable_order_paid": "get_payment_timeline",
    "valid_split_payment": "get_payment_timeline",
    "duplicate_charge": "get_payment_timeline",
    "payment_mismatch": "get_payment_timeline",
    "refund_pending": "get_refund_timeline",
    "refund_failed": "get_refund_timeline",
    "unsupported_claim": "get_payment_timeline",
}
LEAN_DEFAULT_SECOND_TOOL = "get_payment_timeline"


def call_plan(claim_topic: str | None) -> frozenset[str]:
    """Order-scoped tools (besides get_customer_history) the specialists may call."""
    if os.getenv("DAY09_CALL_PLAN", "full").strip().lower() != "lean":
        tools = {"get_order", "get_order_items", "get_payment_timeline", "get_policy"}
        if claim_topic is None or claim_topic in SHIPMENT_TOPICS:
            tools.add("get_shipment_summary")
        if claim_topic in REFUND_TOPICS:
            tools.add("get_refund_timeline")
        return frozenset(tools)
    overrides = json.loads(os.getenv("DAY09_LEAN_PLAN_JSON", "{}") or "{}")  # dev experiments
    second = overrides.get(claim_topic) or LEAN_SECOND_TOOL.get(
        claim_topic or "", LEAN_DEFAULT_SECOND_TOOL
    )
    return frozenset([second] if isinstance(second, str) else second)


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    case_id = case["case_id"]
    request = case.get("customer_request") if isinstance(case.get("customer_request"), dict) else {}
    claims = [claim for claim in request.get("claims") or [] if isinstance(claim, dict)]
    claim_topic = next(
        (
            c.get("topic")
            for c in claims
            if c.get("topic") and c.get("topic") != "requested_full_refund"
        ),
        None,
    )
    opened_at = parse_ts(case.get("opened_at")) or datetime.now(UTC)
    ledger = EvidenceLedger(
        case_id, gateway, trace, available_tools=frozenset(await gateway.list_tools())
    )
    analysts = get_analysts()

    # 1. Coordinator -> entity agent: resolve which order the complaint is about.
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor=COORDINATOR,
        target=ENTITY,
        decision_code="resolve_entity",
        attributes={"candidate_count": len(case.get("candidate_order_ids") or [])},
    )
    entity, history = await _entity_agent(case, request, ledger)
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=ENTITY,
        target=COORDINATOR,
        decision_code=f"entity_{entity.status}",
        attributes={
            "resolved_count": len(entity.resolved_order_ids),
            "rejected_count": len(entity.rejected_candidates),
            "customer_verified": entity.customer_unique_id is not None,
        },
    )

    order_id = entity.resolved_order_ids[0] if entity.status == "resolved" else None
    history_rows = [
        row
        for row in (
            history.data.get("orders") or [] if history and isinstance(history.data, dict) else []
        )
        if isinstance(row, dict) and row.get("order_id") == order_id
    ]

    # 2. Coordinator plans the calls, fans out to specialists (parallel) and loads the policy.
    plan = call_plan(claim_topic)
    policy_version = str(case.get("policy_version") or "")
    policy_task = asyncio.create_task(_policy_intake(ledger, policy_version, plan))
    if order_id is None:
        policy_ev = await policy_task
        return await _finalize_unresolved(
            case_id, claims, entity, policy_ev, policy_version, ledger, trace
        )

    for actor, code in (
        (ORDER, "investigate_order_items"),
        (PAYMENT, "investigate_payment_refund"),
        (SHIPMENT, "investigate_shipment"),
    ):
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor=COORDINATOR,
            target=actor,
            decision_code=code,
            attributes={"order_id": order_id, "planned_calls": len(plan)},
        )
    fetch_shipment = "get_shipment_summary" in plan
    (order_row, items), (payment_tl, refund_tl), shipment, policy_ev = await asyncio.gather(
        _order_agent(ledger, order_id, plan),
        _payment_agent(ledger, order_id, plan),
        _shipment_agent(ledger, order_id, fetch_shipment),
        policy_task,
    )

    # 3. Specialist findings: rule-based reading + independent LLM reading of the same evidence.
    rows = history_rows or (
        [order_row.data] if order_row and isinstance(order_row.data, dict) else []
    )
    timeline = build_timeline(rows, opened_at)
    order_f = analyze_order(order_id, timeline, _data(order_row), _data(items))
    if not order_f.item_ids and shipment is not None:
        order_f.item_ids, order_f.seller_ids = entities_from_shipping_limits(
            timeline, _data(shipment)
        )
    if order_row is None and shipment is not None:
        # The shipment summary carries the same order row as get_order: detect the conflict.
        order_f.order_row_is_anchor, order_f.order_row_status = shipment_row_vs_anchor(
            timeline, _data(shipment)
        )
        order_f.order_row_source = "get_shipment_summary"
    payment_f = analyze_payment(timeline, _data(payment_tl), _data(refund_tl))
    shipment_f = analyze_shipment(
        timeline, order_f, _data(shipment), summary_required=fetch_shipment
    )

    with agents_trace(case_id, enabled=analysts.enabled):
        readings = await asyncio.gather(
            analysts.read("order", _order_packet(opened_at, order_id, rows, order_row)),
            analysts.read(
                "payment", _payment_packet(opened_at, rows, items, payment_tl, refund_tl)
            ),
            analysts.read("shipment", _shipment_packet(opened_at, rows, items, shipment)),
        )
    cross = {
        "order": _compare_order(readings[0], timeline),
        "payment": _compare_payment(readings[1], timeline, payment_f, order_f),
        "shipment": _compare_shipment(readings[2], timeline, shipment_f),
    }
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=ORDER,
        target=POLICY,
        decision_code="order_finding_ready",
        evidence_refs=_refs(ledger, "get_order", "get_order_items") or None,
        attributes={
            "anchor_status": order_f.anchor_status,
            "instance_count": order_f.instance_count,
            "tied_instances": order_f.tied,
            "llm_cross_check": cross["order"],
        },
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=PAYMENT,
        target=POLICY,
        decision_code="payment_finding_ready",
        evidence_refs=_refs(ledger, "get_payment_timeline", "get_refund_timeline") or None,
        attributes={
            "captured_total_brl": payment_f.captured_total,
            "capture_count": len(payment_f.captures),
            "refund_events": len(payment_f.refunds),
            "llm_cross_check": cross["payment"],
        },
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=SHIPMENT,
        target=POLICY,
        decision_code=f"shipment_{shipment_f.verdict}",
        evidence_refs=_refs(ledger, "get_shipment_summary") or None,
        attributes={
            "timeline_complete": shipment_f.timeline_complete,
            "summary_fetched": fetch_shipment,
            "llm_cross_check": cross["shipment"],
        },
    )

    # 4. Policy agent: rule-based decision from findings + MCP policy (no MCP calls).
    policy_data = _data(policy_ev) or KNOWN_POLICIES.get(policy_version)
    decision = decide(claim_topic, order_f, payment_f, shipment_f, policy_data)
    if (
        "get_policy" in plan  # full plan only: the lean plan never exceeds its 2 calls
        and not fetch_shipment
        and "get_shipment_summary" in ISSUE_EVIDENCE_TOOLS.get(decision.primary_issue, ())
    ):
        # The decision ended up delivery-related: fetch the summary it must cite.
        await ledger.fetch(SHIPMENT, "get_shipment_summary", order_id=order_id)
    output = _build_output(
        case_id, claims, entity, order_f, payment_f, shipment_f, decision, ledger
    )
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor=POLICY,
        decision_code=decision.primary_issue,
        evidence_refs=output["evidence_refs"][:20],
        attributes={
            "case_status": decision.case_status,
            "action": decision.recommended_action,
            "refund_brl": decision.refund_brl,
            "claim_tiebreak": decision.claim_tiebreak,
            "evidenced_issue_count": len(decision.evidenced_issues),
        },
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=POLICY,
        target=VERIFIER,
        decision_code="verify_decision",
    )

    # 5. Verifier: invariants + independent readings -> calibrated confidence.
    failed = _verify(output, ledger, trace, decision, shipment_f)
    disagreements = sum(1 for value in cross.values() if value == "disagree")
    confidence = _calibrate(
        claim_topic,
        entity,
        order_f,
        payment_f,
        shipment_f,
        decision,
        ledger,
        disagreements,
        len(failed),
        history_used=bool(history_rows),
        planned_tools=plan,
    )
    output["assessment"]["confidence"] = confidence
    for claim in output.get("claim_assessments", []):
        claim["confidence"] = round(min(confidence, 0.95), 2)
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor=VERIFIER,
        decision_code="passed" if not failed else "passed_with_warnings",
        attributes={
            "confidence": confidence,
            "failed_checks": len(failed),
            "first_failed_check": failed[0] if failed else None,
            "llm_disagreements": disagreements,
            "llm_checks": sum(1 for v in cross.values() if v != "skipped"),
            "mcp_calls": ledger.calls,
            "mcp_failures": len(ledger.failures),
        },
    )
    _debug_dump(
        case_id,
        entity=entity,
        order=order_f,
        payment=payment_f,
        shipment=shipment_f,
        decision=decision,
        cross=cross,
        readings=readings,
        failed=failed,
        ledger=ledger,
    )
    return output


# --------------------------------------------------------------------------- specialists


async def _entity_agent(
    case: dict[str, Any], request: dict[str, Any], ledger: EvidenceLedger
) -> tuple[EntityResolution, Evidence | None]:
    candidates = [str(c) for c in case.get("candidate_order_ids") or [] if c]
    claimed = request.get("claimed_order_id")
    hint = case.get("customer_unique_id_hint")
    history = (
        await ledger.fetch(ENTITY, "get_customer_history", customer_unique_id=str(hint))
        if hint
        else None
    )
    entity = resolve_entity(candidates, claimed, _data(history))
    if entity.status == "not_found":
        # Bounded fallback: verify plausible order ids directly, claimed id first.
        verified: set[str] = set()
        lookups = [
            c for c in dict.fromkeys([claimed, *candidates]) if c and HEX_ORDER_ID.fullmatch(c)
        ]
        for candidate in lookups[:MAX_CANDIDATE_LOOKUPS]:
            if await ledger.fetch(ENTITY, "get_order", order_id=candidate):
                verified.add(candidate)
                break
        entity = resolve_entity(candidates, claimed, _data(history), verified)
    return entity, history


async def _policy_intake(
    ledger: EvidenceLedger, policy_version: str, plan: frozenset[str]
) -> Evidence | None:
    if "get_policy" not in plan:
        return None  # lean plan: the public policy table is applied locally
    return await ledger.fetch(COORDINATOR, "get_policy", policy_version=policy_version)


async def _optional(
    ledger: EvidenceLedger, actor: str, tool: str, plan: frozenset[str], **arguments: str
) -> Evidence | None:
    return await ledger.fetch(actor, tool, **arguments) if tool in plan else None


async def _order_agent(
    ledger: EvidenceLedger, order_id: str, plan: frozenset[str]
) -> tuple[Evidence | None, Evidence | None]:
    order_row, items = await asyncio.gather(
        _optional(ledger, ORDER, "get_order", plan, order_id=order_id),
        _optional(ledger, ORDER, "get_order_items", plan, order_id=order_id),
    )
    return order_row, items


async def _shipment_agent(
    ledger: EvidenceLedger, order_id: str, fetch_summary: bool
) -> Evidence | None:
    if not fetch_summary:
        return None
    return await ledger.fetch(SHIPMENT, "get_shipment_summary", order_id=order_id)


async def _payment_agent(
    ledger: EvidenceLedger, order_id: str, plan: frozenset[str]
) -> tuple[Evidence | None, Evidence | None]:
    # The refund lifecycle is only planned for refund claims: the tool errors (and still counts
    # as an audited call) for orders without refund events.
    payment_tl, refund_tl = await asyncio.gather(
        _optional(ledger, PAYMENT, "get_payment_timeline", plan, order_id=order_id),
        _optional(ledger, PAYMENT, "get_refund_timeline", plan, order_id=order_id),
    )
    return payment_tl, refund_tl


# ------------------------------------------------------------------- LLM packets & comparison


def _instances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "order_purchase_timestamp",
        "order_status",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def _order_packet(
    opened_at: datetime, order_id: str, rows: list[dict[str, Any]], order_row: Evidence | None
) -> dict[str, Any]:
    return {
        "opened_at": opened_at.isoformat(),
        "order_id": order_id,
        "instances": _instances(rows),
        "get_order_row": _data(order_row),
    }


def _payment_packet(
    opened_at: datetime,
    rows: list[dict[str, Any]],
    items: Evidence | None,
    payment_tl: Evidence | None,
    refund_tl: Evidence | None,
) -> dict[str, Any]:
    item_rows = [
        {k: row.get(k) for k in ("shipping_limit_date", "price", "freight_value")}
        for row in (_data(items) or [])
        if isinstance(row, dict)
    ]
    timeline = _data(payment_tl) or {}
    return {
        "opened_at": opened_at.isoformat(),
        "instances": _instances(rows),
        "items": item_rows,
        "payment_events": timeline.get("events") if isinstance(timeline, dict) else None,
        "refund_events": (_data(refund_tl) or {}).get("events") if refund_tl else None,
    }


def _shipment_packet(
    opened_at: datetime,
    rows: list[dict[str, Any]],
    items: Evidence | None,
    shipment: Evidence | None,
) -> dict[str, Any]:
    data = _data(shipment) or {}
    item_limits = [
        {k: row.get(k) for k in ("order_item_id", "seller_id", "shipping_limit_date")}
        for row in (_data(items) or [])
        if isinstance(row, dict)
    ]
    return {
        "opened_at": opened_at.isoformat(),
        "instances": _instances(rows),
        "item_shipping_limits": item_limits,
        "shipping_limits": data.get("shipping_limits"),
        "shipment_events": data.get("events"),
    }


def _same_instance(reading_ts: str, timeline: Timeline) -> bool:
    parsed = parse_ts(reading_ts)
    return bool(timeline.anchor and parsed and parsed == timeline.anchor.purchase_at)


def _compare_order(reading: Any, timeline: Timeline) -> str:
    if reading is None:
        return "skipped"
    agree = _same_instance(reading.relevant_purchase_at, timeline) and (
        timeline.anchor is not None
        and reading.relevant_status.strip().lower() == timeline.anchor.status
    )
    return "agree" if agree else "disagree"


def _specialist_payment_verdicts(payment: PaymentFinding, order: OrderFinding) -> set[str]:
    """Verdicts the evidence supports (more than one when the evidence is ambiguous)."""
    statuses = {status for status, _ in payment.refunds}
    if "failed" in statuses:
        return {"refund_failed"}
    if "pending" in statuses:
        return {"refund_pending"}
    if not payment.has_payment_evidence:
        return {"insufficient_evidence"}
    if payment.mismatch_amounts:
        return {"capture_mismatch"}
    equal_captures = any(payment.captures.count(amount) > 1 for amount in payment.captures)
    if order.items_total is None and len(payment.captures) >= 2 and equal_captures:
        return {"reconciled", "duplicate_capture"}  # split vs duplicate needs the order total
    if is_duplicate_capture(payment.captures, order.items_total):
        return {"duplicate_capture"}
    if payment.refunded_total > 0:
        return {"refunded"}
    return {"reconciled"} if payment.captures else {"insufficient_evidence"}


def _compare_payment(
    reading: Any, timeline: Timeline, payment: PaymentFinding, order: OrderFinding
) -> str:
    if reading is None:
        return "skipped"
    agree = (
        _same_instance(reading.relevant_purchase_at, timeline)
        and reading.verdict in _specialist_payment_verdicts(payment, order)
        and (
            payment.captured_total is None
            or same_money(reading.captured_total_brl, payment.captured_total)
        )
    )
    return "agree" if agree else "disagree"


def _compare_shipment(reading: Any, timeline: Timeline, shipment: ShipmentFinding) -> str:
    if reading is None:
        return "skipped"
    agree = (
        _same_instance(reading.relevant_purchase_at, timeline)
        and reading.verdict == shipment.verdict
    )
    return "agree" if agree else "disagree"


# --------------------------------------------------------------------------- output


def _build_output(
    case_id: str,
    claims: list[dict[str, Any]],
    entity: EntityResolution,
    order: OrderFinding,
    payment: PaymentFinding,
    shipment: ShipmentFinding,
    decision: PolicyDecision,
    ledger: EvidenceLedger,
) -> dict[str, Any]:
    issue_tools = ISSUE_EVIDENCE_TOOLS.get(decision.primary_issue)
    if issue_tools is None:  # insufficient evidence: cite everything that was consumed
        issue_refs = sorted(ledger.consumed_refs)
    else:
        issue_refs = _refs(ledger, "get_customer_history", *issue_tools)
    claim_assessments = []
    for claim in claims[:5]:
        topic = str(claim.get("topic") or "")
        refs = (
            (_refs(ledger, *REFUND_CLAIM_TOOLS) or issue_refs)
            if topic == "requested_full_refund"
            else issue_refs
        )
        claim_assessments.append(
            {
                "claim_id": str(claim.get("claim_id") or f"claim-{len(claim_assessments) + 1}")[
                    :64
                ],
                "verdict": claim_verdict(topic, decision),
                "confidence": 0.0,  # set by the verifier
                "evidence_refs": refs[:30],
            }
        )
    evidence_refs = list(
        dict.fromkeys(
            [
                *_refs(ledger, *BASE_EVIDENCE_TOOLS, *(issue_tools or ())),
                *(issue_refs if issue_tools is None else []),
                *(ref for claim in claim_assessments for ref in claim["evidence_refs"]),
            ]
        )
    )[:30]

    refund = decision.refund_brl
    refund_lines = (
        [
            {
                "reason_code": decision.recommended_action,
                "amount_brl": refund,
                "entity_id": order.order_id,
            }
        ]
        if refund > 0
        else []
    )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "assessment": {
            "primary_issue": decision.primary_issue,
            "secondary_issues": [],
            "case_status": decision.case_status,
            "confidence": 0.0,  # set by the verifier
        },
        "affected_entities": {
            "order_ids": entity.resolved_order_ids,
            "item_ids": order.item_ids,
            "seller_ids": order.seller_ids,
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": claim_assessments,
        "entity_resolution": {
            "status": entity.status,
            "resolved_order_ids": entity.resolved_order_ids,
            "rejected_candidates": entity.rejected_candidates,
            "confidence": entity.confidence,
        },
        "customer_context": {
            "customer_unique_id": entity.customer_unique_id,
            "related_order_ids": entity.related_order_ids,
        },
        "shipment_analysis": {
            "verdict": shipment.verdict,
            "late_seller_ids": shipment.late_seller_ids,
            "timeline_complete": shipment.timeline_complete,
        },
        "payment_analysis": {
            "verdict": decision.payment_verdict,
            "captured_total_brl": decision.captured_total,
            "refunded_total_brl": decision.refunded_total,
            "refundable_total_brl": decision.refundable_total,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": decision.primary_issue.upper(), "rank": 1}],
            "responsible_parties": decision.responsible_parties,
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": data_conflicts(order, payment, shipment, decision),
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": refund,
            "refund_lines": refund_lines,
        },
        "resolution_actions": [decision.recommended_action],
    }


async def _finalize_unresolved(
    case_id: str,
    claims: list[dict[str, Any]],
    entity: EntityResolution,
    policy_ev: Evidence | None,
    policy_version: str,
    ledger: EvidenceLedger,
    trace: TraceWriter,
) -> dict[str, Any]:
    """No order could be resolved: report insufficient evidence instead of guessing."""
    order = OrderFinding("", None, None, False, 0, [], [], None, None, None, None, None)
    payment = PaymentFinding(False, False, [], [], [], 0.0)
    shipment = ShipmentFinding(False, "insufficient_evidence", None, None, [], False, None, False)
    decision = decide(
        None, order, payment, shipment, _data(policy_ev) or KNOWN_POLICIES.get(policy_version)
    )
    output = _build_output(case_id, claims, entity, order, payment, shipment, decision, ledger)
    output["financial_resolution"]["refund_lines"] = []
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor=POLICY,
        decision_code=decision.primary_issue,
        evidence_refs=output["evidence_refs"][:20] or None,
        attributes={"case_status": decision.case_status, "entity_status": entity.status},
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=POLICY,
        target=VERIFIER,
        decision_code="verify_decision",
    )
    failed = _verify(output, ledger, trace, decision, shipment)
    confidence = 0.2
    output["assessment"]["confidence"] = confidence
    for claim in output.get("claim_assessments", []):
        claim["confidence"] = confidence
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor=VERIFIER,
        decision_code="insufficient_evidence",
        attributes={
            "confidence": confidence,
            "failed_checks": len(failed),
            "mcp_calls": ledger.calls,
            "mcp_failures": len(ledger.failures),
        },
    )
    return output


# --------------------------------------------------------------------------- verifier


def _verify(
    output: dict[str, Any],
    ledger: EvidenceLedger,
    trace: TraceWriter,
    decision: PolicyDecision,
    shipment: ShipmentFinding,
) -> list[str]:
    """Return the codes of failed invariants (empty when everything holds)."""
    failed: list[str] = []
    try:
        trace.contracts.validate_output(output, f"outputs/{output['case_id']}.json")
    except ValueError:
        failed.append("schema")
    refs = set(output["evidence_refs"])
    if not refs <= ledger.consumed_refs:
        failed.append("evidence_scope")
    if not refs:
        failed.append("evidence_missing")
    for claim in output.get("claim_assessments", []):
        if not set(claim["evidence_refs"]) <= refs:
            failed.append("claim_evidence_linkage")
            break
    resolution = output["entity_resolution"]
    if set(resolution["resolved_order_ids"]) & set(resolution["rejected_candidates"]):
        failed.append("entity_scope")
    financial = output["financial_resolution"]
    lines_total = round(sum(line["amount_brl"] for line in financial["refund_lines"]), 2)
    if not same_money(lines_total, financial["recommended_refund_brl"]):
        failed.append("refund_lines_total")
    refundable = output["payment_analysis"]["refundable_total_brl"]
    if refundable is not None and financial["recommended_refund_brl"] > refundable + 0.01:
        failed.append("refund_exceeds_refundable")
    status = output["assessment"]["case_status"]
    if status == "no_action" and financial["recommended_refund_brl"] > 0:
        failed.append("no_action_with_refund")
    if financial["recommended_refund_brl"] > 0 and status != "action_required":
        failed.append("refund_without_action")
    parties = output["root_cause_analysis"]["responsible_parties"]
    sellers = set(output["affected_entities"]["seller_ids"])
    if any(p["party_type"] == "seller" and p["party_id"] not in sellers for p in parties):
        failed.append("seller_party_not_affected")
    issue = decision.primary_issue
    party_types = {p["party_type"] for p in parties}
    if issue == "late_delivery_seller" and (
        shipment.verdict != "seller_delay" or "seller" not in party_types
    ):
        failed.append("seller_delay_consistency")
    if issue == "late_delivery_logistics" and (
        shipment.verdict != "logistics_delay" or "seller" in party_types
    ):
        failed.append("logistics_delay_consistency")
    if len(output["resolution_actions"]) != len(set(output["resolution_actions"])):
        failed.append("duplicate_actions")
    return failed


def _calibrate(
    claim_topic: str | None,
    entity: EntityResolution,
    order: OrderFinding,
    payment: PaymentFinding,
    shipment: ShipmentFinding,
    decision: PolicyDecision,
    ledger: EvidenceLedger,
    llm_disagreements: int,
    failed_checks: int,
    *,
    history_used: bool = True,
    planned_tools: frozenset[str] = frozenset(),
) -> float:
    confidence = 0.93
    if decision.primary_issue == "insufficient_evidence":
        confidence = 0.35
    if decision.claim_tiebreak:
        confidence -= 0.2
    elif "payment_pattern_ambiguous" in decision.notes:
        confidence -= 0.08
    elif order.tied:
        confidence -= 0.05
    if claim_topic and decision.primary_issue not in {
        claim_topic,
        "unsupported_claim",
        "insufficient_evidence",
    }:
        confidence -= 0.1  # evidence points to a different fault than the one claimed
    if shipment.event_conflict:
        confidence -= 0.1
    if decision.refund_capped:
        confidence -= 0.05
    missing = [tool for tool in sorted(planned_tools) if ledger.ref(tool) is None]
    confidence -= 0.12 * len(missing)
    if entity.status != "resolved" or entity.customer_unique_id is None:
        confidence -= 0.1
    confidence -= 0.07 * llm_disagreements
    confidence -= 0.1 * failed_checks
    if not history_used:
        # A single get_order row cannot show whether another (observable) instance exists.
        confidence = min(confidence, 0.4)
    return round(min(max(confidence, 0.05), 0.97), 2)


# --------------------------------------------------------------------------- helpers


def _data(evidence: Evidence | None) -> Any:
    return evidence.data if evidence is not None else None


def _refs(ledger: EvidenceLedger, *tools: str) -> list[str]:
    return list(dict.fromkeys(ref for tool in tools if (ref := ledger.ref(tool))))


def _debug_dump(case_id: str, **parts: Any) -> None:
    """Optional local debug record (DAY09_DEBUG_DIR); never part of the submission."""
    directory = os.getenv("DAY09_DEBUG_DIR")
    if not directory:
        return
    ledger: EvidenceLedger = parts.pop("ledger")
    readings = parts.pop("readings")
    record = {
        key: (asdict(value) if hasattr(value, "__dataclass_fields__") else value)
        for key, value in parts.items()
    }
    record["readings"] = [r.model_dump() if r is not None else None for r in readings]
    record["mcp_calls"] = ledger.calls
    record["mcp_failures"] = [asdict(f) for f in ledger.failures]
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{case_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
