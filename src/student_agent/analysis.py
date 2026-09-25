"""Deterministic evidence analysis for the specialist, policy and verifier agents.

Every function here is pure: it takes MCP evidence payloads (already validated against the
evidence envelope contract) and returns typed findings. Nothing is fetched, invented or
guessed; missing evidence stays missing and lowers confidence instead.

Key domain rule: one order id can carry several purchase *instances* (history rows with
different purchase timestamps). The complaint concerns the anchor instance: the latest one
purchased on or before ``opened_at`` whose outcome was observable when the case was opened.
Events are attributed to instances by timestamp, amount or delivery date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any

MONEY_TOLERANCE = 0.01
UNSHIPPED_TERMINAL_STATUSES = frozenset({"canceled", "unavailable"})
COMPLETED_REFUND_STATUSES = frozenset({"succeeded", "completed", "refunded", "processed"})
VOID_CAPTURE_STATUSES = frozenset({"failed", "voided", "reversed", "canceled"})

REFUND_TOPICS = frozenset({"refund_pending", "refund_failed"})
PAYMENT_TOPICS = frozenset({"valid_split_payment", "duplicate_charge", "payment_mismatch"})

# Evidence-backed issues in decreasing severity. `valid_split_payment` explains a payment
# pattern rather than a fault, so it only wins when the claim is about payments.
ISSUE_PRIORITY = (
    "canceled_order_paid",
    "unavailable_order_paid",
    "refund_failed",
    "refund_pending",
    "payment_mismatch",
    "duplicate_charge",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
)
NON_FAULT_ISSUES = frozenset({"valid_split_payment"})


def parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def money(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def same_money(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and abs(left - right) <= MONEY_TOLERANCE


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


# --------------------------------------------------------------------------- instances


@dataclass(frozen=True)
class OrderInstance:
    purchase_at: datetime
    status: str
    carrier_at: datetime | None
    delivered_at: datetime | None
    estimated_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> OrderInstance | None:
        purchase = parse_ts(row.get("order_purchase_timestamp"))
        if purchase is None:
            return None
        return cls(
            purchase_at=purchase,
            status=str(row.get("order_status") or "unknown"),
            carrier_at=parse_ts(row.get("order_delivered_carrier_date")),
            delivered_at=parse_ts(row.get("order_delivered_customer_date")),
            estimated_at=parse_ts(row.get("order_estimated_delivery_date")),
        )

    def observable_at(self, opened_at: datetime) -> bool:
        """True when the instance's outcome could be complained about at ``opened_at``."""
        if self.purchase_at > opened_at:
            return False
        if self.status in UNSHIPPED_TERMINAL_STATUSES:
            return True
        return any(
            ts is not None and ts <= opened_at for ts in (self.delivered_at, self.estimated_at)
        )


@dataclass
class Timeline:
    instances: list[OrderInstance]
    anchor: OrderInstance | None
    tied: bool  # another history row shares the anchor's purchase timestamp

    def owner(self, ts: datetime | None) -> datetime | None:
        """Purchase timestamp of the instance owning an event: latest purchase not after ts."""
        if ts is None:
            return None
        owners = [inst.purchase_at for inst in self.instances if inst.purchase_at <= ts]
        return max(owners) if owners else None

    def owned_by_anchor(self, ts: datetime | None) -> bool:
        return self.anchor is not None and self.owner(ts) == self.anchor.purchase_at


def build_timeline(order_rows: list[dict[str, Any]], opened_at: datetime) -> Timeline:
    instances = sorted(
        (inst for row in order_rows if (inst := OrderInstance.from_row(row)) is not None),
        key=lambda inst: inst.purchase_at,
    )
    if not instances:
        return Timeline([], None, False)
    observable = [inst for inst in instances if inst.observable_at(opened_at)]
    purchased = [inst for inst in instances if inst.purchase_at <= opened_at]
    if observable:
        anchor = max(observable, key=lambda inst: inst.purchase_at)
    elif purchased:
        anchor = max(purchased, key=lambda inst: inst.purchase_at)
    else:  # every known instance postdates the complaint: no anchor, never guess one
        return Timeline(instances, None, False)
    tied = sum(1 for inst in instances if inst.purchase_at == anchor.purchase_at) > 1
    return Timeline(instances, anchor, tied)


# --------------------------------------------------------------------------- specialist findings


@dataclass
class EntityResolution:
    status: str  # resolved | ambiguous | not_found
    resolved_order_ids: list[str]
    rejected_candidates: list[str]
    customer_unique_id: str | None
    related_order_ids: list[str]
    confidence: float


@dataclass
class OrderFinding:
    order_id: str
    anchor_purchase_at: str | None
    anchor_status: str | None
    tied: bool
    instance_count: int
    item_ids: list[str]
    seller_ids: list[str]
    items_total: float | None
    freight_total: float | None
    shipping_limit_at: str | None
    order_row_is_anchor: bool | None
    order_row_status: str | None
    order_row_source: str = "get_order"


@dataclass
class PaymentFinding:
    has_payment_evidence: bool
    has_refund_evidence: bool
    captures: list[float]
    mismatch_amounts: list[float]
    refunds: list[tuple[str, float]]  # (status, amount) attributed to the anchor
    refunded_total: float

    @property
    def captured_total(self) -> float | None:
        if not self.has_payment_evidence:
            return None
        return round(sum(self.captures), 2)


@dataclass
class ShipmentFinding:
    has_shipment_evidence: bool
    verdict: str
    late: bool | None
    late_party: str | None  # seller | logistics_provider
    late_seller_ids: list[str]
    timeline_complete: bool
    event_actor: str | None
    event_conflict: bool


def resolve_entity(
    candidates: list[str],
    claimed_order_id: str | None,
    history: dict[str, Any] | None,
    verified_order_ids: set[str] | None = None,
) -> EntityResolution:
    """Rank candidates against authoritative customer history (and optional get_order hits)."""
    candidates = _unique([*candidates, *([claimed_order_id] if claimed_order_id else [])])
    history_orders = _rows(history.get("orders")) if isinstance(history, dict) else []
    history_ids = _unique(
        [str(row.get("order_id")) for row in history_orders if row.get("order_id")]
    )
    verified = verified_order_ids or set()

    in_history = [cid for cid in candidates if cid in history_ids]
    confirmed = in_history or [cid for cid in candidates if cid in verified]
    if len(confirmed) > 1 and claimed_order_id in confirmed:
        confirmed = [claimed_order_id]  # claim breaks ties between otherwise valid candidates
    status = "resolved" if len(confirmed) == 1 else ("ambiguous" if confirmed else "not_found")
    rejected = [cid for cid in candidates if cid not in confirmed]

    if status == "resolved":
        confidence = 0.96 if in_history else 0.72
    elif status == "ambiguous":
        confidence = 0.45
    else:
        confidence = 0.2
    customer = (
        history.get("customer_unique_id") if isinstance(history, dict) and in_history else None
    )
    return EntityResolution(
        status=status,
        resolved_order_ids=confirmed[:20],
        rejected_candidates=rejected[:20],
        customer_unique_id=str(customer)[:128] if customer else None,
        related_order_ids=history_ids[:20] if in_history else [],
        confidence=confidence,
    )


def analyze_order(
    order_id: str,
    timeline: Timeline,
    order_row: dict[str, Any] | None,
    items: list[dict[str, Any]] | None,
) -> OrderFinding:
    anchor = timeline.anchor
    anchor_items: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _rows(items):
        limit = parse_ts(row.get("shipping_limit_date"))
        if anchor is None or timeline.owned_by_anchor(limit):
            key = (
                row.get("order_item_id"),
                row.get("seller_id"),
                row.get("shipping_limit_date"),
                row.get("price"),
                row.get("freight_value"),
            )
            anchor_items.setdefault(key, row)  # identical rows from tied instances collapse
    unique_items = list(anchor_items.values())
    prices = [money(row.get("price")) for row in unique_items]
    freights = [money(row.get("freight_value")) for row in unique_items]
    items_total = (
        round(
            sum(p for p in prices if p is not None) + sum(f for f in freights if f is not None), 2
        )
        if unique_items and None not in prices
        else None
    )
    limits = [ts for row in unique_items if (ts := parse_ts(row.get("shipping_limit_date")))]

    row_instance = OrderInstance.from_row(order_row) if isinstance(order_row, dict) else None
    return OrderFinding(
        order_id=order_id,
        anchor_purchase_at=iso(anchor.purchase_at) if anchor else None,
        anchor_status=anchor.status if anchor else None,
        tied=timeline.tied,
        instance_count=len(timeline.instances),
        item_ids=_unique([str(row.get("order_item_id") or "") for row in unique_items])[:20],
        seller_ids=_unique([str(row.get("seller_id") or "") for row in unique_items])[:20],
        items_total=items_total,
        freight_total=round(sum(f for f in freights if f is not None), 2) if unique_items else None,
        shipping_limit_at=iso(min(limits)) if limits else None,
        order_row_is_anchor=(
            row_instance.purchase_at == anchor.purchase_at if row_instance and anchor else None
        ),
        order_row_status=row_instance.status if row_instance else None,
    )


def analyze_payment(
    timeline: Timeline,
    payment_timeline: dict[str, Any] | None,
    refund_timeline: dict[str, Any] | None,
) -> PaymentFinding:
    events = _rows(payment_timeline.get("events")) if isinstance(payment_timeline, dict) else []
    captures_by_owner: dict[datetime | None, list[float]] = {}
    anchor_captures: list[tuple[datetime | None, float]] = []
    mismatches: list[float] = []
    for event in events:
        at = parse_ts(event.get("event_at"))
        amount = money(event.get("amount_brl"))
        kind = event.get("event_type")
        status = str(event.get("status") or "").lower()
        if kind == "captured" and amount is not None and status not in VOID_CAPTURE_STATUSES:
            captures_by_owner.setdefault(timeline.owner(at), []).append(amount)
            if timeline.owned_by_anchor(at):
                anchor_captures.append((at, amount))
        elif kind == "reconciliation_mismatch" and timeline.owned_by_anchor(at):
            if status != "resolved" and amount is not None:
                mismatches.append(amount)

    if timeline.tied:  # identical duplicated rows from tied instances are one capture
        anchor_captures = list(dict.fromkeys(anchor_captures))
    captures = [amount for _, amount in anchor_captures]

    refunds: list[tuple[str, float]] = []
    anchor_key = timeline.anchor.purchase_at if timeline.anchor else None
    refund_events = (
        _rows(refund_timeline.get("events")) if isinstance(refund_timeline, dict) else []
    )
    for event in refund_events:
        amount = money(event.get("amount_brl"))
        if amount is None:
            continue
        # Refunds belong to the instance that captured the same amount; fall back to time.
        owners = [
            key
            for key, amounts in captures_by_owner.items()
            if any(same_money(amount, value) for value in amounts)
        ]
        owned = (
            anchor_key in owners
            if owners
            else timeline.owned_by_anchor(parse_ts(event.get("event_at")))
        )
        if owned:
            refunds.append((str(event.get("status") or "").lower(), amount))
    refunded_total = round(
        float(sum(amount for status, amount in refunds if status in COMPLETED_REFUND_STATUSES)), 2
    )
    return PaymentFinding(
        has_payment_evidence=payment_timeline is not None,
        has_refund_evidence=refund_timeline is not None,
        captures=captures,
        mismatch_amounts=mismatches,
        refunds=refunds,
        refunded_total=refunded_total,
    )


def analyze_shipment(
    timeline: Timeline,
    order: OrderFinding,
    shipment: dict[str, Any] | None,
    *,
    summary_required: bool = True,
) -> ShipmentFinding:
    """Delivery verdict for the anchor.

    The verdict comes from the anchor's history dates and item shipping limits; the shipment
    summary adds the delivered_late actor cross-check. When the coordinator deliberately skips
    the summary (claims that are not about delivery), the history-based verdict still holds.
    """
    anchor = timeline.anchor
    if shipment is None and not summary_required:
        shipment = {}
    if shipment is None or anchor is None:
        return ShipmentFinding(False, "insufficient_evidence", None, None, [], False, None, False)

    limits = [
        ts
        for row in _rows(shipment.get("shipping_limits"))
        if (ts := parse_ts(row.get("shipping_limit_at"))) and timeline.owned_by_anchor(ts)
    ]
    shipping_limit = min(limits) if limits else parse_ts(order.shipping_limit_at)

    event_actor = None
    for event in _rows(shipment.get("events")):
        at = parse_ts(event.get("event_at"))
        mine = (at == anchor.delivered_at) if anchor.delivered_at else timeline.owned_by_anchor(at)
        if mine and str(event.get("event_type", "")).startswith("delivered_late"):
            event_actor = event.get("actor")

    timeline_complete = all(
        ts is not None for ts in (anchor.carrier_at, anchor.delivered_at, anchor.estimated_at)
    )
    if anchor.status in UNSHIPPED_TERMINAL_STATUSES or anchor.delivered_at is None:
        return ShipmentFinding(
            True, "insufficient_evidence", None, None, [], False, event_actor, False
        )
    if anchor.estimated_at is None:
        return ShipmentFinding(
            True, "insufficient_evidence", None, None, [], False, event_actor, False
        )

    late = anchor.delivered_at > anchor.estimated_at
    if not late:
        conflict = event_actor is not None  # a late event on an on-time delivery
        verdict = "conflicting" if conflict else "on_time"
        return ShipmentFinding(
            True, verdict, False, None, [], timeline_complete, event_actor, conflict
        )

    if anchor.carrier_at is not None and shipping_limit is not None:
        date_party = "seller" if anchor.carrier_at > shipping_limit else "logistics_provider"
    else:
        date_party = event_actor if event_actor in {"seller", "logistics_provider"} else None
    conflict = event_actor is not None and date_party is not None and event_actor != date_party
    if date_party is None:
        verdict = "insufficient_evidence"
    elif conflict:
        verdict = "conflicting"
    else:
        verdict = "seller_delay" if date_party == "seller" else "logistics_delay"
    sellers = order.seller_ids or _unique(
        [  # items missing: fall back to shipping-limit rows
            str(row.get("seller_id") or "")
            for row in _rows(shipment.get("shipping_limits"))
            if timeline.owned_by_anchor(parse_ts(row.get("shipping_limit_at")))
        ]
    )
    late_sellers = list(sellers) if date_party == "seller" and not conflict else []
    return ShipmentFinding(
        True, verdict, True, date_party, late_sellers, timeline_complete, event_actor, conflict
    )


def shipment_row_vs_anchor(
    timeline: Timeline, shipment: dict[str, Any] | None
) -> tuple[bool | None, str | None]:
    """Does the shipment summary's order row (the get_order row) describe the anchor?"""
    anchor = timeline.anchor
    if anchor is None or not isinstance(shipment, dict) or not shipment.get("order_status"):
        return None, None
    same = (
        shipment.get("order_status") == anchor.status
        and parse_ts(shipment.get("delivered_customer_at")) == anchor.delivered_at
        and parse_ts(shipment.get("estimated_delivery_at")) == anchor.estimated_at
    )
    return same, str(shipment.get("order_status"))


def entities_from_shipping_limits(
    timeline: Timeline, shipment: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    """(item_ids, seller_ids) of the anchor from shipment-summary shipping limits."""
    rows = [
        row
        for row in _rows(shipment.get("shipping_limits") if isinstance(shipment, dict) else None)
        if timeline.owned_by_anchor(parse_ts(row.get("shipping_limit_at")))
    ]
    items = _unique([str(row.get("order_item_id") or "") for row in rows])[:20]
    sellers = _unique([str(row.get("seller_id") or "") for row in rows])[:20]
    return items, sellers


# --------------------------------------------------------------------------- policy


def split_subset(captures: list[float], order_total: float | None) -> list[float] | None:
    """Largest group of >=2 captures that reconciles exactly to the order total."""
    if order_total is None or len(captures) < 2:
        return None
    for size in range(len(captures), 1, -1):
        for combo in combinations(captures, size):
            if same_money(sum(combo), order_total):
                return list(combo)
    return None


def is_duplicate_capture(captures: list[float], order_total: float | None) -> bool:
    if len(captures) < 2 or split_subset(captures, order_total) is not None:
        return False
    return any(captures.count(amount) > 1 for amount in captures)


def evidenced_issues(
    order: OrderFinding, payment: PaymentFinding, shipment: ShipmentFinding
) -> list[str]:
    found: set[str] = set()
    paid = bool(payment.captures)
    if order.anchor_status == "canceled" and paid:
        found.add("canceled_order_paid")
    if order.anchor_status == "unavailable" and paid:
        found.add("unavailable_order_paid")
    statuses = {status for status, _ in payment.refunds}
    if "failed" in statuses:
        found.add("refund_failed")
    if "pending" in statuses:
        found.add("refund_pending")
    if payment.mismatch_amounts:
        found.add("payment_mismatch")
    # Split vs duplicate is only decidable against the order total (both look like 2 equal
    # captures). Without items the pattern is ambiguous: both stay candidates and the claim
    # breaks the tie (flagged as a tie-break, so confidence drops).
    if order.items_total is None:
        if len(payment.captures) >= 2 and any(
            payment.captures.count(amount) > 1 for amount in payment.captures
        ):
            found.update({"valid_split_payment", "duplicate_charge"})
    elif split_subset(payment.captures, order.items_total) is not None:
        found.add("valid_split_payment")
    elif is_duplicate_capture(payment.captures, order.items_total):
        found.add("duplicate_charge")
    if shipment.late and shipment.late_party == "seller":
        found.add("late_delivery_seller")
    elif shipment.late and shipment.late_party == "logistics_provider":
        found.add("late_delivery_logistics")
    return [issue for issue in ISSUE_PRIORITY if issue in found]


def choose_primary_issue(
    issues: list[str], claim_topic: str | None, core_evidence_complete: bool
) -> tuple[str, bool]:
    """Return (primary_issue, claim_tiebreak_used)."""
    faults = [issue for issue in issues if issue not in NON_FAULT_ISSUES]
    if not faults:
        if issues and claim_topic in PAYMENT_TOPICS:
            return "valid_split_payment", False
        return ("unsupported_claim" if core_evidence_complete else "insufficient_evidence"), False
    if claim_topic in faults:
        return claim_topic, len(faults) > 1
    if claim_topic == "valid_split_payment" and "valid_split_payment" in issues:
        return "valid_split_payment", True
    return faults[0], len(faults) > 1


PAYMENT_VERDICT_BY_ISSUE = {
    "refund_failed": "refund_failed",
    "refund_pending": "refund_pending",
    "payment_mismatch": "capture_mismatch",
    "duplicate_charge": "duplicate_capture",
}


def payment_verdict(primary_issue: str, payment: PaymentFinding) -> str:
    if primary_issue in REFUND_TOPICS and payment.has_refund_evidence:
        return PAYMENT_VERDICT_BY_ISSUE[primary_issue]  # the refund lifecycle is payment evidence
    if not payment.has_payment_evidence:
        return "insufficient_evidence"
    if primary_issue in PAYMENT_VERDICT_BY_ISSUE:
        return PAYMENT_VERDICT_BY_ISSUE[primary_issue]
    if payment.refunded_total > 0:
        return "refunded"
    if not payment.captures:
        return "insufficient_evidence"
    return "reconciled"


def reconciled_captured_total(
    primary_issue: str, payment: PaymentFinding, order: OrderFinding
) -> float | None:
    """Captured total for the anchor; a valid split excludes captures from a conflicting record."""
    if not payment.has_payment_evidence:
        return None
    if primary_issue == "valid_split_payment":
        subset = split_subset(payment.captures, order.items_total)
        if subset is not None:
            return round(sum(subset), 2)
    return payment.captured_total


@dataclass
class PolicyDecision:
    primary_issue: str
    case_status: str
    recommended_action: str
    refund_brl: float
    responsible_parties: list[dict[str, Any]]
    evidenced_issues: list[str]
    claim_tiebreak: bool
    refund_capped: bool
    captured_total: float | None
    refunded_total: float | None
    refundable_total: float | None
    payment_verdict: str
    notes: list[str] = field(default_factory=list)


def decide(
    claim_topic: str | None,
    order: OrderFinding,
    payment: PaymentFinding,
    shipment: ShipmentFinding,
    policy: dict[str, Any] | None,
) -> PolicyDecision:
    issues = evidenced_issues(order, payment, shipment)
    core_complete = (
        order.anchor_status is not None
        and payment.has_payment_evidence
        and shipment.has_shipment_evidence
        # a refund claim can only be ruled out with the refund lifecycle in hand
        and (claim_topic not in REFUND_TOPICS or payment.has_refund_evidence)
    )
    primary, tiebreak = choose_primary_issue(issues, claim_topic, core_complete)
    ambiguous_pattern = {"valid_split_payment", "duplicate_charge"} <= set(issues)
    if ambiguous_pattern:
        # Equal captures without the order total: the claim picks split vs duplicate. That is
        # an evidence gap, not a conflict between sources, unless instances are identical.
        tiebreak = order.tied

    rules = policy.get("rules", {}) if isinstance(policy, dict) else {}
    rule = rules.get(primary) if isinstance(rules, dict) else None
    notes: list[str] = ["payment_pattern_ambiguous"] if ambiguous_pattern else []
    if isinstance(rule, dict):
        case_status = str(rule.get("case_status") or "needs_investigation")
        action = str(rule.get("recommended_action") or "escalate_manual_review")
        policy_refund = money(rule.get("refund_brl")) or 0.0
        parties_template = _rows(rule.get("responsible_parties"))
    else:
        if primary != "insufficient_evidence":
            notes.append("policy_rule_missing")
        case_status, action, policy_refund = (
            "needs_investigation",
            "request_additional_evidence",
            0.0,
        )
        parties_template = [{"party_type": "unknown", "party_id": None}]

    captured = reconciled_captured_total(primary, payment, order)
    # Completed refunds are known from either the payment or the refund lifecycle.
    refunded = (
        payment.refunded_total
        if payment.has_payment_evidence or payment.has_refund_evidence
        else None
    )
    refundable = round(max(captured - (refunded or 0.0), 0.0), 2) if captured is not None else None
    refund = policy_refund
    capped = False
    if refundable is not None and refund > refundable + MONEY_TOLERANCE:
        refund, capped = refundable, True
        notes.append("refund_capped_to_refundable")

    parties: list[dict[str, Any]] = []
    for template in parties_template[:5]:
        party_type = str(template.get("party_type") or "unknown")
        if party_type == "seller":
            # Policy templates carry an example seller id; responsibility is case-specific.
            sellers = shipment.late_seller_ids or order.seller_ids
            for seller_id in sellers[:5] or [None]:
                parties.append({"party_type": "seller", "party_id": seller_id})
        else:
            parties.append({"party_type": party_type, "party_id": None})
    unique_parties = list({(p["party_type"], p["party_id"]): p for p in parties}.values())[:5]

    return PolicyDecision(
        primary_issue=primary,
        case_status=case_status,
        recommended_action=action,
        refund_brl=round(refund, 2),
        responsible_parties=unique_parties,
        evidenced_issues=issues,
        claim_tiebreak=tiebreak,
        refund_capped=capped,
        captured_total=captured,
        refunded_total=refunded,
        refundable_total=refundable,
        payment_verdict=payment_verdict(primary, payment),
        notes=notes,
    )


# --------------------------------------------------------------------------- claims & conflicts

FULL_REFUND_ACTIONS = frozenset({"issue_refund", "retry_refund"})
PARTIAL_REFUND_ACTIONS = frozenset(
    {"refund_freight", "refund_duplicate_charge", "reconcile_payment"}
)


def claim_verdict(topic: str, decision: PolicyDecision) -> str:
    if topic == "requested_full_refund":
        if decision.recommended_action in FULL_REFUND_ACTIONS and decision.refund_brl > 0:
            return "supported"
        if decision.recommended_action in PARTIAL_REFUND_ACTIONS and decision.refund_brl > 0:
            return "partially_supported"
        if decision.case_status == "needs_investigation":
            return "insufficient_evidence"
        return "unsupported"
    if decision.primary_issue == "insufficient_evidence":
        return "insufficient_evidence"
    if topic == "unsupported_claim":
        return (
            "unsupported"
            if decision.primary_issue == "unsupported_claim"
            else "partially_supported"
        )
    if topic == decision.primary_issue:
        return "supported"
    if topic in decision.evidenced_issues:
        return "partially_supported"
    return "unsupported"


def data_conflicts(
    order: OrderFinding,
    payment: PaymentFinding,
    shipment: ShipmentFinding,
    decision: PolicyDecision,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    if order.order_row_is_anchor is False:
        conflicts.append(
            {
                "field": "order_instance",
                "sources": [order.order_row_source, "get_customer_history"],
                "selected_source": "get_customer_history",
                "resolution_code": "anchor_instance_observable_at_case_open",
            }
        )
        if order.order_row_status and order.order_row_status != order.anchor_status:
            conflicts.append(
                {
                    "field": "order_status",
                    "sources": [order.order_row_source, "get_customer_history"],
                    "selected_source": "get_customer_history",
                    "resolution_code": "status_of_anchor_instance",
                }
            )
    if shipment.event_conflict:
        conflicts.append(
            {
                "field": "late_delivery_party",
                "sources": ["get_shipment_summary", "get_customer_history"],
                "selected_source": None,
                "resolution_code": "unresolved_event_actor_vs_handoff_dates",
            }
        )
    if (
        decision.captured_total is not None
        and payment.captured_total is not None
        and not same_money(decision.captured_total, payment.captured_total)
    ):
        conflicts.append(
            {
                "field": "captured_total_brl",
                "sources": ["get_payment_timeline", "get_order_items"],
                "selected_source": "get_order_items",
                "resolution_code": "split_captures_reconciled_to_order_total",
            }
        )
    if decision.claim_tiebreak:
        conflicts.append(
            {
                "field": "primary_issue",
                "sources": ["get_customer_history", "get_payment_timeline"],
                "selected_source": None,
                "resolution_code": "identical_instances_claim_consistent_selection",
            }
        )
    return conflicts[:5]
