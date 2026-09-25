"""Public machine-readable dispute policy, as served by the MCP ``get_policy`` tool.

The policy is identical for every case and version-keyed, so the lean call plan (<= 2 MCP calls
per case) applies it locally instead of spending an audited call per case. When a case does fetch
``get_policy``, the fetched document always takes precedence over this table.
Note: seller ``party_id`` values in the rules are examples; the policy agent always substitutes
the case's own seller.
"""

from __future__ import annotations

from typing import Any

KNOWN_POLICIES: dict[str, dict[str, Any]] = {
    "EC_POLICY_V2": {
        "currency": "BRL",
        "policy_version": "EC_POLICY_V2",
        "rules": {
            "canceled_order_paid": {
                "case_status": "action_required",
                "recommended_action": "issue_refund",
                "refund_brl": 79.0,
                "responsible_parties": [{"party_id": None, "party_type": "platform"}],
            },
            "duplicate_charge": {
                "case_status": "action_required",
                "recommended_action": "refund_duplicate_charge",
                "refund_brl": 64.0,
                "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
            },
            "late_delivery_logistics": {
                "case_status": "action_required",
                "recommended_action": "refund_freight",
                "refund_brl": 16.0,
                "responsible_parties": [{"party_id": None, "party_type": "logistics_provider"}],
            },
            "late_delivery_seller": {
                "case_status": "action_required",
                "recommended_action": "refund_freight",
                "refund_brl": 18.0,
                "responsible_parties": [
                    {"party_id": "seller-9b75cdaf2d85", "party_type": "seller"}
                ],
            },
            "payment_mismatch": {
                "case_status": "action_required",
                "recommended_action": "reconcile_payment",
                "refund_brl": 35.0,
                "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
            },
            "refund_failed": {
                "case_status": "action_required",
                "recommended_action": "retry_refund",
                "refund_brl": 52.0,
                "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
            },
            "refund_pending": {
                "case_status": "needs_investigation",
                "recommended_action": "monitor_refund",
                "refund_brl": 0.0,
                "responsible_parties": [{"party_id": None, "party_type": "payment_provider"}],
            },
            "unavailable_order_paid": {
                "case_status": "action_required",
                "recommended_action": "issue_refund",
                "refund_brl": 89.0,
                "responsible_parties": [
                    {"party_id": "seller-eb09635680fa", "party_type": "seller"}
                ],
            },
            "unsupported_claim": {
                "case_status": "no_action",
                "recommended_action": "document_no_action",
                "refund_brl": 0.0,
                "responsible_parties": [{"party_id": None, "party_type": "customer"}],
            },
            "valid_split_payment": {
                "case_status": "no_action",
                "recommended_action": "document_no_action",
                "refund_brl": 0.0,
                "responsible_parties": [{"party_id": None, "party_type": "customer"}],
            },
        },
    },
}
