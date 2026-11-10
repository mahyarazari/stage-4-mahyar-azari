"""Temporal audit of every column the early-warning model may see (D75, D82).

For each column of ews.order_features: when in the life of an order it comes into existence,
whether it exists at the decision moment (approval), and what makes us sure. The model layer
takes its feature list from this table only (`model_features`), and a test compares the table
with the columns of the database view: a new column added to the SQL without an audit row, or
a column marked not available, can never reach the model.

Moments, in the order of an order's life:
  purchase   the customer places the order (items, prices, freight, payment, promised date)
  approval   the payment is approved: THE DECISION MOMENT
  lookup     a reference fact that does not depend on the order (product, seller, zip, calendar)
  history    computed from OTHER orders' events strictly before the decision moment
  carrier / delivery / review / final   after the decision: never allowed
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KEY = "order_id"
TIME = "decision_at"
LABEL = "label_negative"
LABEL_TIME = "label_known_at"


@dataclass(frozen=True)
class Audit:
    moment: str          # when the value comes into existence
    allowed: bool        # exists at the decision moment
    certainty: str       # what makes us sure
    kind: str = "num"    # num, cat or flag (for the model) / meta (not a feature)


_SQL = "sql/ews/20_features.sql"
_HIST = ("history", True, "window ends 1 microsecond before decision_at; reviews counted at answered_at, "
                          "deliveries at delivered_customer_at (test_ews: future events change nothing)")

AUDIT: dict[str, Audit] = {
    KEY: Audit("purchase", True, "primary key", "meta"),
    TIME: Audit("approval", True, "the decision moment itself (purchase time when approval is missing)", "meta"),
    LABEL: Audit("review", False, "the target: score of the review answered after delivery", "meta"),
    LABEL_TIME: Audit("review", False, "when the label became known; used only for label maturity", "meta"),
    "approval_missing": Audit("approval", True, "approved_at is null at the decision moment", "flag"),
    "approval_hours": Audit("approval", True, "purchase -> approval, both before or at the decision"),
    "purchase_hour": Audit("purchase", True, "order_purchase_timestamp"),
    "purchase_weekday": Audit("purchase", True, "order_purchase_timestamp"),
    "promised_days": Audit("purchase", True, "estimated delivery date is shown to the customer at checkout "
                                             "(Olist data description); exploration checks it never changes after"),
    "days_to_ship_limit": Audit("approval", True, "seller shipping limit is set from the approval "
                                                  "(exploration: limit - approval is a fixed handling time)"),
    "n_items": Audit("purchase", True, "order_items rows are the basket"),
    "n_products": Audit("purchase", True, "order_items"),
    "n_sellers": Audit("purchase", True, "order_items"),
    "total_price": Audit("purchase", True, "order_items.price"),
    "total_freight": Audit("purchase", True, "order_items.freight_value, quoted at checkout"),
    "freight_share": Audit("purchase", True, "from price and freight"),
    "n_payments": Audit("purchase", True, "order_payments: chosen at checkout"),
    "max_installments": Audit("purchase", True, "order_payments"),
    "uses_voucher": Audit("purchase", True, "order_payments", "flag"),
    "uses_boleto": Audit("purchase", True, "order_payments; a boleto is approved when it is paid", "flag"),
    "paid_minus_order": Audit("purchase", True, "payment value against basket value"),
    "main_category": Audit("lookup", True, "product category does not depend on the order", "cat"),
    "total_weight_g": Audit("lookup", True, "product catalogue"),
    "max_volume_cm3": Audit("lookup", True, "product catalogue"),
    "min_photos": Audit("lookup", True, "product catalogue"),
    "min_description_length": Audit("lookup", True, "product catalogue"),
    "customer_state": Audit("purchase", True, "delivery address of the order", "cat"),
    "seller_state": Audit("lookup", True, "seller registry", "cat"),
    "same_state": Audit("lookup", True, "from the two states", "flag"),
    "distance_km": Audit("lookup", True, "zip-code centroids (stage 2), no order facts"),
    "seller_prior_orders": Audit(*_HIST),
    "seller_prior_reviews": Audit(*_HIST),
    "seller_prior_negative_rate": Audit(*_HIST),
    "seller_prior_late_rate": Audit(*_HIST),
    "seller_age_days": Audit(*_HIST),
    "customer_prior_orders": Audit(*_HIST),
    "customer_prior_negatives": Audit(*_HIST),
    "category_prior_negative_rate": Audit(*_HIST),
    "promised_working_days": Audit("lookup", True, "holiday calendar (BrasilAPI) is published in advance; "
                                                   "window = decision date -> promised date"),
    "holidays_in_window": Audit("lookup", True, "holiday calendar, same window"),
}

# Tempting columns found in the data and left out on purpose (D81); the cheat model uses them.
TEMPTING: dict[str, Audit] = {
    "final_status": Audit("final", False, "orders.status is the LAST status (delivered, canceled ...), "
                                          "overwritten during the order's life", "cat"),
    "days_late": Audit("delivery", False, "needs delivered_customer_at"),
    "delivery_days": Audit("delivery", False, "needs delivered_customer_at"),
    "flag_status_delivery_mismatch": Audit("final", False, "generated column of orders built from the final "
                                                           "status and the delivery date", "flag"),
    "seller_negative_rate_all_time": Audit("review", False, "seller's negative rate over ALL reviews, "
                                                            "including later ones and this order's own"),
}


def model_features(kind: str | None = None) -> list[str]:
    """Columns the model may use: allowed and not meta, in audit order."""
    return [c for c, a in AUDIT.items() if a.allowed and a.kind != "meta" and (kind is None or a.kind == kind)]


def check_columns(columns) -> list[str]:
    """Problems with a set of model input columns; empty when every column is audited and allowed."""
    problems = []
    for column in columns:
        audit = AUDIT.get(column)
        if audit is None:
            problems.append(f"{column}: not in the temporal audit")
        elif not audit.allowed:
            problems.append(f"{column}: exists only after the decision ({audit.moment})")
    return problems


def audit_table() -> pd.DataFrame:
    rows = [{"column": c, "moment": a.moment, "available_at_approval": a.allowed, "certainty": a.certainty,
             "used_by_model": a.allowed and a.kind != "meta"} for c, a in AUDIT.items()]
    rows += [{"column": c, "moment": a.moment, "available_at_approval": a.allowed, "certainty": a.certainty,
              "used_by_model": False} for c, a in TEMPTING.items()]
    return pd.DataFrame(rows)
