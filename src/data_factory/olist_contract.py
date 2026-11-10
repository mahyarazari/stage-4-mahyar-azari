"""Data contract for the Olist files (stage 2).

One TableSpec per source file: which columns it must have, what each value must look
like, what makes a row unique, and which other table each foreign key points to. The
specs are data, read by the generic engine in `table_validation.py`; nothing here knows
about files, HTTP or databases.

Column names are renamed to the database names here (for example the source's misspelt
`product_name_lenght` becomes `name_length`): the database is designed, not copied.

Foreign-key policy (stage-2 decisions):
  QUARANTINE  a row whose parent does not exist cannot be trusted (an item of an order
              that does not exist): it is quarantined with ORPHAN_FOREIGN_KEY. Because
              parents are validated first, a quarantined order also quarantines its items.
  COMPLETE    the parent table is a lookup that is known to be incomplete (category
              translations, zip-code locations). The child row is kept; the missing parent
              is added in the database marked is_inferred (sql/olist/20_complete_parents.sql)
              and counted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    ID = "id"                # 32 lower-case hex characters (an md5-like key)
    TEXT = "text"
    COUNT = "count"          # whole number >= 0
    MONEY = "money"          # >= 0 with at most two decimals
    DECIMAL = "decimal"      # >= 0, any precision, scientific notation allowed (7.96E-05)
    TIMESTAMP = "timestamp"  # YYYY-MM-DD HH:MM:SS
    DATE = "date"            # a timestamp at 00:00:00, stored as its date
    DAY = "day"              # a plain date YYYY-MM-DD (holiday calendar)
    ZIP = "zip"              # 5-digit zip-code prefix; leading zeros matter, so text
    STATE = "state"          # one of the 27 Brazilian federative units
    COORDINATE = "coordinate"
    CHOICE = "choice"        # one of Column.values


class FkPolicy(str, Enum):
    QUARANTINE = "quarantine"
    COMPLETE = "complete"


class Reason(str, Enum):
    """Why an Olist row was quarantined; written in this order when a row has several."""

    MISSING_REQUIRED = "MISSING_REQUIRED"
    INVALID_VALUE = "INVALID_VALUE"
    IMPOSSIBLE_DATES = "IMPOSSIBLE_DATES"
    DUPLICATE_KEY = "DUPLICATE_KEY"
    ORPHAN_FOREIGN_KEY = "ORPHAN_FOREIGN_KEY"


STATES = ("AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
          "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO")
ORDER_STATUSES = ("approved", "canceled", "created", "delivered", "invoiced", "processing",
                  "shipped", "unavailable")
PAYMENT_TYPES = ("boleto", "credit_card", "debit_card", "not_defined", "voucher")
REVIEW_SCORES = ("1", "2", "3", "4", "5")


@dataclass(frozen=True)
class Column:
    source: str
    target: str
    kind: Kind
    required: bool = False
    values: tuple[str, ...] = ()     # allowed values of a CHOICE column


@dataclass(frozen=True)
class ForeignKey:
    column: str                      # target column name in this table
    parent: str                      # parent table
    parent_column: str
    policy: FkPolicy


@dataclass(frozen=True)
class NotBefore:
    """`later` must not be earlier than `earlier` when both are present (else IMPOSSIBLE_DATES).

    Only physically impossible orders are rejected (delivered before bought); merely
    suspicious ones are flagged in the database instead (as in stage 1, D13).
    """

    earlier: str
    later: str


@dataclass(frozen=True)
class TableSpec:
    table: str                       # database table
    file: str                        # source file name
    columns: tuple[Column, ...]
    key: tuple[str, ...] = ()        # target columns that make a row unique (none: no key)
    foreign_keys: tuple[ForeignKey, ...] = ()
    rules: tuple[NotBefore, ...] = ()
    missing_tokens: tuple[str, ...] = ()   # source texts that mean "no value" (for example NA)

    @property
    def targets(self) -> list[str]:
        return [c.target for c in self.columns]


def _fk(column, parent, parent_column=None, policy=FkPolicy.QUARANTINE):
    return ForeignKey(column, parent, parent_column or column, policy)


# Parents come before children: a child is checked against the parents' accepted keys.
TABLES: tuple[TableSpec, ...] = (
    TableSpec("product_categories", "product_category_name_translation.csv", (
        Column("product_category_name", "category_name", Kind.TEXT, True),
        Column("product_category_name_english", "category_name_english", Kind.TEXT),
    ), key=("category_name",)),
    TableSpec("geolocation_staging", "olist_geolocation_dataset.csv", (
        Column("geolocation_zip_code_prefix", "zip_code_prefix", Kind.ZIP, True),
        Column("geolocation_lat", "latitude", Kind.COORDINATE),
        Column("geolocation_lng", "longitude", Kind.COORDINATE),
        Column("geolocation_city", "city", Kind.TEXT),
        Column("geolocation_state", "state", Kind.STATE),
    )),
    TableSpec("customers", "olist_customers_dataset.csv", (
        Column("customer_id", "customer_id", Kind.ID, True),
        Column("customer_unique_id", "customer_unique_id", Kind.ID, True),
        Column("customer_zip_code_prefix", "zip_code_prefix", Kind.ZIP, True),
        Column("customer_city", "city", Kind.TEXT, True),
        Column("customer_state", "state", Kind.STATE, True),
    ), key=("customer_id",),
        foreign_keys=(_fk("zip_code_prefix", "geolocation_staging", policy=FkPolicy.COMPLETE),)),
    TableSpec("sellers", "olist_sellers_dataset.csv", (
        Column("seller_id", "seller_id", Kind.ID, True),
        Column("seller_zip_code_prefix", "zip_code_prefix", Kind.ZIP, True),
        Column("seller_city", "city", Kind.TEXT, True),
        Column("seller_state", "state", Kind.STATE, True),
    ), key=("seller_id",),
        foreign_keys=(_fk("zip_code_prefix", "geolocation_staging", policy=FkPolicy.COMPLETE),)),
    TableSpec("products", "olist_products_dataset.csv", (
        Column("product_id", "product_id", Kind.ID, True),
        Column("product_category_name", "category_name", Kind.TEXT),
        Column("product_name_lenght", "name_length", Kind.COUNT),
        Column("product_description_lenght", "description_length", Kind.COUNT),
        Column("product_photos_qty", "photos_qty", Kind.COUNT),
        Column("product_weight_g", "weight_g", Kind.COUNT),
        Column("product_length_cm", "length_cm", Kind.COUNT),
        Column("product_height_cm", "height_cm", Kind.COUNT),
        Column("product_width_cm", "width_cm", Kind.COUNT),
    ), key=("product_id",),
        foreign_keys=(_fk("category_name", "product_categories", policy=FkPolicy.COMPLETE),)),
    TableSpec("orders", "olist_orders_dataset.csv", (
        Column("order_id", "order_id", Kind.ID, True),
        Column("customer_id", "customer_id", Kind.ID, True),
        Column("order_status", "status", Kind.CHOICE, True, ORDER_STATUSES),
        Column("order_purchase_timestamp", "purchased_at", Kind.TIMESTAMP, True),
        Column("order_approved_at", "approved_at", Kind.TIMESTAMP),
        Column("order_delivered_carrier_date", "delivered_carrier_at", Kind.TIMESTAMP),
        Column("order_delivered_customer_date", "delivered_customer_at", Kind.TIMESTAMP),
        Column("order_estimated_delivery_date", "estimated_delivery_date", Kind.DATE, True),
    ), key=("order_id",),
        foreign_keys=(_fk("customer_id", "customers"),),
        rules=(NotBefore("purchased_at", "delivered_customer_at"),)),
    TableSpec("order_items", "olist_order_items_dataset.csv", (
        Column("order_id", "order_id", Kind.ID, True),
        Column("order_item_id", "order_item_id", Kind.COUNT, True),
        Column("product_id", "product_id", Kind.ID, True),
        Column("seller_id", "seller_id", Kind.ID, True),
        Column("shipping_limit_date", "shipping_limit_at", Kind.TIMESTAMP, True),
        Column("price", "price", Kind.MONEY, True),
        Column("freight_value", "freight_value", Kind.MONEY, True),
    ), key=("order_id", "order_item_id"),
        foreign_keys=(_fk("order_id", "orders"), _fk("product_id", "products"), _fk("seller_id", "sellers"))),
    TableSpec("order_payments", "olist_order_payments_dataset.csv", (
        Column("order_id", "order_id", Kind.ID, True),
        Column("payment_sequential", "payment_sequential", Kind.COUNT, True),
        Column("payment_type", "payment_type", Kind.CHOICE, True, PAYMENT_TYPES),
        Column("payment_installments", "installments", Kind.COUNT, True),
        Column("payment_value", "payment_value", Kind.MONEY, True),
    ), key=("order_id", "payment_sequential"),
        foreign_keys=(_fk("order_id", "orders"),)),
    TableSpec("order_reviews", "olist_order_reviews_dataset.csv", (
        Column("review_id", "review_id", Kind.ID, True),
        Column("order_id", "order_id", Kind.ID, True),
        Column("review_score", "score", Kind.CHOICE, True, REVIEW_SCORES),
        Column("review_comment_title", "comment_title", Kind.TEXT),
        Column("review_comment_message", "comment_message", Kind.TEXT),
        Column("review_creation_date", "created_at", Kind.TIMESTAMP, True),
        Column("review_answer_timestamp", "answered_at", Kind.TIMESTAMP, True),
    ), key=("review_id", "order_id"),   # one review can cover several orders of a customer
        foreign_keys=(_fk("order_id", "orders"),),
        rules=(NotBefore("created_at", "answered_at"),)),
)
