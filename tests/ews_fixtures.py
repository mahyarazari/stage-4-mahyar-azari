"""A small synthetic Olist history for the early-warning tests: every time is known.

Order i is bought on day i (from 2017-01-02) at 10:00 and approved at 11:00; it is promised
for 15 days later, delivered 9 days later (every 4th order 20 days later: late) and its review
is answered 2 days after delivery. Two sellers; multi-item and late orders are reviewed badly
more often, so a model has something to learn. Customers repeat every 7 orders.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from tests.olist_fixtures import base_rows, hexid, write_files

START = datetime(2017, 1, 2, 10, 0, 0)
FMT = "%Y-%m-%d %H:%M:%S"
HOLIDAYS = {2017: [{"date": "2017-04-21", "name": "Tiradentes", "type": "national"},
                   {"date": "2017-09-07", "name": "Independencia", "type": "national"},
                   {"date": "2017-11-15", "name": "Republica", "type": "national"}],
            2018: [{"date": "2018-01-01", "name": "Ano novo", "type": "national"}]}


def order_times(i: int) -> dict:
    purchased = START + timedelta(days=i)
    late = i % 4 == 0
    delivered = purchased + timedelta(days=20 if late else 9)
    return {"purchased": purchased, "approved": purchased + timedelta(hours=1),
            "estimated": (purchased + timedelta(days=15)).replace(hour=0, minute=0, second=0),
            "delivered": delivered, "answered": delivered + timedelta(days=2), "late": late}


def ews_rows(n: int = 360) -> dict[str, list[dict]]:
    rows = base_rows()
    rows["sellers"].append({"seller_id": hexid(12), "seller_zip_code_prefix": "01037", "seller_city": "sao paulo",
                            "seller_state": "RJ"})
    for table in ("customers", "orders", "order_items", "order_payments", "order_reviews"):
        rows[table] = []
    for i in range(1, n + 1):
        t = order_times(i)
        items = 1 + (i % 3 == 0)
        negative = t["late"] or (items > 1 and i % 2 == 0)
        rows["customers"].append({"customer_id": hexid(1000 + i), "customer_unique_id": hexid(5000 + i % 7),
                                  "customer_zip_code_prefix": "01037", "customer_city": "sao paulo",
                                  "customer_state": "SP" if i % 5 else "BA"})
        rows["orders"].append({
            "order_id": hexid(20000 + i), "customer_id": hexid(1000 + i), "order_status": "delivered",
            "order_purchase_timestamp": t["purchased"].strftime(FMT), "order_approved_at": t["approved"].strftime(FMT),
            "order_delivered_carrier_date": (t["purchased"] + timedelta(days=2)).strftime(FMT),
            "order_delivered_customer_date": t["delivered"].strftime(FMT),
            "order_estimated_delivery_date": t["estimated"].strftime(FMT)})
        for item in range(1, items + 1):
            rows["order_items"].append({
                "order_id": hexid(20000 + i), "order_item_id": str(item), "product_id": hexid(21),
                "seller_id": hexid(11 + (i % 2)), "shipping_limit_date": (t["approved"] + timedelta(days=4)).strftime(FMT),
                "price": "50.00", "freight_value": "12.50"})
        rows["order_payments"].append({"order_id": hexid(20000 + i), "payment_sequential": "1",
                                       "payment_type": "credit_card", "payment_installments": str(1 + i % 4),
                                       "payment_value": f"{items * 62.5:.2f}"})
        rows["order_reviews"].append({
            "review_id": hexid(40000 + i), "order_id": hexid(20000 + i), "review_score": "1" if negative else "5",
            "review_comment_title": "", "review_comment_message": "",
            "review_creation_date": t["delivered"].strftime("%Y-%m-%d 00:00:00"),
            "review_answer_timestamp": t["answered"].strftime(FMT)})
    return rows


def write_ews_files(raw_dir: Path, holidays_dir: Path, rows=None) -> None:
    write_files(raw_dir, rows or ews_rows())
    holidays_dir.mkdir(parents=True, exist_ok=True)
    for year, items in HOLIDAYS.items():            # cached responses: the fetch stage needs no network
        (holidays_dir / f"{year}.json").write_text(json.dumps(items), encoding="utf-8")
