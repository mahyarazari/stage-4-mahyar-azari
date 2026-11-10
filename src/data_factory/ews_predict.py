"""Rank new orders with the saved early-warning model (D84).

    data-factory ews-predict --input <folder> --output ranked.csv

The input is the orders file itself or the folder that holds it, in the Olist file format
(same names, same columns):
  required  olist_orders_dataset.csv
  optional  olist_order_items_dataset.csv, olist_order_payments_dataset.csv,
            olist_customers_dataset.csv, olist_products_dataset.csv, olist_sellers_dataset.csv,
            product_category_name_translation.csv
A part that is not given is taken from the database when it knows the order (a held-back time
slice of Olist); otherwise those features are missing and the model fills them as in training.
Every order of the input file gets a row in the output.
Review files are ignored: a review is the future of an order.

The new rows take the road of the training rows, so no rule is written twice:
  1. the same contract (olist_contract): a missing column stops with SourceSchemaChangedError
     (exit 32), bad rows go to a quarantine file;
  2. inside one transaction they are added to the Olist tables and their features are read from
     the same view (order_features_live), history strictly before each order's decision moment;
  3. the transaction is rolled back: the database is left exactly as it was;
  4. the saved model ranks them and the daily capacity quota marks the orders to follow up.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

from data_factory import ews_model as em
from data_factory.config import Config
from data_factory.db import database
from data_factory.ews_pipeline import SQL_DIR, rank_orders, search_path, use_schemas
from data_factory.exceptions import MissingInputError
from data_factory.olist_contract import TABLES
from data_factory.table_validation import QUARANTINE_FILE, validate_tables

REQUIRED = ("orders",)
OPTIONAL = ("product_categories", "sellers", "products", "customers", "order_items", "order_payments")
SCORING_SQL = SQL_DIR / "30_scoring.sql"
LIVE_QUERY = "SELECT * FROM order_features_live WHERE order_id::text = ANY(:ids) ORDER BY order_id"


def scoring_specs(folder: Path):
    """Contract specs of the files present, parents first; foreign keys to tables that are not
    in the folder are left to the database (lookups it already has)."""
    by_table = {spec.table: spec for spec in TABLES}
    missing = [by_table[t].file for t in REQUIRED if not (folder / by_table[t].file).exists()]
    if missing:
        raise MissingInputError(f"{folder}: required file(s) missing: {missing}")
    present = [t for t in (*OPTIONAL, *REQUIRED) if (folder / by_table[t].file).exists()]
    order = [spec.table for spec in TABLES]
    specs = []
    for table in sorted(present, key=order.index):
        spec = by_table[table]
        specs.append(dataclasses.replace(spec, foreign_keys=tuple(fk for fk in spec.foreign_keys if fk.parent in present)))
    return tuple(specs)


def predict_folder(config: Config, folder: Path, output: Path) -> pd.DataFrame:
    folder = Path(folder).resolve()
    if folder.is_file():
        folder = folder.parent
    model_file = config.ews.output_dir / "model.joblib"
    if not model_file.exists():
        raise MissingInputError(f"{model_file} not found: run `data-factory early-warning` first")
    specs = scoring_specs(folder)
    work = config.paths.work / "ews_scoring"
    bounds = {"latitude": config.olist.latitude_range, "longitude": config.olist.longitude_range}
    stats = validate_tables(specs, folder, work, work, config.clean.chunk_rows, config.max_quarantine_ratio,
                            bounds, "ews scoring")
    frames = {spec.table: pd.read_csv(work / f"{spec.table}.csv", dtype=str) for spec in specs}
    for table in OPTIONAL:                                   # empty stand-ins keep the SQL fixed
        if table not in frames:
            spec = next(s for s in TABLES if s.table == table)
            frames[table] = pd.DataFrame(columns=spec.targets, dtype=str)
    ids = frames["orders"]["order_id"].str.lower().tolist()

    with database(config.database) as engine, engine.connect() as connection:
        transaction = connection.begin()
        try:
            use_schemas(connection, search_path(config))
            for table, frame in frames.items():
                frame.to_sql(f"new_{table}", connection, schema="pg_temp", if_exists="replace", index=False)
            connection.exec_driver_sql(SCORING_SQL.read_text(encoding="utf-8"))
            live = pd.read_sql_query(text(LIVE_QUERY), connection, params={"ids": [_uuid(i) for i in ids]})
        finally:
            transaction.rollback()                        # nothing stays in the database

    missing = len(ids) - len(live)
    if missing:
        raise MissingInputError(f"{missing} order(s) of the input could not be scored")
    features = em.prepare_types(live)
    ranked = rank_orders(joblib.load(model_file), features, config.ews.capacity)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    ranked.attrs["quarantined"] = stats["quarantined_records"]
    ranked.attrs["quarantine_file"] = str(work / QUARANTINE_FILE)
    return ranked


def _uuid(hex32: str) -> str:
    """Olist ids are 32 hex characters; the database shows uuids with dashes."""
    h = hex32.replace("-", "")
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
