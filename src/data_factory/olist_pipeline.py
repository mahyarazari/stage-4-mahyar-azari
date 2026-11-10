"""Stages of the Olist dataset (stage 2): fetch -> validate -> load.

Same building blocks as the 311 factory: the HTTP client with retries, atomic files,
quarantine with reason codes and a failure threshold, the sink layer for PostgreSQL and
the run manifest with its determinism check. What is new is only what Olist needs: a file
source pinned by checksums and a table-driven contract.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import text

from data_factory.config import Config
from data_factory.db import database
from data_factory.file_source import fetch_files, sha256_bytes, sha256_file
from data_factory.http_client import HttpClient
from data_factory.loading import TableLoad, load_tables
from data_factory.dataset_runs import manifest_header, run_stages
from data_factory.manifest import build_code_inputs, describe_outputs, save_manifest
from data_factory.olist_contract import TABLES
from data_factory.sinks import PostgresSink
from data_factory.table_validation import QUARANTINE_FILE, STATS_FILE, validate_tables

log = logging.getLogger(__name__)

DATASET = "olist"
QUARANTINE_TABLE = "quarantine"
INFERRED_SQL = text(
    "SELECT (SELECT count(*) FROM product_categories WHERE is_inferred) AS categories, "
    "(SELECT count(*) FROM zip_codes WHERE is_inferred) AS zip_codes")


def raw_dir(config: Config) -> Path:
    return config.paths.raw / DATASET


def work_dir(config: Config) -> Path:
    return config.paths.work / DATASET


def quarantine_dir(config: Config) -> Path:
    return config.paths.quarantine / DATASET


def snapshot_id(config: Config) -> str:
    """One name for the exact set of input files: a hash of the pinned checksums."""
    joined = "\n".join(f"{name}={digest}" for name, digest in sorted(config.olist.files.items()))
    return "olist-" + sha256_bytes(joined.encode())[:12]


def fetch(context) -> None:
    config = context.config
    http = HttpClient(timeout_seconds=config.fetch.timeout_seconds, max_retries=config.fetch.max_retries,
                      backoff_seconds=config.fetch.backoff_seconds)
    try:
        result = fetch_files(http, config.olist.base_url, config.olist.files, raw_dir(config))
    finally:
        context.record.http.update(asdict(http.stats))
    log.info("fetch olist: %d downloaded, %d already present", len(result.downloaded), len(result.reused))


def validate(context) -> None:
    config = context.config
    bounds = {"latitude": config.olist.latitude_range, "longitude": config.olist.longitude_range}
    validate_tables(TABLES, raw_dir(config), work_dir(config), quarantine_dir(config),
                    config.clean.chunk_rows, config.max_quarantine_ratio, bounds)


def load(context) -> None:
    config = context.config
    items = [TableLoad(spec.table, work_dir(config) / f"{spec.table}.csv", lambda frame: frame)
             for spec in TABLES]
    items.append(TableLoad(QUARANTINE_TABLE, quarantine_dir(config) / QUARANTINE_FILE, _reasons_to_array))
    inputs = build_code_inputs(config.path)
    run = {"load_id": context.record.run_id, "snapshot": snapshot_id(config),
           "config_sha256": inputs["config_sha256"], "git_commit": inputs["git_commit"]}
    schema = config.olist.load.schema
    with database(config.database) as engine:
        sink = PostgresSink(engine, schema, config.olist.load.before_load, config.olist.load.after_load)
        stats = load_tables(items, sink, config.clean.chunk_rows, run)
        with engine.connect() as connection:
            connection.execute(text("SELECT set_config('search_path', :path, false)"), {"path": schema})
            inferred = dict(connection.execute(INFERRED_SQL).mappings().one())
    context.record.load = {"schema": schema, "load_id": stats.load_id, "rows": stats.rows,
                           "inferred_parents": inferred}
    log.info("load olist: parents added as inferred %s", inferred)


def _reasons_to_array(frame):
    out = frame.copy()
    out["reason_codes"] = "{" + out["reason_codes"].str.replace("|", ",", regex=False) + "}"
    return out


STAGE_NAMES = ("fetch", "validate", "load")
STAGES = {"fetch": fetch, "validate": validate, "load": load}


def output_files(config: Config) -> dict[str, Path]:
    files = {f"work/{DATASET}/{spec.table}.csv": work_dir(config) / f"{spec.table}.csv" for spec in TABLES}
    files[f"quarantine/{DATASET}/{QUARANTINE_FILE}"] = quarantine_dir(config) / QUARANTINE_FILE
    return files


def record_run(context, status: str, error: str | None = None) -> None:
    config, record = context.config, context.record
    stats_path = work_dir(config) / STATS_FILE
    present = {name: sha256_file(raw_dir(config) / name) for name in sorted(config.olist.files)
               if (raw_dir(config) / name).exists()}
    manifest = {
        **manifest_header(context, DATASET, status, error, snapshot_id(config)),
        "source": {"base_url": config.olist.base_url, "files_sha256": present,
                   "all_checksums_match": present == dict(sorted(config.olist.files.items()))},
        "requests": {"this_run": record.http, "full_download": len(config.olist.files)},
        "counts": json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else None,
        "outputs": describe_outputs(output_files(config)),
        "load": record.load or None,
    }
    save_manifest(manifest, config.paths.curated / DATASET, work_dir(config) / "manifests")


def run(context, stage_names: list[str]) -> None:
    run_stages(context, DATASET, stage_names, STAGES, record_run)
