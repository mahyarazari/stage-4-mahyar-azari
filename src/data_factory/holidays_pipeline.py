"""Stage 4, external source: Brazilian national holidays from BrasilAPI (fetch -> validate -> load).

Hypothesis (D74): an order whose delivery window contains holidays has fewer working days to
reach the customer, so it is more often late and more often reviewed badly. The holiday
calendar of the coming weeks is published in advance, so it is known when an order is
approved: a feature built from it does not look into the future.

Same factory as Olist and credit: the HTTP client with retries, raw responses kept as they
arrived, the table contract with quarantine and its limit, the PostgreSQL sink in one
transaction and the run manifest. A past year's calendar never changes, so a year already
on disk is not requested again (offline repeat runs).
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from data_factory.config import Config
from data_factory.db import database
from data_factory.exceptions import InvalidResponseError
from data_factory.file_source import sha256_file
from data_factory.http_client import HttpClient
from data_factory.loading import TableLoad, load_tables
from data_factory.dataset_runs import manifest_header, run_stages
from data_factory.manifest import build_code_inputs, describe_outputs, save_manifest
from data_factory.olist_contract import Column, Kind, TableSpec
from data_factory.olist_pipeline import _reasons_to_array
from data_factory.sinks import PostgresSink
from data_factory.table_validation import QUARANTINE_FILE, STATS_FILE, validate_tables
from data_factory.tables import write_csv

DATASET = "holidays"
CSV_FILE = "holidays.csv"
STAGE_NAMES = ("fetch", "validate", "load")
SOURCE_FIELDS = ["date", "name", "type"]

TABLES = (TableSpec("holidays", CSV_FILE, (
    Column("date", "holiday_date", Kind.DAY, True),
    Column("name", "name", Kind.TEXT, True),
    Column("type", "holiday_type", Kind.TEXT, True),
), key=("holiday_date", "name")),)


def raw_dir(config: Config):
    return config.paths.raw / DATASET


def work_dir(config: Config):
    return config.paths.work / DATASET


def quarantine_dir(config: Config):
    return config.paths.quarantine / DATASET


def year_file(config: Config, year: int):
    return raw_dir(config) / f"{year}.json"


def responses_to_rows(responses: dict[int, object]) -> pd.DataFrame:
    """One row per holiday, the source's own field names, years in order. Anything that is
    not a list of objects is a broken response, not data: the run stops."""
    rows = []
    for year in sorted(responses):
        body = responses[year]
        if not isinstance(body, list) or not all(isinstance(item, dict) for item in body):
            raise InvalidResponseError(f"holidays {year}: expected a JSON list of objects")
        rows += [{field: item.get(field) for field in SOURCE_FIELDS} for item in body]
    return pd.DataFrame(rows, columns=SOURCE_FIELDS)


def fetch(context) -> None:
    config = context.config
    raw_dir(config).mkdir(parents=True, exist_ok=True)
    http = HttpClient(timeout_seconds=config.fetch.timeout_seconds, max_retries=config.fetch.max_retries,
                      backoff_seconds=config.fetch.backoff_seconds)
    responses = {}
    try:
        for year in config.ews.holidays_years:
            path = year_file(config, year)
            if not path.exists():
                body = http.get_json(f"{config.ews.holidays_url}/{year}")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                temporary.replace(path)
            responses[year] = json.loads(path.read_text(encoding="utf-8"))
    finally:
        context.record.http.update(asdict(http.stats))
    write_csv(raw_dir(config) / CSV_FILE, responses_to_rows(responses))


def validate(context) -> None:
    config = context.config
    validate_tables(TABLES, raw_dir(config), work_dir(config), quarantine_dir(config), config.clean.chunk_rows,
                    config.max_quarantine_ratio, {}, DATASET)


def load(context) -> None:
    config = context.config
    items = [TableLoad("holidays", work_dir(config) / "holidays.csv", lambda frame: frame),
             TableLoad("quarantine", quarantine_dir(config) / QUARANTINE_FILE, _reasons_to_array)]
    inputs = build_code_inputs(config.path)
    run = {"load_id": context.record.run_id, "snapshot": snapshot_id(config),
           "config_sha256": inputs["config_sha256"], "git_commit": inputs["git_commit"]}
    with database(config.database) as engine:
        sink = PostgresSink(engine, config.ews.load.schema, config.ews.load.before_load, config.ews.load.after_load)
        stats = load_tables(items, sink, config.clean.chunk_rows, run)
    context.record.load = {"schema": config.ews.load.schema, "load_id": stats.load_id, "rows": stats.rows}


def snapshot_id(config: Config) -> str:
    years = "-".join(str(y) for y in config.ews.holidays_years)
    return f"holidays-{years}"


STAGES = {"fetch": fetch, "validate": validate, "load": load}


def record_run(context, status: str, error: str | None = None) -> None:
    config, record = context.config, context.record
    stats_path = work_dir(config) / STATS_FILE
    raw = {f"{year}.json": sha256_file(year_file(config, year)) for year in config.ews.holidays_years
           if year_file(config, year).exists()}
    manifest = {
        **manifest_header(context, DATASET, status, error, snapshot_id(config)),
        "source": {"url": config.ews.holidays_url, "years": list(config.ews.holidays_years), "raw_sha256": raw},
        "requests": {"this_run": record.http, "full_download": len(config.ews.holidays_years)},
        "counts": json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else None,
        "outputs": describe_outputs({f"work/{DATASET}/holidays.csv": work_dir(config) / "holidays.csv",
                                     f"quarantine/{DATASET}/{QUARANTINE_FILE}": quarantine_dir(config) / QUARANTINE_FILE}),
        "load": record.load or None,
    }
    save_manifest(manifest, config.paths.curated / DATASET, work_dir(config) / "manifests")


def run(context, stage_names: list[str]) -> None:
    run_stages(context, DATASET, stage_names, STAGES, record_run)
