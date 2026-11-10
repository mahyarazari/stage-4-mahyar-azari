"""Stages of the credit dataset (stage 3): fetch -> validate -> load.

The same factory as Olist: pinned checksums, the table-driven contract with quarantine and
its threshold, the PostgreSQL sink (one transaction) and the run manifest. What is new is
the kind of source: a local file instead of a URL. A Kaggle competition file needs a login
and accepting the competition rules, so the fetch stage does not download it; it checks
that the file placed in data/raw/credit/ is byte for byte the pinned one (D60).
"""
from __future__ import annotations

import json

from data_factory.config import Config
from data_factory.credit_contract import credit_tables
from data_factory.db import database
from data_factory.exceptions import ChecksumMismatchError, MissingInputError
from data_factory.file_source import sha256_bytes, sha256_file
from data_factory.loading import TableLoad, load_tables
from data_factory.dataset_runs import manifest_header, run_stages
from data_factory.manifest import build_code_inputs, describe_outputs, save_manifest
from data_factory.olist_pipeline import _reasons_to_array
from data_factory.sinks import PostgresSink
from data_factory.table_validation import QUARANTINE_FILE, STATS_FILE, validate_tables

DATASET = "credit"
SOURCE_PAGE = "https://www.kaggle.com/competitions/GiveMeSomeCredit/data"
STAGE_NAMES = ("fetch", "validate", "load")


def raw_dir(config: Config):
    return config.paths.raw / DATASET


def work_dir(config: Config):
    return config.paths.work / DATASET


def quarantine_dir(config: Config):
    return config.paths.quarantine / DATASET


def snapshot_id(config: Config) -> str:
    joined = "\n".join(f"{name}={digest}" for name, digest in sorted(config.credit.files.items()))
    return "credit-" + sha256_bytes(joined.encode())[:12]


def fetch(context) -> None:
    config = context.config
    for name, expected in sorted(config.credit.files.items()):
        path = raw_dir(config) / name
        if not path.exists():
            raise MissingInputError(f"{path} not found: download {name} from {SOURCE_PAGE} "
                                    f"(accept the competition rules first) and place it there")
        actual = sha256_file(path)
        if actual != expected:
            raise ChecksumMismatchError(f"{name}: sha256 {actual} does not match the pinned {expected}")


def validate(context) -> None:
    config = context.config
    validate_tables(credit_tables(config.credit.missing_tokens), raw_dir(config), work_dir(config),
                    quarantine_dir(config), config.clean.chunk_rows, config.max_quarantine_ratio, {}, DATASET)


def load(context) -> None:
    config = context.config
    items = [TableLoad(spec.table, work_dir(config) / f"{spec.table}.csv", lambda frame: frame)
             for spec in credit_tables(config.credit.missing_tokens)]
    items.append(TableLoad("quarantine", quarantine_dir(config) / QUARANTINE_FILE, _reasons_to_array))
    inputs = build_code_inputs(config.path)
    run = {"load_id": context.record.run_id, "snapshot": snapshot_id(config),
           "config_sha256": inputs["config_sha256"], "git_commit": inputs["git_commit"]}
    with database(config.database) as engine:
        sink = PostgresSink(engine, config.credit.load.schema, config.credit.load.before_load,
                            config.credit.load.after_load)
        stats = load_tables(items, sink, config.clean.chunk_rows, run)
    context.record.load = {"schema": config.credit.load.schema, "load_id": stats.load_id, "rows": stats.rows}


STAGES = {"fetch": fetch, "validate": validate, "load": load}


def record_run(context, status: str, error: str | None = None) -> None:
    config, record = context.config, context.record
    stats_path = work_dir(config) / STATS_FILE
    outputs = {f"work/{DATASET}/{spec.table}.csv": work_dir(config) / f"{spec.table}.csv"
               for spec in credit_tables(config.credit.missing_tokens)}
    outputs[f"quarantine/{DATASET}/{QUARANTINE_FILE}"] = quarantine_dir(config) / QUARANTINE_FILE
    present = {name: sha256_file(raw_dir(config) / name) for name in sorted(config.credit.files)
               if (raw_dir(config) / name).exists()}
    manifest = {
        **manifest_header(context, DATASET, status, error, snapshot_id(config)),
        "source": {"page": SOURCE_PAGE, "files_sha256": present,
                   "all_checksums_match": present == dict(sorted(config.credit.files.items()))},
        "counts": json.loads(stats_path.read_text("utf-8")) if stats_path.exists() else None,
        "outputs": describe_outputs(outputs),
        "load": record.load or None,
    }
    save_manifest(manifest, config.paths.curated / DATASET, work_dir(config) / "manifests")


def run(context, stage_names: list[str]) -> None:
    run_stages(context, DATASET, stage_names, STAGES, record_run)
