"""Validate tables against their contract (`olist_contract.TABLES`), chunk by chunk.

For every source file, in parent-before-child order:
  - the header must contain every contract column, otherwise the source changed shape
    and the run stops (SourceSchemaChangedError), as with the 311 metadata check (D4);
  - each value is checked against its kind; required and empty -> MISSING_REQUIRED,
    present but malformed -> INVALID_VALUE;
  - a row whose foreign key (policy QUARANTINE) has no accepted parent -> ORPHAN_FOREIGN_KEY;
  - a second row with an accepted key -> DUPLICATE_KEY;
  - a foreign key with policy COMPLETE is only counted (the database adds the parent).

Accepted rows go to <work>/<table>.csv with the database column names; rejected rows go to
one quarantine file with their table, row number, raw values (JSON) and every reason. As in
stage 1 (D9) the quarantine file is always kept, and the run fails when the quarantined
share is above the configured limit. Only one chunk of a file is in memory at a time; the
sets of accepted keys are kept for the foreign-key checks.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pandas as pd

from data_factory.exceptions import MissingInputError, QuarantineThresholdError, SourceSchemaChangedError
from data_factory.olist_contract import Column, FkPolicy, Kind, Reason, STATES, TableSpec
from data_factory.tables import AtomicCsvWriter, read_csv_text

log = logging.getLogger(__name__)

QUARANTINE_FILE = "quarantine.csv"
QUARANTINE_COLUMNS = ["source_table", "source_row", "record", "reason_codes"]
STATS_FILE = "validate_stats.json"
REASON_SEPARATOR = "|"
KEY_SEPARATOR = "\x1f"

PATTERNS = {
    Kind.ID: r"[0-9a-f]{32}",
    Kind.COUNT: r"[0-9]+",
    Kind.MONEY: r"[0-9]+(\.[0-9]{1,2})?",
    Kind.DECIMAL: r"[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?",
    Kind.ZIP: r"[0-9]{5}",
    Kind.TIMESTAMP: r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}",
    Kind.DATE: r"[0-9]{4}-[0-9]{2}-[0-9]{2} 00:00:00",
    Kind.DAY: r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY_FORMAT = "%Y-%m-%d"


def check_value(values: pd.Series, column: Column, bounds: dict[str, tuple[float, float]]):
    """(value to store, malformed mask) for one column; missing values are never malformed."""
    present = values.notna()
    kind = column.kind
    if kind in PATTERNS:
        bad = present & ~values.str.fullmatch(PATTERNS[kind]).fillna(False)
        if kind in (Kind.TIMESTAMP, Kind.DATE, Kind.DAY):
            parsed = pd.to_datetime(values.where(~bad), format=DAY_FORMAT if kind == Kind.DAY else TIMESTAMP_FORMAT,
                                    errors="coerce")
            bad |= present & ~bad & parsed.isna()          # e.g. 2017-02-30
        if kind == Kind.DATE:
            values = values.str.slice(0, 10)
        return values, bad
    if kind == Kind.STATE:
        return values, present & ~values.isin(STATES)
    if kind == Kind.CHOICE:
        return values, present & ~values.isin(column.values)
    if kind == Kind.COORDINATE:
        number = pd.to_numeric(values, errors="coerce")
        low, high = bounds[column.target]
        inside = number.between(low, high)
        return values.where(inside), present & number.isna()   # outside the box: missing, kept
    return values, pd.Series(False, index=values.index)          # TEXT


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    key = frame[columns[0]].astype("string")
    for column in columns[1:]:
        key = key + KEY_SEPARATOR + frame[column].astype("string")
    return key


def _reason_text(flags: dict[Reason, pd.Series], index) -> pd.Series:
    text = pd.Series("", index=index, dtype="string")
    for reason in Reason:                                         # fixed order
        mark = flags[reason]
        text = text.mask(mark, text + REASON_SEPARATOR + reason.value)
    return text.str.lstrip(REASON_SEPARATOR)


def _records(raw: pd.DataFrame) -> pd.Series:
    return pd.Series([json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()},
                                 ensure_ascii=False, sort_keys=True)
                      for row in raw.to_dict("records")], index=raw.index, dtype="string")


class _TableResult:
    def __init__(self):
        self.raw = 0
        self.accepted = 0
        self.quarantined = 0
        self.reasons = Counter()
        self.completed: dict[str, dict[str, int]] = {}


def validate_tables(tables: tuple[TableSpec, ...], raw_dir: Path, work_dir: Path, quarantine_dir: Path,
                    chunk_rows: int, max_ratio: float, bounds: dict[str, tuple[float, float]],
                    dataset: str = "olist") -> dict:
    for spec in tables:
        if not (raw_dir / spec.file).exists():
            raise MissingInputError(f"{raw_dir / spec.file} not found: run the fetch stage first")
    needed = {(fk.parent, fk.parent_column) for spec in tables for fk in spec.foreign_keys}
    parent_values: dict[tuple[str, str], set[str]] = {pair: set() for pair in needed}
    results: dict[str, _TableResult] = {}

    with ExitStack() as stack:
        bad_out = stack.enter_context(AtomicCsvWriter(quarantine_dir / QUARANTINE_FILE, QUARANTINE_COLUMNS))
        good_out = {spec.table: stack.enter_context(AtomicCsvWriter(work_dir / f"{spec.table}.csv", spec.targets))
                    for spec in tables}
        for spec in tables:
            results[spec.table] = _validate_table(spec, raw_dir / spec.file, chunk_rows, bounds,
                                                  parent_values, needed, good_out[spec.table], bad_out)

        raw = sum(r.raw for r in results.values())
        quarantined = sum(r.quarantined for r in results.values())
        ratio = quarantined / raw if raw else 0.0
        stats = {
            "raw_records": raw, "valid_records": raw - quarantined, "quarantined_records": quarantined,
            "quarantine_ratio": round(ratio, 8), "max_quarantine_ratio": max_ratio,
            "tables": {name: {"raw": r.raw, "accepted": r.accepted, "quarantined": r.quarantined,
                              "reasons": {reason.value: r.reasons[reason.value] for reason in Reason},
                              "completed_parents": r.completed}
                       for name, r in results.items()},
        }
        bad_out.commit()                                           # evidence kept in every case
        if ratio > max_ratio:
            raise QuarantineThresholdError(
                f"{100 * ratio:.4f}% of {dataset} records were quarantined, above the limit of "
                f"{100 * max_ratio:.4f}%; see {quarantine_dir / QUARANTINE_FILE}")
        for writer in good_out.values():
            writer.commit()

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / STATS_FILE).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    log.info("validate %s: %d raw, %d accepted, %d quarantined", dataset, raw, raw - quarantined, quarantined)
    return stats


def _validate_table(spec: TableSpec, path: Path, chunk_rows: int, bounds, parent_values, needed,
                    good_out: AtomicCsvWriter, bad_out: AtomicCsvWriter) -> _TableResult:
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c.source for c in spec.columns if c.source not in header]
    if missing:
        raise SourceSchemaChangedError(f"{path.name}: columns missing from the source: {missing}")

    result = _TableResult()
    seen_keys: set[str] = set()
    completed = {fk.column: Counter() for fk in spec.foreign_keys if fk.policy == FkPolicy.COMPLETE}
    offset = 0
    for chunk in read_csv_text(path, chunksize=chunk_rows):
        chunk.index = range(offset + 1, offset + len(chunk) + 1)     # 1-based data row numbers
        offset += len(chunk)
        raw = chunk[[c.source for c in spec.columns]]
        flags = {reason: pd.Series(False, index=raw.index) for reason in Reason}

        out = pd.DataFrame(index=raw.index)
        for column in spec.columns:
            text = raw[column.source].str.strip()
            text = text.mask((text == "") | text.isin(spec.missing_tokens))
            value, bad = check_value(text, column, bounds)
            out[column.target] = value
            flags[Reason.INVALID_VALUE] |= bad
            if column.required:
                flags[Reason.MISSING_REQUIRED] |= text.isna()

        for rule in spec.rules:      # ISO texts compare in time order; only valid values compared
            earlier, later = out[rule.earlier], out[rule.later]
            both = earlier.notna() & later.notna() & ~flags[Reason.INVALID_VALUE]
            flags[Reason.IMPOSSIBLE_DATES] |= both & (later < earlier).fillna(False)

        for fk in spec.foreign_keys:
            if fk.policy == FkPolicy.QUARANTINE:
                values = out[fk.column]
                flags[Reason.ORPHAN_FOREIGN_KEY] |= values.notna() & ~values.isin(parent_values[(fk.parent, fk.parent_column)])

        if spec.key:
            ok_so_far = ~pd.concat(list(flags.values()), axis=1).any(axis=1)
            key = _key(out, spec.key)
            duplicate = ok_so_far & (key.isin(seen_keys) | (key.duplicated() & ok_so_far))
            flags[Reason.DUPLICATE_KEY] |= duplicate

        rejected = pd.concat(list(flags.values()), axis=1).any(axis=1)
        accepted = out[~rejected]
        good_out.append(accepted)
        if rejected.any():
            bad = raw[rejected]
            bad_out.append(pd.DataFrame({"source_table": spec.table, "source_row": bad.index,
                                         "record": _records(bad), "reason_codes": _reason_text(flags, raw.index)[rejected]}))
            for reason in Reason:
                result.reasons[reason.value] += int(flags[reason].sum())

        if spec.key:
            seen_keys.update(_key(accepted, spec.key).tolist())
        for table, column in needed:
            if table == spec.table:
                parent_values[(table, column)].update(accepted[column].dropna().tolist())
        for fk in spec.foreign_keys:
            if fk.policy == FkPolicy.COMPLETE:
                values = accepted[fk.column].dropna()
                completed[fk.column].update(values[~values.isin(parent_values[(fk.parent, fk.parent_column)])].tolist())

        result.raw += len(raw)
        result.accepted += len(accepted)
        result.quarantined += int(rejected.sum())

    result.completed = {column: {"rows": sum(counter.values()), "missing_parents": len(counter)}
                        for column, counter in completed.items()}
    log.info("validate %s: %d raw, %d accepted, %d quarantined%s", spec.table, result.raw, result.accepted,
             result.quarantined, f"; parents to complete {result.completed}" if result.completed else "")
    return result
