"""Load and validate the factory configuration (a TOML file).

Every tunable value of the factory lives in the configuration file, never in the code.
Relative paths are resolved against the folder of the configuration file, so the factory
behaves the same whatever folder it is started from.

TOML is read with `tomllib` from the standard library (Python 3.11+), so no extra
dependency is needed. Unlike JSON, TOML allows comments, which keeps each setting's
reason next to the setting.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data_factory.contract import BOROUGHS, Scope
from data_factory.exceptions import ConfigError

ZIP_PATTERN = re.compile(r"\d{5}")
AGGREGATE_LEVELS = ("borough", "modzcta")
DATABASE_URL_ENV = "DATABASE_URL"            # replaces [database] url when set (D39)
DATABASE_URL_PREFIX = "postgresql+psycopg://"  # SQLAlchemy dialect + psycopg 3 driver (D38)
SCHEMA_PATTERN = re.compile(r"[a-z_][a-z0-9_]{0,62}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceConfig:
    records_url: str
    metadata_url: str


@dataclass(frozen=True)
class FetchConfig:
    page_size: int
    max_retries: int
    backoff_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    curated: Path
    quarantine: Path
    work: Path        # intermediate results passed from one stage to the next


@dataclass(frozen=True)
class ReferenceConfig:
    population_url: str
    zcta_county_url: str
    county_boroughs: dict[str, str]   # county FIPS code -> borough name as written in 311


@dataclass(frozen=True)
class ZipConfig:
    placeholders: tuple[str, ...]
    aliases: dict[str, str]


@dataclass(frozen=True)
class AggregateConfig:
    levels: tuple[str, ...]
    min_cases_per_cell: int
    rate_per: int            # rates are complaints per this many residents


@dataclass(frozen=True)
class CleanConfig:
    chunk_rows: int                       # rows processed at a time
    unspecified_values: tuple[str, ...]   # source text meaning "unknown" (becomes missing)
    latitude_range: tuple[float, float]   # coordinates outside the city become missing
    longitude_range: tuple[float, float]


@dataclass(frozen=True)
class DatabaseConfig:
    url: str        # SQLAlchemy URL without a password; the driver reads PGPASSWORD / pgpass (D39)
    source: str     # where the URL came from: the config file or the DATABASE_URL variable
    maintenance_database: str = "postgres"   # where CREATE DATABASE is run from


@dataclass(frozen=True)
class LoadConfig:
    schema: str                    # target schema of the load stage
    before_load: tuple[Path, ...]  # SQL files run before the tables are filled (tables)
    after_load: tuple[Path, ...]   # SQL files run after (indexes, views, functions)


@dataclass(frozen=True)
class OlistConfig:
    base_url: str                          # folder URL of the nine files (D35)
    files: dict[str, str]                  # file name -> pinned sha256
    latitude_range: tuple[float, float]    # geolocation points outside Brazil become missing
    longitude_range: tuple[float, float]
    load: LoadConfig


@dataclass(frozen=True)
class ModelConfig:
    seed: int                    # the one seed of the stage-3 notebook and of the model
    test_share: float            # share of applicants set aside once, at the start (D61)
    cv_folds: int                # stratified folds on the rest (D61)
    cost_fn: float               # cost of approving an applicant who then defaults
    cost_fp: float               # cost of rejecting an applicant who would have repaid
    alt_cost_fn: float           # the second cost ratio of the sensitivity check
    model_file: Path             # trained pipeline + threshold (joblib)


@dataclass(frozen=True)
class CreditConfig:
    files: dict[str, str]        # file name -> pinned sha256 (Kaggle download, D60)
    missing_tokens: tuple[str, ...]
    load: LoadConfig
    model: ModelConfig


@dataclass(frozen=True)
class EwsConfig:
    """Stage 4, early-warning system: negative reviews predicted at approval time."""
    holidays_url: str            # BrasilAPI national holidays, one request per year (D74)
    holidays_years: tuple[int, ...]
    load: LoadConfig
    cutoff: pd.Timestamp         # train on decisions before it, evaluate after (D76)
    evaluation_end: pd.Timestamp  # end of the evaluation period (exclusive)
    capacity: float              # share of orders the support team can follow up (operating point)
    capacities: tuple[float, ...]  # capacities of the manager's table
    seed: int
    feature_budget_ms: float     # limit for the feature query (server execution time)
    output_dir: Path             # model, follow-up list, timings, plans


@dataclass(frozen=True)
class DashboardConfig:
    start: pd.Timestamp       # period the dashboard queries are measured on
    end: pd.Timestamp         # (end exclusive)
    top_n: int                # number of sellers in Q4
    budget_ms: float          # 95th-percentile limit per query
    warmup_runs: int
    timed_runs: int


@dataclass(frozen=True)
class Config:
    path: Path
    source: SourceConfig
    scope: Scope
    as_of: pd.Timestamp   # cut-off: a case closed after it counts as still open (D24)
    fetch: FetchConfig
    paths: PathsConfig
    references: ReferenceConfig
    zip: ZipConfig
    max_quarantine_ratio: float
    clean: CleanConfig
    aggregate: AggregateConfig
    database: DatabaseConfig
    load: LoadConfig
    olist: OlistConfig
    credit: CreditConfig
    ews: EwsConfig
    dashboard: DashboardConfig


# ---------------------------------------------------------------- small readers
def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing section [{name}]")
    return value


def _get(section: dict[str, Any], section_name: str, key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in section:
        raise ConfigError(f"missing key '{key}' in [{section_name}]")
    value = section[key]
    # bool is a subclass of int: refuse `true` where a number is expected.
    if not isinstance(value, kind) or (isinstance(value, bool) and bool not in _as_tuple(kind)):
        raise ConfigError(f"[{section_name}] {key} must be {_kind_name(kind)}, got {value!r}")
    return value


def _as_tuple(kind: type | tuple[type, ...]) -> tuple[type, ...]:
    return kind if isinstance(kind, tuple) else (kind,)


def _kind_name(kind: type | tuple[type, ...]) -> str:
    return " or ".join(k.__name__ for k in _as_tuple(kind))


def _positive(value: float, where: str) -> float:
    if value <= 0:
        raise ConfigError(f"{where} must be greater than 0, got {value}")
    return value


def _text_list(section: dict[str, Any], section_name: str, key: str) -> tuple[str, ...]:
    values = _get(section, section_name, key, list)
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise ConfigError(f"[{section_name}] {key} must be a non-empty list of text values")
    return tuple(values)


# ---------------------------------------------------------------- sections
def _read_scope(data: dict[str, Any]) -> Scope:
    section = _section(data, "scope")
    start = _get(section, "scope", "window_start", datetime)
    end = _get(section, "scope", "window_end", datetime)
    if start.tzinfo or end.tzinfo:
        raise ConfigError("[scope] window dates must not carry a time zone (source dates are local NYC time)")
    if start >= end:
        raise ConfigError(f"[scope] window_start ({start}) must be before window_end ({end})")
    return Scope(
        agencies=_text_list(section, "scope", "agencies"),
        complaint_types=_text_list(section, "scope", "complaint_types"),
        window_start=pd.Timestamp(start),
        window_end=pd.Timestamp(end),
    )


def _read_as_of(data: dict[str, Any], scope: Scope) -> pd.Timestamp:
    section = _section(data, "scope")
    as_of = _get(section, "scope", "as_of", datetime)
    if as_of.tzinfo:
        raise ConfigError("[scope] as_of must not carry a time zone")
    if pd.Timestamp(as_of) < scope.window_end:
        raise ConfigError(f"[scope] as_of ({as_of}) must not be before window_end ({scope.window_end})")
    return pd.Timestamp(as_of)


def _read_fetch(data: dict[str, Any]) -> FetchConfig:
    section = _section(data, "fetch")
    max_retries = _get(section, "fetch", "max_retries", int)
    if max_retries < 0:
        raise ConfigError(f"[fetch] max_retries must be 0 or more, got {max_retries}")
    return FetchConfig(
        page_size=int(_positive(_get(section, "fetch", "page_size", int), "[fetch] page_size")),
        max_retries=max_retries,
        backoff_seconds=float(_positive(_get(section, "fetch", "backoff_seconds", (int, float)),
                                        "[fetch] backoff_seconds")),
        timeout_seconds=float(_positive(_get(section, "fetch", "timeout_seconds", (int, float)),
                                        "[fetch] timeout_seconds")),
    )


def _read_paths(data: dict[str, Any], base: Path) -> PathsConfig:
    section = _section(data, "paths")

    def resolve(key: str) -> Path:
        path = Path(_get(section, "paths", key, str))
        return path if path.is_absolute() else (base / path).resolve()

    return PathsConfig(raw=resolve("raw"), curated=resolve("curated"),
                       quarantine=resolve("quarantine"), work=resolve("work"))


def _read_references(data: dict[str, Any]) -> ReferenceConfig:
    section = _section(data, "references")
    counties = _get(section, "references", "county_boroughs", dict)
    for code, borough in counties.items():
        if not re.fullmatch(r"\d{5}", code) or borough not in BOROUGHS:
            raise ConfigError(f"[references.county_boroughs] '{code}' = {borough!r}: "
                              f"need a 5-digit county code and one of {list(BOROUGHS)}")
    if sorted(counties.values()) != sorted(BOROUGHS):
        raise ConfigError("[references.county_boroughs] must map exactly one county to each borough")
    return ReferenceConfig(
        population_url=_get(section, "references", "population_url", str),
        zcta_county_url=_get(section, "references", "zcta_county_url", str),
        county_boroughs=dict(counties),
    )


def _read_zip(data: dict[str, Any]) -> ZipConfig:
    section = _section(data, "zip")
    placeholders = _get(section, "zip", "placeholders", list)
    if not all(isinstance(p, str) for p in placeholders):
        raise ConfigError("[zip] placeholders must be a list of text values")
    aliases = _get(section, "zip", "aliases", dict)
    for old, new in aliases.items():
        if not (isinstance(new, str) and ZIP_PATTERN.fullmatch(old) and ZIP_PATTERN.fullmatch(new)):
            raise ConfigError(f"[zip.aliases] '{old}' = {new!r}: both sides must be 5-digit zips")
    return ZipConfig(placeholders=tuple(placeholders), aliases=dict(aliases))


def _read_quarantine(data: dict[str, Any]) -> float:
    section = _section(data, "quarantine")
    ratio = float(_get(section, "quarantine", "max_ratio", (int, float)))
    if not 0 < ratio <= 1:
        raise ConfigError(f"[quarantine] max_ratio must be in (0, 1], got {ratio}")
    return ratio


def _range(section: dict[str, Any], key: str, section_name: str = "clean") -> tuple[float, float]:
    values = _get(section, section_name, key, list)
    if (len(values) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            or values[0] >= values[1]):
        raise ConfigError(f"[{section_name}] {key} must be [low, high] with low < high, got {values!r}")
    return float(values[0]), float(values[1])


def _read_clean(data: dict[str, Any]) -> CleanConfig:
    section = _section(data, "clean")
    return CleanConfig(
        chunk_rows=int(_positive(_get(section, "clean", "chunk_rows", int), "[clean] chunk_rows")),
        unspecified_values=_text_list(section, "clean", "unspecified_values"),
        latitude_range=_range(section, "latitude_range"),
        longitude_range=_range(section, "longitude_range"),
    )


def _read_aggregate(data: dict[str, Any]) -> AggregateConfig:
    section = _section(data, "aggregate")
    levels = _text_list(section, "aggregate", "levels")
    unknown = sorted(set(levels) - set(AGGREGATE_LEVELS))
    if unknown:
        raise ConfigError(f"[aggregate] unknown levels {unknown}; allowed: {list(AGGREGATE_LEVELS)}")
    min_cases = _get(section, "aggregate", "min_cases_per_cell", int)
    rate_per = _get(section, "aggregate", "rate_per", int)
    return AggregateConfig(levels=levels,
                           min_cases_per_cell=int(_positive(min_cases, "[aggregate] min_cases_per_cell")),
                           rate_per=int(_positive(rate_per, "[aggregate] rate_per")))


def _read_database(data: dict[str, Any]) -> DatabaseConfig:
    section = _section(data, "database")
    url, source = _get(section, "database", "url", str), "factory.toml"
    if os.environ.get(DATABASE_URL_ENV):
        url, source = os.environ[DATABASE_URL_ENV], DATABASE_URL_ENV
    if not url.startswith(DATABASE_URL_PREFIX):
        raise ConfigError(f"database url from {source} must start with '{DATABASE_URL_PREFIX}'")
    maintenance = _get(section, "database", "maintenance_database", str)
    return DatabaseConfig(url=url, source=source, maintenance_database=maintenance)


def _read_load(data: dict[str, Any], base: Path, name: str = "load") -> LoadConfig:
    section = data.get(name) if name == "load" else _section(data, name.split(".")[0]).get("load")
    if not isinstance(section, dict):
        raise ConfigError(f"missing section [{name}]")
    schema = _get(section, name, "schema", str)
    if not SCHEMA_PATTERN.fullmatch(schema):
        raise ConfigError(f"[{name}] schema must be a lower-case SQL name, got {schema!r}")

    def files(key: str) -> tuple[Path, ...]:
        values = _get(section, name, key, list)
        if not all(isinstance(v, str) and v.endswith(".sql") for v in values):
            raise ConfigError(f"[{name}] {key} must be a list of .sql file paths")
        # Existence is checked by the load stage, so a copied config file stays loadable.
        return tuple(Path(v) if Path(v).is_absolute() else (base / v).resolve() for v in values)

    return LoadConfig(schema=schema, before_load=files("before_load"), after_load=files("after_load"))


def _read_olist(data: dict[str, Any], base: Path) -> OlistConfig:
    section = _section(data, "olist")
    files = _get(section, "olist", "files", dict)
    for name, digest in files.items():
        if not (name.endswith(".csv") and isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest)):
            raise ConfigError(f"[olist.files] '{name}' must be a .csv name with a 64-character sha256")
    load = _read_load(data, base, name="olist.load")
    if load.schema == _read_load(data, base).schema:
        raise ConfigError("[olist.load] schema must differ from [load] schema")
    return OlistConfig(base_url=_get(section, "olist", "base_url", str).rstrip("/"), files=dict(files),
                       latitude_range=_range(section, "latitude_range", "olist"),
                       longitude_range=_range(section, "longitude_range", "olist"), load=load)


def _read_credit(data: dict[str, Any], base: Path) -> CreditConfig:
    section = _section(data, "credit")
    files = _get(section, "credit", "files", dict)
    for name, digest in files.items():
        if not (name.endswith(".csv") and isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest)):
            raise ConfigError(f"[credit.files] '{name}' must be a .csv name with a 64-character sha256")
    load = _read_load(data, base, name="credit.load")
    model = section.get("model")
    if not isinstance(model, dict):
        raise ConfigError("missing section [credit.model]")
    where = "credit.model"
    test_share = float(_get(model, where, "test_share", float))
    if not 0 < test_share < 1:
        raise ConfigError(f"[{where}] test_share must be between 0 and 1, got {test_share}")
    folds = int(_get(model, where, "cv_folds", int))
    if folds < 2:
        raise ConfigError(f"[{where}] cv_folds must be at least 2, got {folds}")
    model_file = Path(_get(model, where, "model_file", str))
    return CreditConfig(
        files=dict(files), missing_tokens=_text_list(section, "credit", "missing_tokens"), load=load,
        model=ModelConfig(
            seed=int(_get(model, where, "seed", int)), test_share=test_share, cv_folds=folds,
            cost_fn=float(_positive(_get(model, where, "cost_fn", (int, float)), f"[{where}] cost_fn")),
            cost_fp=float(_positive(_get(model, where, "cost_fp", (int, float)), f"[{where}] cost_fp")),
            alt_cost_fn=float(_positive(_get(model, where, "alt_cost_fn", (int, float)), f"[{where}] alt_cost_fn")),
            model_file=model_file if model_file.is_absolute() else (base / model_file).resolve()))


def _read_ews(data: dict[str, Any], base: Path) -> EwsConfig:
    section = _section(data, "ews")
    where = "ews"
    years = _get(section, where, "holidays_years", list)
    if not years or not all(isinstance(y, int) and not isinstance(y, bool) for y in years):
        raise ConfigError(f"[{where}] holidays_years must be a non-empty list of years")
    cutoff = pd.Timestamp(_get(section, where, "cutoff", datetime))
    end = pd.Timestamp(_get(section, where, "evaluation_end", datetime))
    if cutoff >= end:
        raise ConfigError(f"[{where}] cutoff ({cutoff}) must be before evaluation_end ({end})")
    capacity = float(_get(section, where, "capacity", float))
    capacities = tuple(float(c) for c in _get(section, where, "capacities", list))
    if not all(0 < c < 1 for c in (capacity, *capacities)):
        raise ConfigError(f"[{where}] capacity and capacities must be shares between 0 and 1")
    output = Path(_get(section, where, "output_dir", str))
    load = _read_load(data, base, name="ews.load")
    return EwsConfig(
        holidays_url=_get(section, where, "holidays_url", str).rstrip("/"), holidays_years=tuple(sorted(years)),
        load=load, cutoff=cutoff, evaluation_end=end, capacity=capacity, capacities=capacities,
        seed=int(_get(section, where, "seed", int)),
        feature_budget_ms=float(_positive(_get(section, where, "feature_budget_ms", (int, float)),
                                          f"[{where}] feature_budget_ms")),
        output_dir=output if output.is_absolute() else (base / output).resolve())


def _read_dashboard(data: dict[str, Any]) -> DashboardConfig:
    section = _section(data, "dashboard")
    start = _get(section, "dashboard", "start", datetime)
    end = _get(section, "dashboard", "end", datetime)
    if start >= end:
        raise ConfigError(f"[dashboard] start ({start}) must be before end ({end})")
    return DashboardConfig(
        start=pd.Timestamp(start), end=pd.Timestamp(end),
        top_n=int(_positive(_get(section, "dashboard", "top_n", int), "[dashboard] top_n")),
        budget_ms=float(_positive(_get(section, "dashboard", "budget_ms", (int, float)), "[dashboard] budget_ms")),
        warmup_runs=int(_get(section, "dashboard", "warmup_runs", int)),
        timed_runs=int(_positive(_get(section, "dashboard", "timed_runs", int), "[dashboard] timed_runs")))


# ---------------------------------------------------------------- entry point
def load_config(path: str | Path) -> Config:
    """Read, validate and return the configuration. Any problem raises ConfigError."""
    path = Path(path).resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"configuration file is not valid TOML ({path}): {error}") from None

    source = _section(data, "source")
    scope = _read_scope(data)
    return Config(
        path=path,
        source=SourceConfig(
            records_url=_get(source, "source", "records_url", str),
            metadata_url=_get(source, "source", "metadata_url", str),
        ),
        scope=scope,
        as_of=_read_as_of(data, scope),
        fetch=_read_fetch(data),
        paths=_read_paths(data, base=path.parent),
        references=_read_references(data),
        zip=_read_zip(data),
        max_quarantine_ratio=_read_quarantine(data),
        clean=_read_clean(data),
        aggregate=_read_aggregate(data),
        database=_read_database(data),
        load=_read_load(data, base=path.parent),
        olist=_read_olist(data, base=path.parent),
        credit=_read_credit(data, base=path.parent),
        ews=_read_ews(data, base=path.parent),
        dashboard=_read_dashboard(data),
    )
