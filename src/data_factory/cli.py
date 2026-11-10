"""Command line entry point: `data-factory` (or `python -m data_factory`).

Examples:
    data-factory run                          # both datasets, every stage: empty database -> full
    data-factory run --dataset olist          # only the Olist dataset
    data-factory run --stages fetch           # only one stage
    data-factory run --stages load            # only load the published outputs into PostgreSQL
    data-factory run --since-last-run         # only records newer than the previous run
    data-factory check-config                 # validate the configuration and stop
    data-factory check-db                     # connect to the database and print its version
    data-factory benchmark --label baseline   # time the dashboard queries, save plans
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import text

from data_factory import benchmark, credit_pipeline, holidays_pipeline, olist_pipeline, pipeline
from data_factory.config import load_config
from data_factory.db import database, ensure_database, safe_url, server_version
from data_factory.exceptions import ExitCode, FactoryError

log = logging.getLogger("data_factory")

DEFAULT_CONFIG = Path("factory.toml")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


DATASETS = ("nyc311", "olist", "credit", "holidays")
PIPELINES = {"olist": olist_pipeline, "credit": credit_pipeline, "holidays": holidays_pipeline}


def _stage_order(dataset: str) -> tuple[str, ...]:
    return pipeline.STAGE_NAMES if dataset == "nyc311" else PIPELINES[dataset].STAGE_NAMES


def _stage_list(text: str) -> list[str]:
    """Parse 'fetch,clean' into stage names; each dataset later runs the ones it has, in its order."""
    requested = [name.strip() for name in text.split(",") if name.strip()]
    known = set(pipeline.STAGE_NAMES) | set(olist_pipeline.STAGE_NAMES)
    unknown = [name for name in requested if name not in known]
    if not requested or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {unknown}; nyc311: {list(pipeline.STAGE_NAMES)}, "
            f"olist: {list(olist_pipeline.STAGE_NAMES)}")
    return requested


def _run_dataset(dataset: str, config, args) -> None:
    stages = [name for name in _stage_order(dataset) if args.stages is None or name in args.stages]
    if not stages:
        return
    log.info("dataset %s: stages %s", dataset, stages)
    if dataset == "nyc311":
        pipeline.run(pipeline.RunContext(config=config, since_last_run=args.since_last_run), stages)
    else:
        PIPELINES[dataset].run(pipeline.RunContext(config=config), stages)


def _benchmark(config, label: str, query_set: str = "baseline") -> None:
    path = config.paths.work / "benchmarks" / f"{label}.json"
    with database(config.database) as engine, engine.connect() as connection:
        connection.execute(text("SELECT set_config('search_path', :path, false)"),
                           {"path": config.olist.load.schema})
        timings = benchmark.run_dashboard(connection, config.dashboard, label, query_set)
    benchmark.save_timings(timings, path)
    for t in timings:
        log.info("%-40s median %8.2f ms  p95 %8.2f ms  max %8.2f ms  server %8.2f ms  rows %5d  %s",
                 t.query, t.median_ms, t.p95_ms, t.max_ms, t.server_ms, t.rows,
                 "OK" if t.within_budget else "OVER BUDGET")
    log.info("saved timings and plans: %s", path)


def _index_evidence(config) -> None:
    from datetime import date
    from data_factory import index_evidence as ev
    month = date(2018, 1, 1)                     # a busy month (7,269 orders)
    schema = config.olist.load.schema
    with database(config.database) as engine:
        result = {"month": str(month), "with_and_without": ev.with_and_without(engine, schema, month),
                  "sizes": ev.sizes(engine, schema), "insert_cost_ms": ev.insert_cost(engine, schema)}
    path = config.paths.work / "benchmarks" / "index_evidence.json"
    path.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    for index, r in result["with_and_without"].items():
        log.info("%s (%d KB): used by planner: %s; refresh %s ms without -> %s ms with; plan without %s; with %s",
                 index, r["with"]["bytes"] // 1024, r["used_by_planner"], r["without"]["refresh_ms"],
                 r["with"]["refresh_ms"], r["without"]["plan"]["nodes"], r["with"]["plan"]["nodes"])
    log.info("insert of all orders (ms): %s", result["insert_cost_ms"])
    for row in result["sizes"]:
        log.info("size %-34s %-16s index %8d KB  table %8d KB", row["index"], row["table_name"],
                 row["index_bytes"] // 1024, row["table_bytes"] // 1024)
    log.info("saved: %s", path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-factory",
                                     description="Build a clean NYC 311 dataset from the live API.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"configuration file (default: {DEFAULT_CONFIG})")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the factory (all stages by default)")
    run.add_argument("--dataset", choices=("all",) + DATASETS, default="all",
                     help="which dataset to build (default: all, nyc311, olist, then credit)")
    run.add_argument("--stages", type=_stage_list, default=None,
                     help=f"comma-separated subset; nyc311: {','.join(pipeline.STAGE_NAMES)}; "
                          f"olist and credit: {','.join(olist_pipeline.STAGE_NAMES)}")
    run.add_argument("--since-last-run", action="store_true",
                     help="fetch only records created after the previous run")

    commands.add_parser("check-config", help="load and validate the configuration, then stop")
    commands.add_parser("check-db", help="connect to the database and print the server version")
    bench = commands.add_parser("benchmark", help="time the dashboard queries and save their plans")
    bench.add_argument("--label", required=True, help="name of this measurement (e.g. baseline)")
    bench.add_argument("--set", dest="query_set", choices=tuple(benchmark.QUERY_SETS), default="baseline",
                       help="which versions of the queries to time (default: baseline)")
    commands.add_parser("index-evidence", help="plans, sizes and write cost of the indexes (rolled back)")
    predict = commands.add_parser("predict", help="score a new credit file with the saved model")
    predict.add_argument("--input", type=Path, required=True, help="CSV with the schema of cs-training.csv")
    predict.add_argument("--output", type=Path, required=True, help="CSV of probabilities and decisions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        config = load_config(args.config)
        if args.command == "check-config":
            log.info("configuration is valid: %s", config.path)
            return ExitCode.OK
        if args.command == "index-evidence":
            _index_evidence(config)
            return ExitCode.OK
        if args.command == "benchmark":
            _benchmark(config, args.label, args.query_set)
            return ExitCode.OK
        if args.command == "predict":
            from data_factory.credit_predict import predict_file
            result = predict_file(config, args.input, args.output)
            log.info("predict: %d applicants scored, %d rejected -> %s", len(result),
                     int((result["decision"] == "reject").sum()), args.output)
            return ExitCode.OK
        if args.command == "check-db":
            with database(config.database) as engine:
                log.info("connected to %s (url from %s): %s", safe_url(config.database.url),
                         config.database.source, server_version(engine))
            return ExitCode.OK
        if args.stages is None or "load" in args.stages:
            ensure_database(config.database)        # an empty server is enough (one command)
        for dataset in (DATASETS if args.dataset == "all" else (args.dataset,)):
            _run_dataset(dataset, config, args)
        return ExitCode.OK
    except FactoryError as error:
        log.error("%s: %s", type(error).__name__, error)
        return error.exit_code
    except Exception:  # a bug: show the full traceback
        log.exception("unexpected error")
        return ExitCode.UNEXPECTED
