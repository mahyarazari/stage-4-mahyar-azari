"""Stage 4: the whole early-warning system with one command, each stage timed (D84).

    data-factory early-warning      empty server -> follow-up list

  olist      fetch -> validate -> load (stage-2 factory, pinned files)
  holidays   fetch -> validate -> load (external API through the same factory)
  features   build the feature objects in SQL, fill the snapshot, time the feature query
  model      time split, baseline, two model families, choice on a time validation, final
             test, cheat model (leakage cost), drift; saves the model and the evidence
  followup   the daily 10% follow-up lists of the latest month

Every stage writes its seconds into <output_dir>/timings.json, also when a later stage fails.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import text

from data_factory import ews_model as em
from data_factory import holidays_pipeline, olist_pipeline
from data_factory.config import Config
from data_factory.db import database, ensure_database
from data_factory.ews_audit import LABEL, TEMPTING, TIME, audit_table
from data_factory.pipeline import RunContext

log = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "ews"
FEATURES_SQL = SQL_DIR / "20_features.sql"
STAGE_NAMES = ("olist", "holidays", "features", "model", "followup")
VALIDATION_MONTHS = 2.5      # D80: last part of the training period, for choosing the model family
WARMUP_RUNS, TIMED_RUNS = 3, 20


def search_path(config: Config) -> str:
    return f"{config.ews.load.schema},{config.olist.load.schema}"


def use_schemas(connection, path: str) -> None:
    connection.execute(text("SELECT set_config('search_path', :path, false)"), {"path": path})


# ---------------------------------------------------------------- stages
def stage_olist(config: Config) -> dict:
    olist_pipeline.run(RunContext(config=config), list(olist_pipeline.STAGE_NAMES))
    return {}


def stage_holidays(config: Config) -> dict:
    holidays_pipeline.run(RunContext(config=config), list(holidays_pipeline.STAGE_NAMES))
    return {}


def build_features(engine, path: str) -> float:
    """Create the feature objects and fill the snapshot in one transaction; returns seconds."""
    started = time.perf_counter()
    with engine.begin() as connection:
        use_schemas(connection, path)
        connection.exec_driver_sql(FEATURES_SQL.read_text(encoding="utf-8"))
        connection.exec_driver_sql("CALL refresh_features()")
    return time.perf_counter() - started


def time_feature_query(engine, path: str, query: str = em.FEATURE_QUERY) -> tuple[dict, dict]:
    """Server execution time of the feature query (EXPLAIN ANALYZE): warm-up + timed runs.
    Returns (statistics, the plan of the median run)."""
    plans = []
    with engine.connect() as connection:
        use_schemas(connection, path)
        for run in range(WARMUP_RUNS + TIMED_RUNS):
            plan = connection.exec_driver_sql("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query).scalar()
            if run >= WARMUP_RUNS:
                plans.append(plan[0])
    times = sorted(p["Execution Time"] for p in plans)
    stats = {"runs": len(times), "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
             "max_ms": max(times)}
    median_plan = min(plans, key=lambda p: abs(p["Execution Time"] - stats["median_ms"]))
    return stats, median_plan


def stage_features(config: Config) -> dict:
    out = config.ews.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with database(config.database) as engine:
        refresh_seconds = build_features(engine, search_path(config))
        stats, plan = time_feature_query(engine, search_path(config))
    (out / "feature_query_plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    stats.update({"refresh_seconds": round(refresh_seconds, 2), "budget_ms": config.ews.feature_budget_ms,
                  "within_budget": stats["p95_ms"] < config.ews.feature_budget_ms})
    log.info("features: refresh %.1f s; feature query p95 %.1f ms (budget %.0f ms)", refresh_seconds,
             stats["p95_ms"], config.ews.feature_budget_ms)
    return {"feature_query": stats}


def read_all(config: Config) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    with database(config.database) as engine, engine.connect() as connection:
        frame, stats = em.read_features(connection, config.ews.load.schema, config.clean.chunk_rows)
        use_schemas(connection, search_path(config))
        cheat = pd.read_sql_query(text("SELECT * FROM cheat_columns"), connection)
    return frame, stats, cheat


def choose_family(train: pd.DataFrame, seed: int, capacity: float) -> tuple[str, pd.DataFrame]:
    """Model family chosen on the last months of the training period only (the test is not used)."""
    start = train[TIME].max() - pd.DateOffset(days=int(VALIDATION_MONTHS * 30))
    fit_part = train[(train[TIME] < start) & (train["label_known_at"] < start)]
    valid = train[train[TIME] >= start]
    rows = []
    for kind in em.MODEL_KINDS:
        scores = em.fit(kind, fit_part, seed).predict_proba(valid)[:, 1]
        r = em.report(valid[LABEL], scores, valid[TIME], capacity)
        rows.append({"model": kind, "validation_capture": r["capture"], "validation_roc_auc": r["roc_auc"]})
    table = pd.DataFrame(rows).sort_values("validation_capture", ascending=False, ignore_index=True)
    return table.loc[0, "model"], table.assign(validation_from=str(start.date()), fit_rows=len(fit_part),
                                               validation_rows=len(valid))


def stage_model(config: Config) -> dict:
    E, out = config.ews, config.ews.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame, read_stats, cheat = read_all(config)
    split = em.temporal_split(frame, E.cutoff, E.evaluation_end)
    train, test = split.train, split.test
    y_test = test[LABEL].astype(bool).to_numpy()

    column, baseline_table = em.choose_baseline(train, E.capacity)
    baseline = em.report(y_test, em.baseline_scores(test, column, train), test[TIME], E.capacity)
    family, family_table = choose_family(train, E.seed, E.capacity)

    results, scores = {}, {}
    for kind in em.MODEL_KINDS:                         # both families on the test, for the comparison
        model = em.fit(kind, train, E.seed)
        scores[kind] = model.predict_proba(test)[:, 1]
        results[kind] = em.report(y_test, scores[kind], test[TIME], E.capacity)
        if kind == family:
            final_model = model
    final = scores[family]

    leaky = TEMPTING.keys()
    train_c, test_c = (part.merge(cheat, on="order_id", how="left") for part in (train, test))
    for part in (train_c, test_c):
        part["final_status"] = part["final_status"].astype(str)
        part["flag_status_delivery_mismatch"] = part["flag_status_delivery_mismatch"].astype(float)
    cheat_model = em.fit(family, train_c, E.seed, allow_leaky=True,
                         extra_num=tuple(c for c in leaky if c != "final_status"), extra_cat=("final_status",))
    cheat_report = em.report(y_test, cheat_model.predict_proba(test_c)[:, 1], test[TIME], E.capacity)

    last_month = test[TIME].dt.to_period("M").max()
    in_last = (test[TIME].dt.to_period("M") == last_month).to_numpy()
    last = em.report(y_test[in_last], final[in_last], test[TIME][in_last], E.capacity)

    key_features = ["n_items", "total_freight", "promised_days", "seller_prior_negative_rate", "distance_km",
                    "holidays_in_window", "promised_working_days", "approval_hours", "customer_state", "main_category"]
    tables = {
        "baseline_rules": baseline_table, "family_choice": family_table,
        "capacity": em.capacity_table(y_test, final, test[TIME], E.capacities),
        "capacity_baseline": em.capacity_table(y_test, em.baseline_scores(test, column, train), test[TIME], E.capacities),
        "monthly": em.monthly(y_test, final, test[TIME], E.capacity),
        "drift": em.drift_table(train, test, key_features),
        "audit": audit_table(),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    joblib.dump({"pipeline": final_model, "family": family, "capacity": E.capacity, "cutoff": str(E.cutoff),
                 "sklearn": sklearn.__version__}, out / "model.joblib")
    metrics = {"read": read_stats, "split": split.counts, "baseline_rule": f"highest {column}",
               "baseline": baseline, "family": family, "models": results, "cheat": cheat_report,
               "leakage_gap_capture": cheat_report["capture"] - results[family]["capture"],
               "last_month": {"month": str(last_month), **last}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float) + "\n", encoding="utf-8")
    log.info("model: %s; capture at %.0f%%: model %.3f, baseline %.3f, cheat %.3f", family, 100 * E.capacity,
             results[family]["capture"], baseline["capture"], cheat_report["capture"])
    return {"family": family, "capture": results[family]["capture"]}


def stage_followup(config: Config) -> dict:
    """Score every order of the latest month (with or without a review) and write the daily lists."""
    E, out = config.ews, config.ews.output_dir
    saved = joblib.load(out / "model.joblib")
    with database(config.database) as engine, engine.connect() as connection:
        frame, _ = em.read_features(connection, E.load.schema, config.clean.chunk_rows)
    period = frame[(frame[TIME] >= E.cutoff) & (frame[TIME] < E.evaluation_end)]
    month = period[TIME].dt.to_period("M").max()
    latest = period[period[TIME].dt.to_period("M") == month].reset_index(drop=True)
    ranked = rank_orders(saved, latest, E.capacity)
    ranked.to_csv(out / "followup_list.csv", index=False)
    return {"month": str(month), "orders": len(latest), "to_follow": int(ranked["follow_up"].sum())}


def rank_orders(saved: dict, frame: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Risk score, daily quota and rank (1 = call first) for decoded feature rows."""
    score = saved["pipeline"].predict_proba(frame)[:, 1]
    result = pd.DataFrame({"order_id": frame["order_id"].astype(str), "decision_day": frame[TIME].dt.date,
                           "risk_score": score, "follow_up": em.follow_up(score, frame[TIME], capacity)})
    result["rank_in_day"] = result.groupby("decision_day")["risk_score"].rank(method="first", ascending=False).astype(int)
    return result.sort_values(["decision_day", "rank_in_day"], ignore_index=True)


STAGES = {"olist": stage_olist, "holidays": stage_holidays, "features": stage_features,
          "model": stage_model, "followup": stage_followup}


def run(config: Config, stage_names=STAGE_NAMES) -> dict:
    """Run the stages in order; timings.json is written after every stage."""
    out = config.ews.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ensure_database(config.database)
    timings = {"started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "stages": {}, "results": {}}
    total = time.perf_counter()
    for name in stage_names:
        started = time.perf_counter()
        log.info("early-warning stage %s: start", name)
        try:
            timings["results"][name] = STAGES[name](config)
        finally:
            timings["stages"][name] = round(time.perf_counter() - started, 1)
            timings["total_seconds"] = round(time.perf_counter() - total, 1)
            (out / "timings.json").write_text(json.dumps(timings, indent=1, default=str) + "\n", encoding="utf-8")
    return timings
