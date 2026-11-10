"""Stage 4, model layer: rank approved orders by the risk of a negative review.

Discipline of stage 3, with time: the split is by decision moment, a training label counts only
if it was known before the cut-off, every learnt step is inside one Pipeline fitted on training
rows, and the decision is a capacity quota (the top share of each day's orders), not a
probability threshold (D80).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Connection

from data_factory.ews_audit import KEY, LABEL, LABEL_TIME, TIME, check_columns, model_features

FEATURE_QUERY = "SELECT * FROM order_features ORDER BY order_id"
MODEL_KINDS = ("logreg", "hgb")


# ---------------------------------------------------------------- reading and splitting
def read_features(connection: Connection, schema: str, chunk_rows: int, query: str = FEATURE_QUERY):
    """The feature table, in chunks; returns (frame, read statistics)."""
    connection.execute(text("SELECT set_config('search_path', :path, false)"), {"path": schema})
    started = time.perf_counter()
    chunks = list(pd.read_sql_query(text(query), connection, chunksize=chunk_rows))
    frame = pd.concat(chunks, ignore_index=True)
    stats = {"rows": len(frame), "chunks": len(chunks), "seconds": round(time.perf_counter() - started, 2)}
    return prepare_types(frame), stats


def prepare_types(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in (TIME, LABEL_TIME):
        if column in out:
            out[column] = pd.to_datetime(out[column])
    for column in model_features("num") + model_features("flag"):
        if column in out:
            out[column] = pd.to_numeric(out[column].astype("object").where(out[column].notna()), errors="coerce").astype("float64")
    for column in model_features("cat"):
        if column in out:
            out[column] = out[column].astype("object").where(out[column].notna(), "missing").astype(str)
    if LABEL in out:
        out[LABEL] = out[LABEL].astype("boolean")
    return out


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    counts: dict


def temporal_split(frame: pd.DataFrame, cutoff, end) -> Split:
    """Train: decided before the cut-off AND label known before it (what a team training on the
    cut-off day would have). Test: decided in [cutoff, end) with a review. Orders without a
    review have no label: kept out of both, still scored by the follow-up list (D79)."""
    cutoff, end = pd.Timestamp(cutoff), pd.Timestamp(end)
    labelled = frame[LABEL].notna()
    before = frame[TIME] < cutoff
    mature = frame[LABEL_TIME] < cutoff
    train = frame[before & labelled & mature]
    test = frame[(frame[TIME] >= cutoff) & (frame[TIME] < end) & labelled]
    counts = {"orders": len(frame), "no_review": int((~labelled).sum()),
              "before_cutoff_labelled": int((before & labelled).sum()),
              "dropped_label_not_known_at_cutoff": int((before & labelled & ~mature).sum()),
              "train": len(train), "test": len(test), "after_end": int((frame[TIME] >= end).sum()),
              "train_negative_rate": _rate(train), "test_negative_rate": _rate(test)}
    return Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True), counts=counts)


def _rate(part: pd.DataFrame) -> float:
    return float(part[LABEL].astype("float64").mean()) if len(part) else float("nan")


# ---------------------------------------------------------------- models
def build_model(kind: str, seed: int, extra_num: tuple[str, ...] = (), extra_cat: tuple[str, ...] = (),
                drop: tuple[str, ...] = ()) -> Pipeline:
    """Preparation + classifier as one Pipeline. `extra_*` exists only for the cheat model."""
    numeric = [c for c in model_features("num") + model_features("flag") if c not in drop] + list(extra_num)
    categorical = [c for c in model_features("cat") if c not in drop] + list(extra_cat)
    if kind == "logreg":
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True)),
                     ("log", FunctionTransformer(_signed_log1p)), ("scale", StandardScaler())]
        classifier = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    elif kind == "hgb":
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        classifier = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                                    min_samples_leaf=300, l2_regularization=1.0,
                                                    early_stopping=False, random_state=seed)
    else:
        raise ValueError(f"unknown model {kind!r}")
    prepare = ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("category", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                   sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("prepare", prepare), ("model", classifier)])


def _signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def model_inputs(pipeline: Pipeline) -> list[str]:
    columns = []
    for _, _, cols in pipeline.named_steps["prepare"].transformers:
        columns += list(cols)
    return columns


def recency_weights(train: pd.DataFrame, days: float) -> np.ndarray:
    """Weight exp(-age / days), age counted back from the newest training decision (D86)."""
    age = (train[TIME].max() - train[TIME]).dt.total_seconds() / 86400
    return np.exp(-age / days).to_numpy()


def fit(kind: str, train: pd.DataFrame, seed: int, allow_leaky: bool = False, weights=None, **options) -> Pipeline:
    model = build_model(kind, seed, **options)
    problems = check_columns(model_inputs(model))
    if problems and not allow_leaky:
        raise ValueError("refusing to train on columns that fail the temporal audit: " + "; ".join(problems))
    fit_options = {} if weights is None else {"model__sample_weight": weights}
    return model.fit(train, train[LABEL].astype(int), **fit_options)


class RankBlend:
    """Average of the two families' scores, each turned into its rank among the TRAINING scores
    (empirical CDF), so the blend scores one new order alone the same way as in a batch (D86)."""

    def __init__(self, models: dict[str, Pipeline], train: pd.DataFrame):
        self.models = models
        self.reference = {name: np.sort(m.predict_proba(train)[:, 1]) for name, m in models.items()}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = [np.searchsorted(self.reference[name], m.predict_proba(frame)[:, 1], side="right")
                 / len(self.reference[name]) for name, m in self.models.items()]
        score = np.mean(ranks, axis=0)
        return np.column_stack([1 - score, score])


# A candidate is a family (logreg, hgb or their blend) with or without recency weights.
CANDIDATES = tuple(f"{kind}{suffix}" for kind in (*MODEL_KINDS, "blend") for suffix in ("", "+recency"))


def fit_candidate(name: str, train: pd.DataFrame, seed: int, recency_days: float, **options):
    kind, _, weighting = name.partition("+")
    weights = recency_weights(train, recency_days) if weighting == "recency" else None
    if kind == "blend":
        return RankBlend({k: fit(k, train, seed, weights=weights, **options) for k in MODEL_KINDS}, train)
    return fit(kind, train, seed, weights=weights, **options)


# ---------------------------------------------------------------- capacity decision
def follow_up(scores, days, capacity: float) -> np.ndarray:
    """True for the top `capacity` share of each day's orders (at least one per day, ties by order).
    The team gets one list per day of approvals: an operational quota, not a probability threshold."""
    frame = pd.DataFrame({"score": np.asarray(scores, dtype=float), "day": pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize()})
    frame["rank"] = frame.groupby("day")["score"].rank(method="first", ascending=False)
    quota = np.ceil(frame.groupby("day")["score"].transform("size") * capacity)
    return (frame["rank"] <= quota).to_numpy()


def capture(y, flagged) -> float:
    """Share of all negative reviews that are on the follow-up lists (the main metric)."""
    y = np.asarray(y).astype(bool)
    return float((y & flagged).sum() / max(y.sum(), 1))


def report(y, scores, days, capacity: float) -> dict:
    y = np.asarray(y).astype(bool)
    flagged = follow_up(scores, days, capacity)
    tp, fp = int((y & flagged).sum()), int((~y & flagged).sum())
    fn, tn = int((y & ~flagged).sum()), int((~y & ~flagged).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"orders": len(y), "flagged": int(flagged.sum()), "capture": recall, "precision": precision,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "lift": precision / max(y.mean(), 1e-12),
            "roc_auc": roc_auc_score(y, scores), "pr_auc": average_precision_score(y, scores),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "base_rate": float(y.mean())}


def capacity_table(y, scores, days, capacities) -> pd.DataFrame:
    rows = []
    for share in capacities:
        r = report(y, scores, days, share)
        rows.append({"capacity": share, "orders_followed": r["flagged"], "negatives_caught": r["tp"],
                     "share_of_negatives_caught": r["capture"], "hit_rate_of_calls": r["precision"]})
    return pd.DataFrame(rows)


def monthly(y, scores, days, capacity: float) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y).astype(bool), "s": np.asarray(scores, dtype=float),
                          "d": pd.to_datetime(days)})
    rows = []
    for month, part in frame.groupby(frame["d"].dt.to_period("M")):
        r = report(part["y"], part["s"], part["d"], capacity)
        rows.append({"month": str(month), "orders": r["orders"], "negative_rate": r["base_rate"],
                     "capture": r["capture"], "precision": r["precision"], "roc_auc": r["roc_auc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- baseline and drift
BASELINE_CANDIDATES = ("promised_days", "seller_prior_negative_rate", "n_items", "n_sellers", "distance_km",
                       "total_freight", "seller_prior_late_rate", "holidays_in_window")


def choose_baseline(train: pd.DataFrame, capacity: float) -> tuple[str, pd.DataFrame]:
    """A rule the operations team understands: 'call the orders with the highest X'. X is the
    single column that catches most negatives on the TRAINING period."""
    rows = []
    for column in BASELINE_CANDIDATES:
        score = train[column].fillna(train[column].median())
        rows.append({"rule": f"highest {column}", "column": column,
                     "train_capture": capture(train[LABEL], follow_up(score, train[TIME], capacity))})
    table = pd.DataFrame(rows).sort_values("train_capture", ascending=False, ignore_index=True)
    return table.loc[0, "column"], table


def baseline_scores(frame: pd.DataFrame, column: str, train: pd.DataFrame) -> np.ndarray:
    return frame[column].fillna(train[column].median()).to_numpy(dtype=float)


def rolling_origin(frame: pd.DataFrame, origins, horizon_days: int, candidate: str, seed: int,
                   recency_days: float, capacity: float) -> pd.DataFrame:
    """The whole procedure again from several cut-offs (D87): for each origin, train on what was
    known then (same maturity rule), choose the baseline column on that training part, and score
    the next `horizon_days`. One split can be lucky; several show the spread."""
    rows = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        split = temporal_split(frame, origin, origin + pd.Timedelta(days=horizon_days))
        train, test = split.train, split.test
        column, _ = choose_baseline(train, capacity)
        model = fit_candidate(candidate, train, seed, recency_days)
        scores = model.predict_proba(test)[:, 1]
        r = report(test[LABEL], scores, test[TIME], capacity)
        b = report(test[LABEL], baseline_scores(test, column, train), test[TIME], capacity)
        rows.append({"origin": str(origin.date()), "train": len(train), "test": len(test),
                     "negative_rate": r["base_rate"], "model_capture": r["capture"], "baseline_rule": column,
                     "baseline_capture": b["capture"], "gain": r["capture"] - b["capture"], "roc_auc": r["roc_auc"]})
    return pd.DataFrame(rows)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index: bins from the training quantiles (numbers) or categories."""
    expected, actual = expected.dropna(), actual.dropna()
    if not pd.api.types.is_numeric_dtype(expected):
        e = expected.value_counts(normalize=True)
        a = actual.value_counts(normalize=True).reindex(e.index, fill_value=0)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def drift_table(train: pd.DataFrame, test: pd.DataFrame, columns) -> pd.DataFrame:
    def typical(values: pd.Series):
        return values.median() if pd.api.types.is_numeric_dtype(values) else values.mode().iat[0]
    rows = [{"feature": c, "psi": psi(train[c], test[c]), "train_typical": typical(train[c]),
             "test_typical": typical(test[c])} for c in columns]
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
