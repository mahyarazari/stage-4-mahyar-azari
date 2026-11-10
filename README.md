# Early-warning system for negative reviews (final challenge)

At the moment an Olist order is **approved**, rank it by the risk that it ends in a negative
review (score 1 or 2), so the support team, which can follow up **10%** of orders, calls the
right customers. Built from the three earlier stages: the data factory (stage 1), the
PostgreSQL database (stage 2) and the modelling discipline (stage 3).

```
[Olist CSV] ──┐
              ├─> data factory (contract, quarantine, manifest) ─> PostgreSQL
[holidays API]┘                                                       │
                          features in SQL: view + procedure + snapshot (sql/ews/20_features.sql)
                                                                      │
                          model layer: time split, two families, capacity quota (ews_model.py)
                                                                      │
                          daily follow-up lists: top 10% of each day's approved orders
```

## Bring it up (from nothing)

You need **Python 3.12** and a **PostgreSQL server** (tested on 18.6, no Docker). About 1 GB of free disk.

```bash
git clone https://github.com/mahyarazari/stage-4-mahyar-azari.git
cd stage-4-mahyar-azari
python -m venv .venv
.venv\Scripts\activate                      # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

# Connection: nothing secret is stored in the repository.
set PGPASSWORD=<postgres password>          # macOS/Linux: export PGPASSWORD=...
set PGPORT=5432                             # the port of your server
# or one variable for everything: set DATABASE_URL=postgresql+psycopg://user@host:port/dbname

data-factory check-db                       # 1. connection test
data-factory early-warning                  # 2. THE ONE COMMAND: empty server -> follow-up lists
python -m unittest discover -s tests -t .   # 3. tests (all four stages)
```

`data-factory early-warning` creates the database if it does not exist and runs five timed stages:

| Stage | What it does | Time on an empty server (measured) |
|---|---|---|
| `olist` | downloads the 9 Olist files (pinned sha256; skipped when already on disk), validates, quarantines, loads | 128 s |
| `holidays` | BrasilAPI holiday calendar through the same factory | 0.4 s |
| `features` | builds the feature view/procedure in SQL, fills the snapshot, times the feature query 20 times | 14 s |
| `model` | time split, baseline, LogReg vs HistGradientBoosting, cheat model, drift | 72 s |
| `followup` | daily top-10% lists of the latest month | 5 s |

Total: **3.7 minutes** (budget: 45). Timings go to `data/curated/ews/timings.json`.

## Score new orders (the evaluator's entry point)

```bash
data-factory ews-predict --input <folder> --output ranked.csv
```

`<folder>` holds new orders in the original Olist file format (same file names and columns):

| File | Required |
|---|---|
| `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_customers_dataset.csv` | yes |
| `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv` | only for products/sellers the database does not know |

Review files are ignored (a review is the order's future). The output has one row per order:
`order_id, decision_day, risk_score, follow_up, rank_in_day`. A missing column stops with exit
code 32 (`SourceSchemaChangedError`), a missing required file with 40. The database is left
unchanged (everything runs inside a transaction that is rolled back). Run `data-factory early-warning` first: it saves the model.

## Deliverables

| File | Content |
|---|---|
| `01_exploration.ipynb` | EDA, temporal audit of the columns with evidence, the holiday hypothesis, failed paths |
| `02_system.ipynb` | the system run end to end: timings, features, models, capacity table, leakage cost, drift, entry point |
| `report.md` | the reasoning: key decisions, the temporal-leakage section, limits |
| `decisions.md` | architecture decisions D72–D85 (context, options, decision, accepted costs) |
| `manager_summary.md` | one page for the operations manager, no technical terms |
| `data/curated/ews/` | outputs: `followup_list.csv`, `metrics.json`, `capacity.csv`, `monthly.csv`, `drift.csv`, `feature_query_plan.json`, `timings.json`, `model.joblib` |

## Project layout

| Path | Content |
|---|---|
| `factory.toml` | every setting: sources, pinned checksums, schemas, cut-off, capacity |
| `src/data_factory/` | the package; stage 4: `holidays_pipeline.py`, `ews_audit.py`, `ews_model.py`, `ews_pipeline.py`, `ews_predict.py` |
| `sql/ews/` | holiday tables, feature view + procedure + snapshot, scoring inserts |
| `tests/` | 231 tests; stage 4: `test_ews.py` (16, incl. the leak guard and the whole chain) |
| `docs/` | stages 1–3: notebooks, reports, decisions D1–D71 |

Earlier commands still work: `data-factory run [--dataset nyc311|olist|credit|holidays]`,
`benchmark`, `index-evidence`, `predict` (stage 3).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 10 | configuration error |
| 20–21 | network error / invalid response |
| 30–34 | data error (validation, quarantine limit, source schema change, reference data, checksum mismatch) |
| 40 | input of an earlier stage is missing |
| 50 | database error (connection, constraint, load rolled back) |
| 1 | unexpected error (bug) |
