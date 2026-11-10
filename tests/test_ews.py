"""Stage 4: early-warning system. Units first; then database tests on a synthetic history
(temporary schemas), including the leak guard and the whole chain from files to a ranked list."""
import dataclasses
import json
import logging
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg import sql

from data_factory import ews_model as em
from data_factory import ews_pipeline, holidays_pipeline, olist_pipeline
from data_factory.cli import main
from data_factory.config import load_config
from data_factory.ews_audit import AUDIT, KEY, LABEL, LABEL_TIME, TEMPTING, TIME, check_columns, model_features
from data_factory.ews_predict import predict_folder
from data_factory.exceptions import ConfigError, ExitCode, InvalidResponseError, MissingInputError, SourceSchemaChangedError
from data_factory.olist_contract import TABLES
from data_factory.pipeline import RunContext
from data_factory.table_validation import validate_tables
from tests.db_helpers import PROJECT_CONFIG, TemporarySchemaMixin, needs_database
from tests.ews_fixtures import ews_rows, order_times, write_ews_files
from tests.olist_fixtures import hexid, write_files


def frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-01-01", periods=n, freq="6h")
    out = pd.DataFrame({KEY: range(n), TIME: days, LABEL: pd.array(rng.random(n) < 0.3, dtype="boolean"),
                        LABEL_TIME: days + pd.Timedelta(days=20)})
    return out


# ---------------------------------------------------------------- units
class HolidaySourceTests(unittest.TestCase):
    def test_responses_become_rows_and_broken_bodies_stop_the_run(self):
        rows = holidays_pipeline.responses_to_rows({2018: [{"date": "2018-01-01", "name": "Ano novo",
                                                            "type": "national", "extra": 1}]})
        self.assertEqual(rows.to_dict("records"), [{"date": "2018-01-01", "name": "Ano novo", "type": "national"}])
        with self.assertRaises(InvalidResponseError):
            holidays_pipeline.responses_to_rows({2018: {"message": "rate limited"}})

    def test_contract_quarantines_an_impossible_date(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        (folder / "holidays.csv").write_text("date,name,type\n2018-01-01,Ano novo,national\n2018-02-30,X,national\n",
                                             encoding="utf-8")
        stats = validate_tables(holidays_pipeline.TABLES, folder, folder / "w", folder / "q", 10, 1.0, {}, "holidays")
        self.assertEqual((stats["valid_records"], stats["quarantined_records"]), (1, 1))


class AuditTests(unittest.TestCase):
    def test_model_features_are_audited_and_allowed(self):
        self.assertGreaterEqual(len(model_features()), 10)
        self.assertEqual(check_columns(model_features()), [])
        for column in (LABEL, LABEL_TIME, *TEMPTING):
            self.assertNotIn(column, model_features())

    def test_tempting_and_unknown_columns_are_reported(self):
        problems = check_columns(["n_items", "days_late", "brand_new_column"])
        self.assertEqual(len(problems), 2)
        self.assertIn("after the decision", problems[0])
        self.assertIn("not in the temporal audit", problems[1])

    def test_training_refuses_a_leaky_column(self):
        with self.assertRaises(ValueError):
            em.fit("logreg", frame(), seed=1, extra_num=("days_late",))


class SplitAndCapacityTests(unittest.TestCase):
    def test_split_uses_time_and_label_maturity(self):
        data = frame(80)
        data.loc[5, LABEL] = pd.NA
        cutoff = data[TIME].iloc[60]
        split = em.temporal_split(data, cutoff, data[TIME].max() + pd.Timedelta(days=1))
        self.assertTrue((split.train[TIME] < cutoff).all() and (split.train[LABEL_TIME] < cutoff).all())
        self.assertTrue((split.test[TIME] >= cutoff).all())
        self.assertGreater(split.counts["dropped_label_not_known_at_cutoff"], 0)   # bought before, answered after
        self.assertNotIn(5, set(split.train[KEY]) | set(split.test[KEY]))           # no review, no label

    def test_daily_quota_takes_the_top_of_each_day(self):
        days = pd.to_datetime(["2018-01-01 09:00"] * 20 + ["2018-01-02 09:00"] * 5)
        scores = np.r_[np.arange(20), np.arange(5)]
        flagged = em.follow_up(scores, days, 0.10)
        self.assertEqual(flagged.sum(), 2 + 1)                    # ceil(2.0) and ceil(0.5)
        self.assertTrue(flagged[18] and flagged[19] and flagged[24])

    def test_report_counts_and_capacity_table(self):
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], bool)
        scores = np.array([.9, .8, .1, .2, .3, .1, .1, .1, .1, .7])
        days = pd.to_datetime(["2018-01-01"] * 10)
        r = em.report(y, scores, days, 0.2)                       # two orders followed: .9 and .8
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 2, 6))
        self.assertAlmostEqual(r["capture"], 1 / 3)
        table = em.capacity_table(y, scores, days, (0.1, 0.3, 0.5))
        self.assertTrue(table["share_of_negatives_caught"].is_monotonic_increasing)

    def test_baseline_picks_the_informative_column_and_psi_sees_a_shift(self):
        data = frame(200)
        for column in em.BASELINE_CANDIDATES:
            data[column] = np.random.default_rng(3).random(200)
        data["n_items"] = data[LABEL].astype(float) * 2 + 1
        column, _ = em.choose_baseline(data, 0.2)
        self.assertEqual(column, "n_items")
        same = pd.Series(np.random.default_rng(1).normal(size=2000))
        self.assertLess(em.psi(same, same.sample(frac=1, random_state=2)), 0.01)
        self.assertGreater(em.psi(same, same + 1), 0.2)


class BlendTests(unittest.TestCase):
    def test_blend_scores_one_order_alone_as_in_a_batch_and_recent_orders_weigh_more(self):
        data = frame(300, seed=4)
        rng = np.random.default_rng(5)
        for column in em.model_features("num") + em.model_features("flag"):
            data[column] = rng.random(300)
        for column in em.model_features("cat"):
            data[column] = "x"
        data["n_items"] = data[LABEL].astype(float) + rng.random(300)
        train, new = data.iloc[:220], data.iloc[220:]          # new orders: not among the training scores
        blend = em.fit_candidate("blend+recency", train, seed=1, recency_days=30)
        batch = blend.predict_proba(new)[:, 1]
        alone = np.array([blend.predict_proba(new.iloc[[i]])[0, 1] for i in (0, 40, 79)])
        np.testing.assert_allclose(alone, batch[[0, 40, 79]])
        self.assertTrue(((batch >= 0) & (batch <= 1)).all())
        rolling = em.rolling_origin(data, [data[TIME].iloc[150], data[TIME].iloc[200]], 15, "logreg", 1, 30, 0.1)
        self.assertEqual(len(rolling), 2)
        self.assertTrue((rolling["train"].diff().dropna() > 0).all())      # later origin, more known history
        weights = em.recency_weights(train, 30)
        self.assertEqual(weights[-1], 1.0)
        self.assertTrue(np.all(np.diff(weights) > 0))


class ConfigTests(unittest.TestCase):
    def test_cutoff_after_end_and_unknown_stage_are_rejected(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        text = PROJECT_CONFIG.read_text("utf-8").replace("evaluation_end = 2018-09-01T00:00:00",
                                                         "evaluation_end = 2018-01-01T00:00:00")
        (folder / "factory.toml").write_text(text, "utf-8")
        with self.assertRaises(ConfigError):
            load_config(folder / "factory.toml")
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.assertEqual(main(["--config", str(PROJECT_CONFIG), "early-warning", "--stages", "bake"]), ExitCode.CONFIG)

    def test_prediction_without_a_trained_model_says_what_to_run(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        config = load_config(PROJECT_CONFIG)
        config = dataclasses.replace(config, ews=dataclasses.replace(config.ews, output_dir=folder / "empty"))
        with self.assertRaises(MissingInputError) as caught:
            predict_folder(config, folder, folder / "out.csv")
        self.assertIn("data-factory early-warning", str(caught.exception))


# ---------------------------------------------------------------- database
@needs_database
class EwsDatabaseTests(TemporarySchemaMixin, unittest.TestCase):
    """Synthetic history in two temporary schemas (olist and ews), loaded by the factory."""

    def setUp(self):
        super().setUp()
        logging.disable(logging.CRITICAL)
        self.ews_schema = f"test_{uuid.uuid4().hex[:10]}"
        self.root = Path(tempfile.mkdtemp())
        config = load_config(PROJECT_CONFIG)
        paths = dataclasses.replace(config.paths, raw=self.root / "raw", work=self.root / "work",
                                    quarantine=self.root / "q", curated=self.root / "curated")
        ews = dataclasses.replace(config.ews, holidays_years=(2017, 2018), cutoff=pd.Timestamp("2017-09-01"),
                                  evaluation_end=pd.Timestamp("2018-01-01"), output_dir=self.root / "ews",
                                  rolling_origins=(pd.Timestamp("2017-08-01"), pd.Timestamp("2017-10-01")),
                                  load=dataclasses.replace(config.ews.load, schema=self.ews_schema))
        olist = dataclasses.replace(config.olist, load=dataclasses.replace(config.olist.load, schema=self.schema))
        self.config = dataclasses.replace(config, paths=paths, ews=ews, olist=olist)
        write_ews_files(olist_pipeline.raw_dir(self.config), holidays_pipeline.raw_dir(self.config))
        olist_pipeline.run(RunContext(config=self.config), ["validate", "load"])
        ews_pipeline.run(self.config, ["holidays", "features"])

    def tearDown(self):
        with self.engine.begin() as connection:
            with connection.connection.driver_connection.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.ews_schema)))
        shutil.rmtree(self.root, ignore_errors=True)
        logging.disable(logging.NOTSET)
        super().tearDown()

    def features(self, view="order_features"):
        with self.engine.connect() as connection:
            ews_pipeline.use_schemas(connection, ews_pipeline.search_path(self.config))
            return pd.read_sql_query(f"SELECT * FROM {view} ORDER BY order_id", connection)

    def execute(self, statement):
        with self.engine.begin() as connection:
            ews_pipeline.use_schemas(connection, ews_pipeline.search_path(self.config))
            connection.exec_driver_sql(statement)

    def test_every_feature_column_is_in_the_temporal_audit(self):
        columns = list(self.features().columns)
        self.assertEqual(sorted(columns), sorted(AUDIT))      # a new SQL column without an audit row fails here

    def test_changing_the_future_changes_no_feature(self):
        """THE LEAK GUARD: rewrite everything that happens after a moment T (later reviews,
        deliveries, statuses). Features of orders decided before T must not move by one bit.
        A feature that peeks at its own order's delivery, status or review turns this red."""
        before = self.features("order_features_live")
        moment = "2017-06-01"
        self.execute(f"""
            UPDATE order_reviews SET score = 6 - score WHERE answered_at >= '{moment}';
            UPDATE orders SET delivered_customer_at = delivered_customer_at + interval '30 days',
                              status = 'canceled' WHERE delivered_customer_at >= '{moment}';""")
        after = self.features("order_features_live")
        decided = before[TIME] < pd.Timestamp(moment)
        columns = [c for c in before.columns if c not in (LABEL, LABEL_TIME)]
        pd.testing.assert_frame_equal(before.loc[decided, columns], after.loc[decided, columns])
        self.assertFalse(before[LABEL].equals(after[LABEL]))    # the change really happened

    def test_seller_history_counts_only_answers_before_the_decision(self):
        data = self.features().set_index(KEY)
        i = 100
        decided = order_times(i)["approved"]
        expected = sum(1 for j in range(1, 361) if j % 2 == i % 2 and order_times(j)["answered"] < decided)
        row = data.loc[uuid.UUID(hexid(20000 + i))]
        self.assertEqual(row["seller_prior_reviews"], expected)
        self.assertEqual(row["seller_prior_orders"], sum(1 for j in range(1, i) if j % 2 == i % 2))

    def test_holiday_window_counts_working_days(self):
        row = self.features().set_index(KEY).loc[uuid.UUID(hexid(20000 + 245))]   # bought 2017-09-04, Independencia on 09-07
        self.assertEqual(row["holidays_in_window"], 1)
        days = pd.bdate_range("2017-09-05", "2017-09-19")
        self.assertEqual(row["promised_working_days"], len(days) - 1)

    def test_whole_chain_from_files_to_a_ranked_list(self):
        timings = ews_pipeline.run(self.config, ["model", "followup"])
        metrics = json.loads((self.root / "ews" / "metrics.json").read_text("utf-8"))
        self.assertEqual(set(timings["stages"]), {"model", "followup"})
        self.assertGreater(metrics["split"]["train"], 0)
        self.assertGreater(metrics["cheat"]["capture"], metrics["models"][metrics["family"]]["capture"] - 1e-9)
        followup = pd.read_csv(self.root / "ews" / "followup_list.csv")
        self.assertGreater(followup["follow_up"].sum(), 0)

        new = self.root / "new"                                   # three brand-new orders of a new customer
        rows = ews_rows(363)
        keep = {hexid(20000 + i) for i in (361, 362, 363)}
        for table in ("orders", "order_items", "order_payments"):
            rows[table] = [r for r in rows[table] if r["order_id"] in keep]
        rows["customers"] = [r for r in rows["customers"] if r["customer_id"] in {hexid(1000 + i) for i in (361, 362, 363)}]
        write_files(new, rows)
        before = self.query("SELECT count(*) AS n FROM orders")[0].n
        ranked = predict_folder(self.config, new, self.root / "ranked.csv")
        self.assertEqual(sorted(ranked["order_id"].str.replace("-", "")), sorted(keep))
        self.assertEqual(self.query("SELECT count(*) AS n FROM orders")[0].n, before)   # rolled back
        self.assertTrue(ranked["risk_score"].between(0, 1).all())

    def test_prediction_refuses_a_wrong_schema_with_a_clear_error(self):
        ews_pipeline.run(self.config, ["model"])
        bad = self.root / "bad"
        write_files(bad, ews_rows(5), drop_column=("orders", "order_estimated_delivery_date"))
        with self.assertRaises(SourceSchemaChangedError):
            predict_folder(self.config, bad, self.root / "out.csv")
        (bad / next(s.file for s in TABLES if s.table == "order_payments")).unlink()
        with self.assertRaises(MissingInputError):
            predict_folder(self.config, bad, self.root / "out.csv")


if __name__ == "__main__":
    unittest.main()
