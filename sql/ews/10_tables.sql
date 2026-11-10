-- Stage 4: early-warning system. The holiday calendar (external API) and the logs of its loads.
-- Emptied and refilled on every holidays load, never dropped: the feature view
-- (sql/ews/20_features.sql) depends on holidays, and dropping it would drop that view too.
CREATE TABLE IF NOT EXISTS holidays (
    holiday_date date NOT NULL,
    name         text NOT NULL,
    holiday_type text NOT NULL,              -- 'national' in BrasilAPI
    PRIMARY KEY (holiday_date, name)
);

CREATE TABLE IF NOT EXISTS quarantine (
    source_table text    NOT NULL,
    source_row   integer NOT NULL,
    record       jsonb   NOT NULL,
    reason_codes text[]  NOT NULL CHECK (cardinality(reason_codes) > 0),
    PRIMARY KEY (source_table, source_row)
);

CREATE TABLE IF NOT EXISTS load_runs (
    load_id        text      PRIMARY KEY,
    loaded_at_utc  timestamp NOT NULL,
    snapshot       text      NOT NULL,
    config_sha256  text      NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    git_commit     text      NOT NULL,
    files          jsonb     NOT NULL
);

DELETE FROM holidays;
DELETE FROM quarantine;
