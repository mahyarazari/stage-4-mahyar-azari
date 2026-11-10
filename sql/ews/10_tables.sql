-- Stage 4: early-warning system. The holiday calendar (external API) and the logs of its loads.
-- Recreated on every holidays load; load_runs is a log and is never dropped. The feature
-- objects (sql/ews/20_features.sql) depend on these tables and are recreated after them.
DROP TABLE IF EXISTS holidays, quarantine CASCADE;

CREATE TABLE holidays (
    holiday_date date NOT NULL,
    name         text NOT NULL,
    holiday_type text NOT NULL,              -- 'national' in BrasilAPI
    PRIMARY KEY (holiday_date, name)
);

CREATE TABLE quarantine (
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
