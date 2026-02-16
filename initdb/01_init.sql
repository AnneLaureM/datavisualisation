CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS model_metrics (
  time TIMESTAMPTZ NOT NULL,
  y_true DOUBLE PRECISION,
  y_pred DOUBLE PRECISION,
  y_lower DOUBLE PRECISION,
  y_upper DOUBLE PRECISION,
  anomaly_flag INTEGER,
  drift_score DOUBLE PRECISION
);

SELECT create_hypertable('model_metrics', 'time', if_not_exists => TRUE);
