import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import psycopg2
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/time-series")

PG = dict(host="timescaledb", port=5432, dbname="postgres", user="postgres", password="postgres")

def fetch():
    try:
        conn = psycopg2.connect(**PG)
        q = "SELECT time, y_true, y_pred, y_lower, y_upper, anomaly_flag, drift_score FROM model_metrics ORDER BY time DESC LIMIT 2000;"
        df = pd.read_sql(q, conn)
        conn.close()
        df = df.sort_values("time")
        return df
    except Exception:
        return pd.DataFrame({"time":[], "y_true":[], "y_pred":[], "y_lower":[], "y_upper":[], "anomaly_flag":[], "drift_score":[]})

layout = dbc.Container([
    html.H2("Time-Series Monitoring"),
    dbc.Button("Refresh", id="refresh", color="primary"),
    dcc.Graph(id="ts"),
    dcc.Graph(id="drift"),
    dcc.Interval(id="tick", interval=10_000, n_intervals=0),
    html.Div(id="status")
], fluid=True)

@dash.callback(
    Output("ts","figure"),
    Output("drift","figure"),
    Output("status","children"),
    Input("refresh","n_clicks"),
    Input("tick","n_intervals"),
)
def update(n, t):
    df = fetch()
    if df.empty:
        return px.line(title="No data yet (is the DB initialized?)"), px.line(title="No drift yet"), "No data. Run notebook section 5.2 to insert rows."

    fig = px.line(df, x="time", y=["y_true","y_pred"], title="y_true vs y_pred")
    fig.add_scatter(x=df["time"], y=df["y_lower"], mode="lines", name="lower", opacity=0.4)
    fig.add_scatter(x=df["time"], y=df["y_upper"], mode="lines", name="upper", opacity=0.4)

    anomalies = df[df["anomaly_flag"]==1]
    if len(anomalies):
        fig.add_scatter(x=anomalies["time"], y=anomalies["y_true"], mode="markers", name="anomaly",
                        marker={"size":10})

    fig2 = px.line(df, x="time", y="drift_score", title="Drift score")
    return fig, fig2, f"Rows loaded: {len(df)}"
