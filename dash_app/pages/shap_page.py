import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import shap
import plotly.graph_objects as go

dash.register_page(__name__, path="/shap")

DATA_DIR = Path("data")
wine_path = DATA_DIR / "winequality-red.csv"

# ----------------------------
# 1) Load dataset
# ----------------------------
if wine_path.exists():
    df = pd.read_csv(wine_path, sep=";")
    y = df["quality"]
    X = df.drop(columns=["quality"])
    dataset_name = "UCI Wine Quality (red)"
else:
    from sklearn.datasets import load_diabetes
    diab = load_diabetes(as_frame=True)
    X = diab.data
    y = diab.target
    dataset_name = "sklearn Diabetes (fallback)"

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=400, random_state=42)
rf.fit(Xtr, ytr)

background = Xtr.sample(min(200, len(Xtr)), random_state=42)
explainer = shap.TreeExplainer(rf, data=background, feature_perturbation="interventional")

_SHAP_CACHE = {"X_sample": None, "shap_values": None}

def compute_shap_if_needed():
    if _SHAP_CACHE["X_sample"] is not None:
        return _SHAP_CACHE["X_sample"], _SHAP_CACHE["shap_values"]

    n = min(500, len(Xte))  # a bit more helps the summary view
    X_sample = Xte.sample(n, random_state=42)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    _SHAP_CACHE["X_sample"] = X_sample
    _SHAP_CACHE["shap_values"] = shap_values
    return X_sample, shap_values

# ----------------------------
# 2) Plotly builders
# ----------------------------
def shap_summary_plotly(X_sample: pd.DataFrame, shap_values: np.ndarray, max_features: int = 12):
    """
    Plotly 'beeswarm-like' summary:
    - For each feature: scatter of SHAP values (x) with jittered y.
    - Color by feature value (normalized).
    """
    # Mean absolute SHAP importance
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1][:max_features]
    features = X_sample.columns[order].tolist()

    fig = go.Figure()
    rng = np.random.default_rng(42)

    for rank, j in enumerate(order):
        f = X_sample.columns[j]
        x = shap_values[:, j]
        v = X_sample.iloc[:, j].to_numpy()

        # Normalize feature values for coloring
        if np.nanstd(v) > 0:
            v_norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        else:
            v_norm = np.zeros_like(v)

        y = np.full_like(x, rank, dtype=float) + rng.normal(0, 0.12, size=len(x))

        fig.add_trace(go.Scattergl(
            x=x, y=y,
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.7,
                color=v_norm,
                colorbar=dict(title="Feature<br>value"),
            ),
            name=f,
            showlegend=False,
            hovertemplate=(
                f"<b>{f}</b><br>"
                "SHAP=%{x:.3f}<br>"
                "y=%{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"SHAP Summary (top {len(features)} features) — {dataset_name}",
        xaxis_title="SHAP value (impact on model output)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(features))),
            ticktext=features,
            autorange="reversed",
            title=""
        ),
        height=650,
        margin=dict(l=160, r=40, t=60, b=40)
    )
    return fig

def shap_dependence_plotly(X_sample: pd.DataFrame, shap_values: np.ndarray, feature: str, color_feature: str | None = None):
    """
    Plotly dependence plot:
    x = feature value
    y = SHAP(feature)
    color = optional other feature (interaction proxy)
    """
    j = X_sample.columns.get_loc(feature)
    x = X_sample[feature].to_numpy()
    y = shap_values[:, j]

    marker = dict(size=7, opacity=0.75)

    if color_feature and color_feature in X_sample.columns and color_feature != feature:
        c = X_sample[color_feature].to_numpy()
        marker["color"] = c
        marker["colorbar"] = dict(title=color_feature)

    fig = go.Figure(go.Scattergl(
        x=x, y=y,
        mode="markers",
        marker=marker,
        hovertemplate=(
            f"<b>{feature}</b><br>"
            "x=%{x:.3f}<br>"
            "SHAP=%{y:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=f"SHAP Dependence — {feature}",
        xaxis_title=f"{feature} (feature value)",
        yaxis_title=f"SHAP({feature})",
        height=520,
        margin=dict(l=60, r=40, t=60, b=50)
    )
    return fig

# ----------------------------
# 3) Layout
# ----------------------------
layout = dbc.Container([
    html.H2("SHAP Explainability (Plotly-native)"),
    dbc.Alert(f"Dataset used: {dataset_name}", color="info"),

    dbc.Row([
        dbc.Col([
            html.P("Main feature:"),
            dcc.Dropdown(
                options=[{"label": c, "value": c} for c in X.columns],
                value=X.columns[0],
                id="feature"
            )
        ], md=6),

        dbc.Col([
            html.P("Color (interaction proxy):"),
            dcc.Dropdown(
                options=[{"label": "(none)", "value": ""}] + [{"label": c, "value": c} for c in X.columns],
                value="",
                id="color_feature"
            )
        ], md=6),
    ]),

    html.Hr(),

    html.H4("Global explanation (summary)"),
    dcc.Graph(id="summary_fig"),

    html.H4("Dependence plot"),
    dcc.Graph(id="dep_fig"),

    html.Pre(id="debug", style={"whiteSpace": "pre-wrap"})
], fluid=True)

# ----------------------------
# 4) Callback
# ----------------------------
@dash.callback(
    Output("summary_fig", "figure"),
    Output("dep_fig", "figure"),
    Output("debug", "children"),
    Input("feature", "value"),
    Input("color_feature", "value"),
)
def update(feature, color_feature):
    try:
        X_sample, shap_values = compute_shap_if_needed()
        summary = shap_summary_plotly(X_sample, shap_values, max_features=12)
        dep = shap_dependence_plotly(
            X_sample, shap_values,
            feature=feature,
            color_feature=(color_feature if color_feature else None)
        )
        return summary, dep, f"OK — n={len(X_sample)} | check_additivity=False"
    except Exception as e:
        empty = go.Figure()
        return empty, empty, f"SHAP Plotly error: {repr(e)}"
