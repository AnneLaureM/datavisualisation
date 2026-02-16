import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

layout = dbc.Container([
    html.H2("Welcome"),
    html.P("This is a Dash app. Use the pages to:"),
    html.Ul([
        html.Li("Visualize monitoring time-series from TimescaleDB"),
        html.Li("Inspect SHAP explanations for a model"),
        html.Li("Bridge research diagnostics and production monitoring")
    ]),
    dbc.Alert("Start the stack with: docker compose up -d --build", color="info"),
    html.P("Grafana is available at http://localhost:3000 (admin/admin)."),
], fluid=True)
