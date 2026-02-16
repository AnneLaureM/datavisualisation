import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="AI Monitoring Dashboard (Dash + TimescaleDB + Grafana)",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Time Series", href="/time-series")),
            dbc.NavItem(dbc.NavLink("SHAP Explainability", href="/shap")),
        ],
    ),
    html.Hr(),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
