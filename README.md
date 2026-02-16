# Data Visualization Package (Notebook + Dash + TimescaleDB + Grafana)

## Start the stack
```bash
docker compose up -d --build
```

Services:
- TimescaleDB (PostgreSQL): localhost:5432 (postgres/postgres)
- Grafana: http://localhost:3000 (admin/admin)
- Dash: http://localhost:8050

## Populate data
Open the notebook:
`Scientific_DataViz_PhD_Research_Notebook.ipynb`
Run section **5.2** to insert monitoring data into `model_metrics`.

## Grafana dashboard
Provisioned automatically:
- Data source: TimescaleDB
- Dashboard: "Model Monitoring (TimescaleDB)"
