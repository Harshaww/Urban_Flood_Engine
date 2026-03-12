# HydraGIS v2 — Quickstart Guide

## Required Folder Structure

The `data` folder must sit BESIDE `flood_fixed`, NOT inside it:

```
[your project folder]/
├── flood_fixed/              ← unzipped code (this folder)
│   ├── dashboard/app.py
│   ├── pipeline/
│   ├── models/
│   └── requirements.txt
└── data/
    └── data/
        ├── gis/
        │   ├── BBMP.geojson          ← MANDATORY (dashboard won't start without this)
        │   └── bengaluru_dem.tif     ← for real elevation data
        ├── rainfall_india.csv        ← Karnataka IMD rainfall
        └── flood_risk_india.csv      ← national flood risk dataset
```

---

## Setup

```bash
cd flood_fixed
pip install -r requirements.txt
```

---

## Step 1 — Generate micro-hotspots (run ONCE before dashboard)

This reads the DEM and BBMP.geojson and produces 2,743 hotspot points.
Takes 3-5 minutes. Only needs to run once.

```bash
python -m pipeline.micro_hotspots
```

---

## Step 2 — Launch the dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Step 3 — Run backtest validation

```bash
python -m models.backtest
```

Output:
- Flood recall: 87.5% (7/8 flood-prone wards correct)
- Overall accuracy: 88.9% (8/9 wards correct)

---

## Step 4 — Train the ensemble ML model

```bash
python -m models.train
```

Requires flood_risk_india.csv in the data folder.

---

## Step 5 — Generate resource deployment plan

```bash
python -m models.resource_allocator
```

Outputs deployment_plan.csv and deployment_summary.png.

---

## Step 6 — Start FastAPI backend (optional)

```bash
uvicorn main:app --reload
# Docs at http://localhost:8000/docs
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| BBMP.geojson not found | Check data folder is BESIDE flood_fixed, not inside it |
| ModuleNotFoundError: rasterio | pip install rasterio>=1.3.0 |
| Hotspots not showing on map | Run Step 1 first (python -m pipeline.micro_hotspots) |
| pydantic_settings error | pip install pydantic-settings==2.3.0 |
