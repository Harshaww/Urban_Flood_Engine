import logging
import io
import json
import math
from contextlib import asynccontextmanager
from typing import Optional, Set

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from models.train import train_all, load_models
from models.predict import score_all_wards, generate_hotspots, score_single_ward, compute_deployment_plan, predict_flood_depth, generate_flood_spread, run_monsoon_simulator
from models.backtest import run_backtest, backtest_report_text
from api.schemas import (
    WardScoresResponse, HotspotsResponse, DeploymentResponse,
    TrainMetrics, TrainTaskResponse, WhatIfRequest, WhatIfResponse,
    UploadResponse, HealthResponse,
    FloodDepthRequest, FloodDepthResponse,
    FloodSpreadResponse,
    MonsoonSimRequest, MonsoonSimResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("hydragis.main")

# ── App state ─────────────────────────────────────────────────────────────────
APP_STATE: dict = {"models": None, "ward_scores": None}

# ── Redis client (optional — graceful fallback) ───────────────────────────────
_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        r.ping()
        _redis_client = r
        log.info("Redis connected ✓ (%s)", settings.REDIS_URL)
        return _redis_client
    except Exception as e:
        log.warning("Redis unavailable (%s) — cache disabled", e)
        return None

CACHE_KEY = "hydragis:ward_scores"

def _cache_get(key: str):
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None

def _cache_set(key: str, value, ttl: int = None):
    r = _get_redis()
    if r is None:
        return
    try:
        r.set(key, json.dumps(value), ex=ttl or settings.CACHE_TTL_SECONDS)
    except Exception as e:
        log.warning("Redis SET failed: %s", e)

def _cache_delete(key: str):
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(key)
        log.info("Cache invalidated: %s", key)
    except Exception:
        pass

# ── WebSocket connection manager ──────────────────────────────────────────────
class AlertConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info("WebSocket client connected — total=%d", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        log.info("WebSocket client disconnected — total=%d", len(self.active))

    async def broadcast(self, message: dict):
        dead = set()
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        self.active -= dead

manager = AlertConnectionManager()


# ══════════════════════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═" * 60)
    log.info("  HydraGIS Flood Intelligence Engine — starting up")
    log.info("═" * 60)
    try:
        APP_STATE["models"] = load_models()
        cached = _cache_get(CACHE_KEY)
        if cached:
            APP_STATE["ward_scores"] = cached
            log.info("Startup: ward_scores loaded from Redis cache (%d wards) ✓", len(cached))
        else:
            APP_STATE["ward_scores"] = score_all_wards(APP_STATE["models"])
            _cache_set(CACHE_KEY, APP_STATE["ward_scores"])
            log.info("Startup complete — %d wards scored and cached ✓", len(APP_STATE["ward_scores"]))
    except Exception as e:
        log.error("Startup error: %s. API will serve partial responses.", e)
    yield
    log.info("Shutting down HydraGIS engine")


# ══════════════════════════════════════════════════════════════════════════════
#  APP INIT
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title     = settings.APP_NAME,
    version   = settings.VERSION,
    description = """
## HydraGIS — Urban Flood Prediction API

Predicts **ward-level flood risk** for all **243 BBMP wards** in Bengaluru using an
ensemble of Random Forest + XGBoost models, generating 2,500+ GIS-derived micro-hotspot
zones and pre-monsoon deployment plans.

### Model Architecture
| Model | Training Data | Target |
|---|---|---|
| Random Forest | flood_risk_india.csv (10,000-row national dataset, real binary labels) | Flood Occurred (binary) |
| XGBoost | 243 BBMP ward features from GIS + SRTM DEM + IMD rainfall | NDMA-weighted composite risk index |
| Ensemble | Weighted blend (RF 45% + XGBoost 55%) | Pre-Monsoon Readiness Score (0–100) |

### Readiness Score Methodology (NDMA-weighted)
| Factor | Weight | Source |
|---|---|---|
| Drainage coverage | 30% | BBMP SWD audit methodology |
| Terrain elevation | 25% | NDMA Urban Flood Guidelines 2010 |
| Historical rainfall | 20% | IMD Karnataka 15-year data |
| Infrastructure age | 15% | CPHEEO manual degradation rates |
| Pump capacity | 10% | BBMP pump station registry |

### Key Endpoints
| Endpoint | Description |
|---|---|
| `GET /wards/scores` | Pre-Monsoon Readiness Score for all wards |
| `GET /hotspots` | 2,500+ GIS-derived flood micro-hotspot zones |
| `GET /deployment/plan` | Resource pre-positioning plan |
| `GET /validation/backtest` | Retrospective validation vs BBMP flood records 2017–2022 |
| `POST /predict/flood` | Physics-based flood depth prediction for a ward |
| `POST /simulate/monsoon` | City-wide monsoon event simulation |
| `POST /model/train` | Trigger model retraining |
| `WS /ws/alerts` | Live rainfall alert stream |
    """,
    lifespan  = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return {
        "status":        "ok",
        "models_loaded": APP_STATE["models"] is not None,
        "version":       settings.VERSION,
        "ward_count":    len(APP_STATE.get("ward_scores") or []),
    }


@app.get("/wards/scores", response_model=WardScoresResponse, tags=["Prediction"])
def get_ward_scores(
    risk_filter:      Optional[str] = Query(None, description="Filter by risk: CRITICAL, HIGH, MODERATE, LOW"),
    sort_by:          str           = Query("readiness_score", description="Field to sort by"),
    ascending:        bool          = Query(True),
    include_polygons: bool          = Query(False, description="Attach ward boundary polygon [[lat,lng],...] to each ward"),
):
    """
    Returns Pre-Monsoon Readiness Score (0-100) for all wards.
    Results are Redis-cached for 1 hour; cache is invalidated on retrain/upload.
    Pass `?include_polygons=true` to include ward boundary shapes for map rendering.
    """
    _require_models()

    # Use cache only for plain (no polygons, no filter) requests
    if not include_polygons and not risk_filter:
        cached = _cache_get(CACHE_KEY)
        if cached:
            wards = cached
        else:
            wards = score_all_wards(APP_STATE["models"])
            _cache_set(CACHE_KEY, wards)
            APP_STATE["ward_scores"] = wards
    else:
        wards = score_all_wards(APP_STATE["models"], include_polygons=include_polygons)

    if risk_filter:
        wards = [w for w in wards if w["risk_level"] == risk_filter.upper()]

    try:
        wards.sort(key=lambda x: x.get(sort_by, 0), reverse=not ascending)
    except Exception:
        pass

    critical  = sum(1 for w in wards if w["risk_level"] == "CRITICAL")
    high      = sum(1 for w in wards if w["risk_level"] == "HIGH")
    avg_score = round(sum(w["readiness_score"] for w in wards) / max(len(wards), 1), 1)

    return {
        "total_wards":          len(wards),
        "critical_count":       critical,
        "high_count":           high,
        "city_readiness_index": avg_score,
        "wards":                wards,
    }


@app.get("/wards/{ward_id}", tags=["Prediction"])
def get_single_ward(ward_id: str):
    _require_models()
    ward = next((w for w in APP_STATE["ward_scores"] if w["ward_id"] == ward_id.upper()), None)
    if not ward:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")
    return ward


@app.get("/hotspots", response_model=HotspotsResponse, tags=["Prediction"])
def get_hotspots(
    severity: Optional[str] = Query(None, description="Filter by: CRITICAL, HIGH, MODERATE"),
    ward_id:  Optional[str] = Query(None, description="Filter by ward ID"),
    limit:    int           = Query(500,  le=2500, description="Max hotspots to return"),
):
    _require_models()
    all_hotspots = generate_hotspots(APP_STATE["ward_scores"], n_total=2500)
    if severity:
        all_hotspots = [h for h in all_hotspots if h["severity"] == severity.upper()]
    if ward_id:
        all_hotspots = [h for h in all_hotspots if h["ward_id"] == ward_id.upper()]
    all_hotspots = all_hotspots[:limit]
    critical = sum(1 for h in all_hotspots if h["severity"] == "CRITICAL")
    high     = sum(1 for h in all_hotspots if h["severity"] == "HIGH")
    return {"total_hotspots": len(all_hotspots), "critical_count": critical,
            "high_count": high, "hotspots": all_hotspots}


@app.get("/deployment/plan", response_model=DeploymentResponse, tags=["Planning"])
def get_deployment_plan():
    _require_models()
    return compute_deployment_plan(APP_STATE["ward_scores"])


@app.post("/whatif", response_model=WhatIfResponse, tags=["Prediction"])
def what_if_scenario(req: WhatIfRequest):
    _require_models()
    baseline = next(
        (w["readiness_score"] for w in APP_STATE["ward_scores"] if w["ward_id"] == req.ward_id.upper()),
        None
    )
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"Ward {req.ward_id} not found")
    result = score_single_ward(req.ward_id.upper(), req.overrides, APP_STATE["models"])
    result["improvement"] = round(result["readiness_score"] - baseline, 1)
    return result


# ── Model Training (async via Celery) ─────────────────────────────────────────
@app.post("/model/train", response_model=TrainTaskResponse, status_code=202, tags=["Model"])
def retrain_models():
    """
    Queues RF + XGBoost retraining as a background Celery task.
    Returns immediately with a task_id.
    Poll `GET /model/train/status/{task_id}` for progress.

    Falls back to synchronous training if Celery/Redis is unavailable.
    Start the Celery worker with:
        celery -A tasks worker --loglevel=info
    """
    log.info("Retraining triggered via API")
    try:
        from tasks import train_models_task
        task = train_models_task.delay()
        log.info("Training task queued — task_id=%s", task.id)
        return {"task_id": task.id, "status": "PENDING",
                "message": f"Training queued. Poll GET /model/train/status/{task.id}"}
    except Exception as celery_err:
        log.warning("Celery unavailable (%s) — falling back to sync training", celery_err)
        metrics = train_all()
        APP_STATE["models"] = load_models()
        APP_STATE["ward_scores"] = score_all_wards(APP_STATE["models"])
        _cache_delete(CACHE_KEY)
        _cache_set(CACHE_KEY, APP_STATE["ward_scores"])
        return {
            "task_id": "sync",
            "status":  "SUCCESS",
            "message": f"Sync training complete — RF r²={metrics['random_forest']['r2']:.3f}",
        }


@app.get("/model/train/status/{task_id}", tags=["Model"])
def training_status(task_id: str):
    """
    Poll training task status.
    States: PENDING → PROGRESS → SUCCESS / FAILURE
    """
    if task_id == "sync":
        return {"task_id": "sync", "status": "SUCCESS",
                "result": "Synchronous training already completed."}
    try:
        from celery.result import AsyncResult
        from tasks import celery_app
        result = AsyncResult(task_id, app=celery_app)
        response = {"task_id": task_id, "status": result.state}
        if result.state == "PROGRESS":
            response["meta"] = result.info
        elif result.state == "SUCCESS":
            response["result"] = result.result
            try:
                APP_STATE["models"] = load_models()
                APP_STATE["ward_scores"] = score_all_wards(APP_STATE["models"])
                _cache_delete(CACHE_KEY)
                _cache_set(CACHE_KEY, APP_STATE["ward_scores"])
            except Exception:
                pass
        elif result.state == "FAILURE":
            response["error"] = str(result.info)
        return response
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"Celery unavailable: {e}")


@app.post("/upload/dataset", response_model=UploadResponse, tags=["Data"])
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Query(..., description="flood_train | rainfall | risk_india"),
):
    allowed = {"flood_train", "rainfall", "risk_india"}
    if dataset_type not in allowed:
        raise HTTPException(status_code=400, detail=f"dataset_type must be one of {allowed}")
    name_map = {
        "flood_train": settings.KAGGLE_FLOOD_TRAIN,
        "rainfall":    settings.KAGGLE_RAINFALL,
        "risk_india":  settings.KAGGLE_RISK_INDIA,
    }
    save_path = settings.DATA_DIR / name_map[dataset_type]
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot parse CSV: {e}")
    save_path.write_bytes(content)
    log.info(f"Uploaded {file.filename} → {save_path} ({len(df)} rows)")
    metrics = train_all()
    APP_STATE["models"] = load_models()
    APP_STATE["ward_scores"] = score_all_wards(APP_STATE["models"])
    _cache_delete(CACHE_KEY)
    _cache_set(CACHE_KEY, APP_STATE["ward_scores"])
    return {
        "filename":    file.filename,
        "rows_loaded": len(df),
        "columns":     list(df.columns),
        "preview":     df.head(3).to_dict(orient="records"),
        "status":      f"Saved and retrained. RF r²={metrics['random_forest']['r2']}, XGB r²={metrics['xgboost']['r2']}",
    }


@app.get("/city/summary", tags=["Prediction"])
def city_summary():
    _require_models()
    wards = APP_STATE["ward_scores"]
    risk_dist = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for w in wards:
        risk_dist[w["risk_level"]] += 1
    avg_score  = round(sum(w["readiness_score"] for w in wards) / len(wards), 1)
    total_pop  = sum(w["population"] for w in wards)
    at_risk    = sum(w["population"] for w in wards if w["risk_level"] in ["CRITICAL", "HIGH"])
    total_hs   = sum(w["hotspot_count"] for w in wards)
    return {
        "city":                  "Bengaluru Urban",
        "wards_analyzed":        len(wards),
        "city_readiness_index":  avg_score,
        "risk_distribution":     risk_dist,
        "total_population":      total_pop,
        "population_at_risk":    at_risk,
        "pct_population_at_risk": round(at_risk / total_pop * 100, 1),
        "estimated_hotspots":    total_hs,
        "worst_ward":            min(wards, key=lambda x: x["readiness_score"])["name"],
        "best_ward":             max(wards, key=lambda x: x["readiness_score"])["name"],
    }




# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — Real-Time Flood Depth Prediction
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/flood", response_model=FloodDepthResponse, tags=["Flood Intelligence"])
def predict_flood(req: FloodDepthRequest):
    """
    Predicts flood depth, duration and impact for a single ward given
    a rainfall event and optional drainage failure.

    Uses physics-based hydrology (rational method):
    runoff volume = rainfall x runoff_coefficient x ward_area
    flood depth = excess_volume / flood_spread_area

    Example: {"ward_id": "W02", "rainfall_mm": 280, "drainage_failure_pct": 20}
    Returns flood depth, duration, affected population, road closures.
    """
    _require_models()
    try:
        result = predict_flood_depth(
            req.ward_id.upper(),
            req.rainfall_mm,
            req.drainage_failure_pct,
            APP_STATE["ward_scores"],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — Flood Spread Visualisation Map
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/flood/spread", response_model=FloodSpreadResponse, tags=["Flood Intelligence"])
def flood_spread_map(
    rainfall_mm:          float = Query(...,  ge=0,   le=1000, description="Rainfall in mm"),
    drainage_failure_pct: float = Query(0.0,  ge=0,   le=100,  description="% drainage failure"),
    grid_points:          int   = Query(800,  ge=100, le=2000, description="Number of map grid points"),
):
    """
    Returns a flood propagation map - a grid of lat/lng points each with
    predicted flood depth. Plug directly into Leaflet heatmap or Mapbox GL.
    Each point: {lat, lng, depth_m, ward_id, severity}
    """
    _require_models()
    result = generate_flood_spread(
        APP_STATE["ward_scores"], rainfall_mm, drainage_failure_pct, grid_points
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — Monsoon Night Simulator
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/simulate/monsoon", response_model=MonsoonSimResponse, tags=["Flood Intelligence"])
def monsoon_simulator(req: MonsoonSimRequest):
    """
    The killer feature. Full city-wide monsoon simulation.

    Input: rainfall_mm + drainage_failure_pct
    Output:
    - Which wards flood and how deep
    - Total population affected
    - Road closures across the city
    - Pumps, NDRF teams, boats, barriers needed
    - Estimated economic damage in Rs crores
    - Hour-by-hour event timeline (T+0h to T+6h)
    - Top wards to evacuate (priority ranked)

    Try: {"rainfall_mm": 310, "drainage_failure_pct": 20}
    """
    _require_models()
    result = run_monsoon_simulator(
        req.rainfall_mm,
        req.drainage_failure_pct,
        APP_STATE["ward_scores"],
    )
    return result

# ── WebSocket — Live Rainfall Alerts ─────────────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    Live rainfall alert WebSocket.

    Send:   {"rainfall_mm": 250}
    Receive broadcast when thresholds crossed:
      >= 200 mm → WATCH alert
      >= 300 mm → CRITICAL_FLOOD alert
    """
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
                rainfall_mm = float(payload.get("rainfall_mm", 0))
            except Exception:
                await ws.send_json({"error": "Send JSON: {\"rainfall_mm\": <float>}"})
                continue

            alert_level = None
            if rainfall_mm >= settings.FLOOD_THRESHOLD_MM:
                alert_level = "CRITICAL_FLOOD"
            elif rainfall_mm >= settings.ALERT_THRESHOLD_MM:
                alert_level = "WATCH"

            wards = APP_STATE.get("ward_scores") or []
            if alert_level and wards:
                affected = [
                    {"ward_id": w["ward_id"], "name": w["name"],
                     "risk_level": w["risk_level"], "readiness_score": w["readiness_score"]}
                    for w in wards if w["risk_level"] in ("CRITICAL", "HIGH")
                ]
                alert_payload = {
                    "alert_level":    alert_level,
                    "rainfall_mm":    rainfall_mm,
                    "threshold_mm":   settings.FLOOD_THRESHOLD_MM if alert_level == "CRITICAL_FLOOD"
                                      else settings.ALERT_THRESHOLD_MM,
                    "affected_wards": affected[:20],
                    "affected_count": len(affected),
                    "message": (
                        f"CRITICAL: {rainfall_mm}mm exceeds flood threshold! {len(affected)} wards at risk."
                        if alert_level == "CRITICAL_FLOOD"
                        else f"WATCH: {rainfall_mm}mm approaching flood levels. Monitor {len(affected)} high-risk wards."
                    ),
                }
                await manager.broadcast(alert_payload)
                log.info("Alert broadcast: %s — %.0fmm — %d wards affected",
                         alert_level, rainfall_mm, len(affected))
            else:
                await ws.send_json({
                    "alert_level": "NORMAL",
                    "rainfall_mm": rainfall_mm,
                    "message":     f"Rainfall {rainfall_mm}mm — below alert thresholds.",
                })
    except WebSocketDisconnect:
        manager.disconnect(ws)


def _require_models():
    if APP_STATE["models"] is None:
        raise HTTPException(status_code=503,
                            detail="Models not loaded. Call POST /model/train first.")


# ══════════════════════════════════════════════════════════════════════════════
#  RETROSPECTIVE VALIDATION — BBMP FLOOD INCIDENT RECORDS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/validation/backtest", tags=["Validation"])
def get_backtest_results(text_report: bool = Query(False, description="Return human-readable report text")):
    """
    Runs retrospective validation of model predictions against historically
    flood-prone wards from BBMP Flood Incident Reports 2017–2022.

    Measures:
      - Flood recall: % of known flood-prone wards correctly classified as CRITICAL/HIGH
      - Safe precision: % of known safe wards correctly classified as LOW/MODERATE
      - Overall accuracy across both categories

    This is the 'backtesting' validation referenced in the written report.
    It demonstrates model reliability against real-world historical outcomes.
    """
    _require_models()

    ward_scores = APP_STATE.get("ward_scores")
    if not ward_scores:
        raise HTTPException(status_code=503, detail="Ward scores not available")

    summary = run_backtest(ward_scores)

    if text_report:
        return {"report": backtest_report_text(summary), **summary}

    return summary


@app.get("/")
def root():
    return {"message": "HydraGIS Flood Intelligence API"}


# ══════════════════════════════════════════════════════════════════════════════
#  ADD 2 — Live OpenWeatherMap Rainfall Integration
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/forecast", tags=["Live Rainfall"])
def get_live_forecast():
    """
    ADD 2: Fetch live 72-hour rainfall forecast for Bengaluru from OpenWeatherMap.

    If OWM_API_KEY is set in config/.env, returns live OWM forecast data.
    Falls back to IMD Karnataka historical average when API key is absent.

    Live data dynamically adjusts ward risk scores — wards with poor drainage
    receive larger upward adjustments during high-rainfall forecasts.

    Set key: export OWM_API_KEY=your_key  (free tier at openweathermap.org/api)
    """
    from pipeline.rainfall_forecast import get_forecast, adjust_risk_for_forecast
    import datetime

    api_key = settings.OWM_API_KEY or None
    rain_path = str(settings.DATA_DIR / settings.KAGGLE_RAINFALL)

    forecast = get_forecast(api_key=api_key, rain_path=rain_path)

    # Compute adjusted risk scores if ward_scores available
    adjusted_count = 0
    critical_delta = 0
    if APP_STATE.get("ward_scores"):
        ws = APP_STATE["ward_scores"]
        total_mm = forecast["total_72h_mm"]
        rain_factor = float(np.clip((total_mm - 30) / 270, 0, 1))
        upgraded = []
        for w in ws:
            drain_sensitivity = float(w.get("drain_deficit", 0.5))
            delta = rain_factor * 15 * (0.5 + 0.5 * drain_sensitivity)
            new_score = round(min(100.0, w["readiness_score"] + delta), 1)
            if new_score > w["readiness_score"]:
                adjusted_count += 1
            if new_score >= 70 and w["readiness_score"] < 70:
                critical_delta += 1
            upgraded.append({**w, "forecast_adjusted_score": new_score})
        top_adjusted = sorted(
            upgraded, key=lambda x: x["forecast_adjusted_score"] - x["readiness_score"], reverse=True
        )[:10]
    else:
        top_adjusted = []

    return {
        "forecast": forecast,
        "live_data": api_key is not None,
        "fetched_at": forecast.get("fetched_at"),
        "total_72h_mm": forecast["total_72h_mm"],
        "max_3h_mm": forecast["max_3h_mm"],
        "source": forecast["source"],
        "risk_adjustment": {
            "wards_adjusted": adjusted_count,
            "new_critical_wards": critical_delta,
            "note": (
                "Risk scores adjusted upward for wards with poor drainage during "
                "high-rainfall forecast. Worst-drainage wards get up to +15pt adjustment."
            ),
        },
        "top_impacted_wards": [
            {
                "ward_id": w["ward_id"],
                "name": w["name"],
                "baseline_score": w["readiness_score"],
                "forecast_score": w.get("forecast_adjusted_score"),
                "delta": round(w.get("forecast_adjusted_score", 0) - w["readiness_score"], 1),
                "risk_level": w["risk_level"],
            }
            for w in top_adjusted
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ADD 4 — Temporal Risk Trend: 2020 vs 2024 Rainfall Comparison
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/wards/trend", tags=["Temporal Analysis"])
def get_ward_risk_trend():
    """
    ADD 4: Computes per-ward risk delta between 2020 (IMD normal) and 2024
    (observed above-normal season) rainfall scenarios.

    Uses IMD Bengaluru Sub-division seasonal normals (1991-2020 baseline) vs
    2024 actual observed data (ENSO-active year, ~21% above normal city-wide).

    Source: IMD Pune Bengaluru Sub-division normals; IMD 2024 monsoon report.

    Returns: per-ward risk_delta, with notable examples like:
      "Bellandur risk increased 23% from 2020 to 2024 season"
    """
    _require_models()
    ward_scores = APP_STATE.get("ward_scores")
    if not ward_scores:
        raise HTTPException(status_code=503, detail="Ward scores not available")

    z2020 = settings.RAINFALL_2020_ZONE
    z2024 = settings.RAINFALL_2024_ZONE

    def _zone_for_ward(lat, lng):
        """Assign a ward to N/S/E/W zone based on coordinates."""
        if lat is None or lng is None:
            return "city"
        dlat = lat - 12.9716
        dlng = lng - 77.5946
        if abs(dlat) < 0.05 and abs(dlng) < 0.05:
            return "city"
        if abs(dlng) > abs(dlat):
            return "east" if dlng > 0 else "west"
        return "north" if dlat > 0 else "south"

    trend_results = []
    for ward in ward_scores:
        zone = _zone_for_ward(ward.get("lat"), ward.get("lng"))
        rain_2020 = z2020.get(zone, z2020["city"])
        rain_2024 = z2024.get(zone, z2024["city"])

        # Risk contribution from rainfall: W_RAINFALL = 0.20 in NDMA formula
        # Delta in rainfall → proportional delta in risk score
        rain_ratio = (rain_2024 - rain_2020) / rain_2020
        risk_delta = round(ward["readiness_score"] * (-1) * settings.W_RAINFALL * rain_ratio, 2)
        # Negative readiness_score delta = worse (readiness degrades with more rain)
        score_2020 = round(min(100, max(0, ward["readiness_score"] - risk_delta)), 1)
        score_2024 = ward["readiness_score"]
        pct_change = round((score_2024 - score_2020) / max(score_2020, 0.1) * 100, 1)

        trend_results.append({
            "ward_id":      ward["ward_id"],
            "name":         ward["name"],
            "zone":         zone,
            "rainfall_2020_mm": rain_2020,
            "rainfall_2024_mm": rain_2024,
            "rainfall_increase_pct": round(rain_ratio * 100, 1),
            "score_2020":   score_2020,
            "score_2024":   score_2024,
            "risk_delta":   round(score_2024 - score_2020, 1),
            "pct_change":   pct_change,
            "risk_level_2020": _risk_label(score_2020),
            "risk_level_2024": ward["risk_level"],
            "worsened": score_2024 < score_2020,
        })

    # Sort by worsening magnitude
    trend_results.sort(key=lambda x: x["risk_delta"])

    city_avg_2020 = round(sum(t["score_2020"] for t in trend_results) / len(trend_results), 1)
    city_avg_2024 = round(sum(t["score_2024"] for t in trend_results) / len(trend_results), 1)
    worsened_count = sum(1 for t in trend_results if t["worsened"])

    return {
        "analysis_period": "2020 (IMD Normal) vs 2024 (Observed)",
        "source": "IMD Bengaluru Sub-division seasonal normals + IMD 2024 monsoon report",
        "city_summary": {
            "avg_readiness_2020": city_avg_2020,
            "avg_readiness_2024": city_avg_2024,
            "city_delta": round(city_avg_2024 - city_avg_2020, 1),
            "wards_worsened": worsened_count,
            "rainfall_increase_city_pct": round(
                (z2024["city"] - z2020["city"]) / z2020["city"] * 100, 1
            ),
        },
        "most_worsened": trend_results[:10],
        "most_improved": sorted(trend_results, key=lambda x: -x["risk_delta"])[:5],
        "all_wards": trend_results,
    }


def _risk_label(score: float) -> str:
    if score < 30:   return "CRITICAL"
    if score < 50:   return "HIGH"
    if score < 65:   return "MODERATE"
    return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
#  ADD 6 — Exportable Pre-Monsoon Action Plan PDF
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/deployment/plan/pdf", tags=["Planning"])
def get_deployment_plan_pdf():
    """
    ADD 6: Downloads the pre-monsoon deployment plan as a formatted PDF.

    Generates a professional BBMP-style action plan document including:
    - Executive summary with city totals and cost estimate
    - Per-ward resource allocation table (pump units, NDRF teams, barriers)
    - Risk-tiered ward listing with readiness scores
    - Priority action deadlines (T-30 to T-0 days before monsoon)

    Built with reportlab. Suitable for submission to BBMP / KSNDMC.
    """
    _require_models()

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import datetime

    plan   = compute_deployment_plan(APP_STATE["ward_scores"])
    wards  = plan["ward_plans"]
    totals = plan["city_totals"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="HydraGIS Pre-Monsoon Action Plan",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                  fontSize=16, spaceAfter=6, alignment=TA_CENTER)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=3,
                         textColor=colors.HexColor("#1a3c6b"))
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=8, leading=11)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7, leading=10)

    RISK_COLOR = {
        "CRITICAL": colors.HexColor("#c0392b"),
        "HIGH":     colors.HexColor("#e67e22"),
        "MODERATE": colors.HexColor("#f1c40f"),
        "LOW":      colors.HexColor("#27ae60"),
    }

    story = []
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M IST")

    # ── Cover / Header ────────────────────────────────────────────────────────
    story.append(Paragraph("HYDRAGIS — PRE-MONSOON FLOOD PREPAREDNESS PLAN", title_style))
    story.append(Paragraph("Bruhat Bengaluru Mahanagara Palike | Urban Flood Risk Division", styles["Normal"]))
    story.append(Paragraph(f"Generated: {now}  |  Powered by HydraGIS v1.0", small))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3c6b")))
    story.append(Spacer(1, 0.3*cm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("1. EXECUTIVE SUMMARY", h1))
    story.append(Paragraph(
        f"This plan covers all <b>{plan['total_wards_assessed']} BBMP wards</b> of Bengaluru Urban. "
        f"Pre-monsoon risk assessment identifies <b>{plan['critical_wards']} CRITICAL</b> and "
        f"<b>{plan['high_wards']} HIGH</b> risk wards requiring immediate resource deployment. "
        f"Total estimated deployment cost: <b>₹{plan['estimated_cost_lakhs']:.1f} lakhs</b>. "
        f"Deployment window: <b>{plan['deployment_window_days']} days</b> before monsoon onset.",
        body
    ))
    story.append(Spacer(1, 0.3*cm))

    # City totals table
    story.append(Paragraph("City-Wide Resource Requirements", h2))
    totals_data = [
        ["Resource", "Units Required", "Unit Cost (₹L)", "Total Cost (₹L)"],
        ["Pump Units",      totals["pump_units"],     "1.20", f"{totals['pump_units']*1.20:.1f}"],
        ["NDRF Teams",      totals["ndrf_teams"],     "5.00", f"{totals['ndrf_teams']*5.00:.1f}"],
        ["Flood Barriers",  totals["flood_barriers"], "0.40", f"{totals['flood_barriers']*0.40:.1f}"],
        ["Alert Sirens",    totals["alert_sirens"],   "0.80", f"{totals['alert_sirens']*0.80:.1f}"],
        ["Rescue Boats",    totals["rescue_boats"],   "3.50", f"{totals['rescue_boats']*3.50:.1f}"],
        ["TOTAL", "", "", f"{plan['estimated_cost_lakhs']:.2f}"],
    ]
    t = Table(totals_data, colWidths=[5*cm, 3*cm, 3.5*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c6b")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.whitesmoke, colors.white]),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#ecf0f1")),
        ("FONTNAME",   (0,-1), (-1,-1), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Priority Action Timeline ───────────────────────────────────────────────
    story.append(Paragraph("2. PRIORITY DEPLOYMENT TIMELINE", h1))
    timeline_data = [
        ["Milestone", "Target Date", "Action Required", "Owner"],
        ["T-30 days", "By 1 May",   "Deploy pumps to all CRITICAL wards",    "BBMP SWD"],
        ["T-21 days", "By 10 May",  "NDRF teams briefed + pre-positioned",    "KSNDMC/NDRF"],
        ["T-14 days", "By 17 May",  "Flood barriers installed (CRITICAL)",    "BBMP Eng."],
        ["T-7 days",  "By 24 May",  "Alert sirens tested + hotline active",   "BBMP CDMU"],
        ["T-3 days",  "By 28 May",  "Rescue boats on standby (HIGH wards)",   "NDRF"],
        ["T-0",       "1 June",     "Pre-monsoon readiness confirmed ✓",      "Commissioner"],
    ]
    tl = Table(timeline_data, colWidths=[2.5*cm, 2.5*cm, 7.5*cm, 2.5*cm])
    tl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#fdfefe"), colors.HexColor("#eaf4fb")]),
        ("GRID",       (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(tl)
    story.append(Spacer(1, 0.5*cm))

    # ── Per-Ward Resource Table ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. PER-WARD RESOURCE ALLOCATION (Top 50 Priority Wards)", h1))
    story.append(Paragraph(
        "Sorted by priority rank (1 = highest risk). Full 243-ward table available via API.",
        small
    ))
    story.append(Spacer(1, 0.2*cm))

    ward_header = ["Rank", "Ward Name", "Risk", "Score", "Pumps", "NDRF", "Barriers", "Sirens", "Boats"]
    ward_rows   = [ward_header]
    for w in wards[:50]:
        rc = RISK_COLOR.get(w["risk_level"], colors.grey)
        ward_rows.append([
            str(w["priority_rank"]),
            w["ward_name"][:22],
            w["risk_level"],
            str(w["readiness_score"]),
            str(w["pump_units"]),
            str(w["ndrf_teams"]),
            str(w["flood_barriers"]),
            str(w["alert_sirens"]),
            str(w["rescue_boats"]),
        ])

    wt = Table(ward_rows, colWidths=[1.2*cm, 5.5*cm, 2*cm, 1.5*cm,
                                      1.5*cm, 1.3*cm, 1.8*cm, 1.5*cm, 1.3*cm])
    risk_ts = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c6b")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ALIGN",      (1,0), (1,-1), "LEFT"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    # Colour risk cells
    for idx, row in enumerate(wards[:50], start=1):
        rc = RISK_COLOR.get(row["risk_level"], colors.grey)
        risk_ts.append(("BACKGROUND", (2, idx), (2, idx), rc))
        risk_ts.append(("TEXTCOLOR",  (2, idx), (2, idx), colors.white))
        risk_ts.append(("FONTNAME",   (2, idx), (2, idx), "Helvetica-Bold"))
        if idx % 2 == 0:
            risk_ts.append(("ROWBACKGROUNDS", (0, idx), (1, idx), [colors.HexColor("#f5f7fa")]))

    wt.setStyle(TableStyle(risk_ts))
    story.append(wt)

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "Generated by HydraGIS Urban Flood Prediction Engine v1.0 | "
        "Model: XGBoost (55%) + RF (45%) ensemble on 243 BBMP wards | "
        "Data: BBMP SWD audits, IMD rainfall, SRTM DEM, Census 2011",
        small
    ))

    doc.build(story)
    buf.seek(0)

    filename = f"HydraGIS_Deployment_Plan_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/")
def root():
    return {"message": "HydraGIS Flood Intelligence API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
