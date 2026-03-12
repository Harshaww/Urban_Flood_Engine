# api/schemas.py  — All Pydantic request & response models

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"


# ── Responses ─────────────────────────────────────────────────────────────────

class WardScore(BaseModel):
    ward_id:           str
    name:              str
    lat:               Optional[float] = None
    lng:               Optional[float] = None
    # Opt-in polygon (outer ring) — returned only when ?include_polygons=true
    polygon:           Optional[List[List[float]]] = None
    readiness_score:   float = Field(..., ge=0, le=100)
    risk_level:        RiskLevel
    xgb_score:         float
    rf_score:          float

    elevation:         Optional[int] = None
    drainage_pct:      Optional[int] = None
    rainfall_avg:      Optional[int] = None
    sewer_age:         Optional[int] = None
    pump_stations:     Optional[int] = None

    lakes:             int
    population:        int
    hotspot_count:     int
    composite_vulnerability: float
    drain_deficit:     float
    runoff_coefficient: float
    feature_contributions: Dict[str, float]
    deployment_priority:   int


class WardScoresResponse(BaseModel):
    total_wards:          int
    critical_count:       int
    high_count:           int
    city_readiness_index: float   = Field(..., description="Average readiness score across all wards")
    wards:                List[WardScore]


class HotspotPoint(BaseModel):
    lat:        float
    lon:        float
    ward_id:    str
    ward_name:  str
    severity:   str
    depth_m:    float
    area_m2:    float
    volume_m3:  float
    source:     Optional[str] = None   # dem_low_elevation | polygon_interior | centroid_scatter


class HotspotsResponse(BaseModel):
    total_hotspots: int
    critical_count: int
    high_count:     int
    hotspots:       List[HotspotPoint]


class WardDeployPlan(BaseModel):
    ward_id:         str
    ward_name:       str
    risk_level:      RiskLevel
    readiness_score: float
    pump_units:      int
    ndrf_teams:      int
    flood_barriers:  int
    alert_sirens:    int
    rescue_boats:    int
    priority_rank:   int


class DeploymentResponse(BaseModel):
    total_wards_assessed:  int
    critical_wards:        int
    high_wards:            int
    moderate_wards:        int
    deployment_window_days: int
    ward_plans:            List[WardDeployPlan]
    city_totals:           Dict[str, int]
    estimated_cost_lakhs:  float


class TrainMetrics(BaseModel):
    status:           str
    random_forest:    Dict[str, Any]
    xgboost:          Dict[str, Any]
    ensemble_weights: Dict[str, float]
    kaggle_rows_used: int


class TrainTaskResponse(BaseModel):
    task_id:  str
    status:   str = "PENDING"
    message:  str = "Training queued. Poll GET /model/train/status/{task_id}"


class WhatIfRequest(BaseModel):
    ward_id:   str          = Field(..., example="W02")
    overrides: Dict[str, float] = Field(
        ...,
        example={"drainage_pct": 60, "pump_stations": 8},
        description="Ward parameters to override for scenario analysis"
    )


class WhatIfResponse(BaseModel):
    ward_id:          str
    readiness_score:  float
    risk_level:       str
    applied_overrides: Dict[str, float]
    improvement:      Optional[float] = Field(None, description="Score delta vs baseline")


class UploadResponse(BaseModel):
    filename:     str
    rows_loaded:  int
    columns:      List[str]
    preview:      List[Dict]
    status:       str


class HealthResponse(BaseModel):
    status:         str
    models_loaded:  bool
    version:        str
    ward_count:     int

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — Real-Time Flood Depth Prediction
# ══════════════════════════════════════════════════════════════════════════════

class FloodDepthRequest(BaseModel):
    ward_id:               str   = Field(..., example="W02")
    rainfall_mm:           float = Field(..., example=280.0, ge=0, le=1000)
    drainage_failure_pct:  float = Field(0.0, example=20.0, ge=0, le=100,
                                         description="% of drainage capacity lost (0 = fully working)")

class FloodDepthResponse(BaseModel):
    ward_id:               str
    ward_name:             str
    rainfall_mm:           float
    drainage_failure_pct:  float
    flood_depth_m:         float
    flood_duration_hours:  float
    affected_population:   int
    road_closures:         int
    excess_volume_m3:      float
    runoff_coefficient:    float
    effective_drainage_m3h: float
    severity:              str
    pumps_needed:          int
    risk_level:            str
    readiness_score:       Optional[float]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — Flood Spread Map
# ══════════════════════════════════════════════════════════════════════════════

class FloodGridPoint(BaseModel):
    lat:       float
    lng:       float
    depth_m:   float
    ward_id:   str
    ward_name: str
    severity:  str

class FloodSpreadResponse(BaseModel):
    rainfall_mm:                float
    drainage_failure_pct:       float
    flooded_wards_count:        int
    total_affected_population:  int
    max_depth_m:                float
    grid_points:                int
    flood_grid:                 List[FloodGridPoint]
    flooded_wards_summary:      List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — Monsoon Night Simulator
# ══════════════════════════════════════════════════════════════════════════════

class MonsoonSimRequest(BaseModel):
    rainfall_mm:           float = Field(..., example=310.0, ge=0, le=1000,
                                         description="Rainfall intensity in mm")
    drainage_failure_pct:  float = Field(0.0, example=20.0, ge=0, le=100,
                                         description="% of drainage network failed (blocked/broken)")

class TimelineEvent(BaseModel):
    time:    str
    event:   str
    status:  str
    details: str

class MonsoonSimResponse(BaseModel):
    scenario:            Dict[str, Any]
    impact:              Dict[str, Any]
    resources_needed:    Dict[str, int]
    wards_to_evacuate:   List[Dict[str, Any]]
    timeline:            List[TimelineEvent]
    all_flooded_wards:   List[Dict[str, Any]]
