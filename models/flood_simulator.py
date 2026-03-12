"""
models/flood_simulator.py
Event-driven flood depth simulator using IS:3048 Rational Method hydrology.
Supports POST /simulate with configurable rainfall scenarios.
"""

import math
import numpy as np
import pandas as pd

# IS:3048 Rational Method: Q = C × I × A / 360
# Q = peak discharge (m³/s)
# C = runoff coefficient (drainage coverage proxy)
# I = rainfall intensity (mm/hr)
# A = ward area (km²)

WARD_AREA_KM2   = 2.1    # average BBMP ward area
PEAK_MULTIPLIER = 2.5    # converts event mm to peak mm/hr intensity
MAX_DEPTH_CM    = 250    # physical cap

SEVERITY_THRESHOLDS = {
    "SEVERE":     120,   # cm — building ground floor inundation
    "MAJOR":       60,   # cm — road impassable
    "MODERATE":    30,   # cm — waterlogging
    "MINOR":       10,   # cm — surface ponding
    "NEGLIGIBLE":   0,
}


def compute_runoff_coefficient(drain: float, infra_age: float) -> float:
    """
    Effective runoff coefficient C (0.2 – 0.95).
    Poor drainage + old infra → high C (more runoff, less absorption).
    """
    base_C = 0.40 + 0.35 * (1 - drain) + 0.15 * infra_age
    return float(np.clip(base_C, 0.20, 0.95))


def simulate_ward(
    ward_name:    str,
    rainfall_mm:  float,
    drain:        float,
    infra_age:    float,
    min_elev_m:   float,
    risk_score:   float,
    area_km2:     float = WARD_AREA_KM2,
) -> dict:
    """
    Simulate flood depth for a single ward given rainfall scenario.
    """
    C = compute_runoff_coefficient(drain, infra_age)
    I = rainfall_mm * PEAK_MULTIPLIER   # peak intensity mm/hr
    Q = C * I * area_km2 / 360          # m³/s peak discharge

    # Depth: Q scaled by terrain depression factor
    # Flatter + lower elevation = more water accumulation
    terrain_factor = (risk_score / 100) * (1 + (1 - drain))
    depth_cm = min(Q * 30 * terrain_factor, MAX_DEPTH_CM)

    # Severity
    severity = "NEGLIGIBLE"
    for sev, threshold in SEVERITY_THRESHOLDS.items():
        if depth_cm >= threshold:
            severity = sev
            break

    # Time to inundation (hours): quicker for CRITICAL wards
    tti = max(0.5, 6 - (risk_score / 100) * 4)

    # Water accumulation volume (m³)
    area_m2    = area_km2 * 1e6
    volume_m3  = round((depth_cm / 100) * area_m2 * (C * 0.3), 0)

    return {
        "ward_name":             ward_name,
        "rainfall_mm":           rainfall_mm,
        "runoff_coefficient":    round(C, 3),
        "peak_discharge_m3s":    round(Q, 3),
        "predicted_flood_depth_cm": round(depth_cm, 1),
        "expected_water_m3":     int(volume_m3),
        "severity":              severity,
        "time_to_inundation_hr": round(tti, 1),
        "risk_level":            "CRITICAL" if depth_cm >= 60 else (
                                 "HIGH"     if depth_cm >= 30 else (
                                 "MODERATE" if depth_cm >= 10 else "LOW")),
    }


def simulate_all_wards(
    ward_df:     pd.DataFrame,
    rainfall_mm: float,
) -> pd.DataFrame:
    """
    Run flood simulation for all wards.
    ward_df must have: ward_name, drain, infra_age, min_elevation_m, risk_score
    """
    results = []
    for _, row in ward_df.iterrows():
        result = simulate_ward(
            ward_name   = row["ward_name"],
            rainfall_mm = rainfall_mm,
            drain       = row.get("drain", 0.5),
            infra_age   = row.get("infra_age", 0.5),
            min_elev_m  = row.get("min_elevation_m", 850.0),
            risk_score  = row.get("risk_score", 50.0),
        )
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("predicted_flood_depth_cm", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def scenario_report(ward_df: pd.DataFrame) -> dict:
    """
    Generate multi-scenario summary for report/dashboard.
    Scenarios: 50, 100, 150, 200, 250, 300 mm
    """
    scenarios = {}
    for mm in [50, 100, 150, 200, 250, 300]:
        sim = simulate_all_wards(ward_df, mm)
        scenarios[mm] = {
            "rainfall_mm":   mm,
            "critical_wards": int((sim["risk_level"] == "CRITICAL").sum()),
            "high_wards":    int((sim["risk_level"] == "HIGH").sum()),
            "max_depth_cm":  float(sim["predicted_flood_depth_cm"].max()),
            "avg_depth_cm":  float(sim["predicted_flood_depth_cm"].mean().round(1)),
            "top_5":         sim.head(5)[["ward_name","predicted_flood_depth_cm","severity"]].to_dict("records"),
        }
    return scenarios


if __name__ == "__main__":
    from pipeline.ward_pipeline import build_ward_scores
    df  = build_ward_scores()
    sim = simulate_all_wards(df, rainfall_mm=200)
    print(sim.head(10)[["ward_name","predicted_flood_depth_cm","severity","risk_level"]].to_string())
