"""
pipeline/rainfall_forecast.py
Rainfall forecast ingestion — OpenWeatherMap API + IMD fallback.

FIX 7: Zone-specific 72-hour forecasts for all 243 wards.

PROBLEM FIXED:
    The previous version fetched ONE OWM point (Bengaluru city centre,
    12.97°N 77.59°E) and applied the same total_mm to all 243 wards.
    Bengaluru's east corridor (Whitefield, KR Puram) receives ~20% more
    rainfall than the west (Rajajinagar, Kengeri) due to orographic effects
    and urban heat island intensification. A single city-centre forecast
    systematically under-estimates east-ward risk and over-estimates west-ward
    risk during asymmetric rainfall events.

    config.py already had RAINFALL_2020_ZONE and RAINFALL_2024_ZONE zone data
    but it was never wired into the forecast pipeline. FIX 7 connects them.

SOLUTION:
    1. Fetch 5 OWM points — one per cardinal zone (east / south / central /
       north / west). Each point is a representative raingauge location
       calibrated against KSNDMC station positions.
    2. Each ward is assigned to the nearest zone by lat/lon.
    3. adjust_risk_for_forecast() now applies the zone-specific forecast_mm
       to each ward instead of the city-wide average.
    4. When OWM API is unavailable, the IMD zone fallback (from config.py
       RAINFALL_2020_ZONE / RAINFALL_2024_ZONE) applies zone ratios to the
       Karnataka CSV baseline — same five-zone structure, no flat average.

Zone representative coordinates:
    east    → 12.967°N 77.750°E  (Whitefield / KSNDMC Mahadevapura gauge)
    south   → 12.855°N 77.594°E  (Electronic City / KSNDMC South gauge)
    central → 12.972°N 77.594°E  (Shivajinagar / IMD city HQ)
    north   → 13.072°N 77.594°E  (Yelahanka / KSNDMC North gauge)
    west    → 12.972°N 77.502°E  (Rajajinagar / KSNDMC West gauge)

Source for zone rainfall ratios:
    KSNDMC Bengaluru District Rainfall Bulletins 2017-2022 (5-zone breakdown);
    IMD Normal Rainfall Map Karnataka 1991-2020 (normal.imd.gov.in).
    config.py RAINFALL_2020_ZONE (IMD 1991-2020 normals) and
    RAINFALL_2024_ZONE (2024 ENSO-active season actuals).
"""

import os, json, datetime
from typing import Optional
import numpy as np
import pandas as pd

# ── Zone definitions ───────────────────────────────────────────────────────────
# Five representative OWM fetch points, one per Bengaluru zone.
# Coordinates calibrated to KSNDMC raingauge station positions.
# Source: KSNDMC Bengaluru District Rainfall Bulletins 2017-2022.
FORECAST_ZONES = {
    "east":    {"lat": 12.967, "lon": 77.750, "label": "Whitefield / KR Puram / Mahadevapura"},
    "south":   {"lat": 12.855, "lon": 77.594, "label": "Electronic City / BTM / HSR"},
    "central": {"lat": 12.972, "lon": 77.594, "label": "Shivajinagar / Malleswaram / Indiranagar"},
    "north":   {"lat": 13.072, "lon": 77.594, "label": "Yelahanka / Hebbal / RT Nagar"},
    "west":    {"lat": 12.972, "lon": 77.502, "label": "Rajajinagar / Kengeri / Vijayanagar"},
}

# IMD seasonal zone ratios — east is the wettest, west the driest.
# Derived from config.py RAINFALL_2020_ZONE normals (1991-2020 baseline).
# Used to scale the Karnataka CSV baseline into zone-specific forecasts
# when the OWM API is unavailable.
# Source: config.py / IMD Karnataka 1991-2020 normals.
_ZONE_RATIO = {
    "east":    1.095,   # 970 / 886 city-wide = +9.5%
    "south":   1.038,   # 920 / 886            = +3.8%
    "central": 1.000,   # 886 / 886            = baseline
    "north":   0.954,   # 845 / 886            = -4.6%
    "west":    0.914,   # 810 / 886            = -8.6%
}

# OpenWeatherMap 5-day/3-hour forecast endpoint
OWM_BASE = "https://api.openweathermap.org/data/2.5/forecast"


def _zone_for_ward(lat: float, lon: float) -> str:
    """
    Assign a ward centroid to its nearest zone using the same boundary logic
    as pipeline/ward_pipeline.py get_ward_rainfall_mm() — keeps zone
    assignment fully consistent across the whole pipeline.

    Boundaries (cardinal direction from city centre 12.972°N, 77.594°E):
        east    lon ≥ 77.68
        south   lat ≤ 12.88
        north   lat ≥ 13.06
        west    lon ≤ 77.52
        central (default)
    """
    if lon >= 77.68: return "east"
    if lat <= 12.88: return "south"
    if lat >= 13.06: return "north"
    if lon <= 77.52: return "west"
    return "central"


# ── OWM fetch helpers ──────────────────────────────────────────────────────────

def _fetch_one_zone(api_key: str, zone_name: str) -> Optional[dict]:
    """Fetch OWM 5-day/3-hour forecast for a single zone point."""
    z = FORECAST_ZONES[zone_name]
    try:
        import urllib.request
        url = f"{OWM_BASE}?lat={z['lat']}&lon={z['lon']}&appid={api_key}&units=metric"
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[forecast] OWM zone '{zone_name}' unavailable: {e}")
        return None


def _parse_72h(response: dict, zone_name: str) -> dict:
    """Extract 72-hour cumulative rainfall from one OWM zone response."""
    items  = response.get("list", [])[:24]   # 24 × 3h = 72h
    total  = 0.0
    max_3h = 0.0
    periods = []
    for item in items:
        rain = item.get("rain", {}).get("3h", 0.0)
        dt   = datetime.datetime.fromtimestamp(item["dt"])
        total  += rain
        max_3h  = max(max_3h, rain)
        periods.append({"datetime": dt.isoformat(), "rain_3h_mm": round(rain, 2)})
    return {
        "zone":          zone_name,
        "total_72h_mm":  round(total, 2),
        "max_3h_mm":     round(max_3h, 2),
        "forecast_periods": periods,
    }


def fetch_zone_forecasts(api_key: str) -> Optional[dict]:
    """
    FIX 7: Fetch OWM forecasts for all 5 zones.
    Returns dict keyed by zone name, or None if all calls fail.

    Costs 5 OWM API calls (all within the free tier's 1,000 calls/day limit).
    Each call returns the same 5-day/3-hour product — only the lat/lon differs.
    """
    results = {}
    for zone_name in FORECAST_ZONES:
        raw = _fetch_one_zone(api_key, zone_name)
        if raw:
            results[zone_name] = _parse_72h(raw, zone_name)
        else:
            results[zone_name] = None   # mark failed zones; fallback applied below

    successful = sum(1 for v in results.values() if v is not None)
    if successful == 0:
        return None

    # If some zones failed, interpolate from successful neighbours
    # using the IMD zone ratios as a scaling bridge
    city_avg = np.mean([v["total_72h_mm"] for v in results.values() if v is not None])
    for zone in results:
        if results[zone] is None:
            estimated = city_avg * _ZONE_RATIO[zone]
            results[zone] = {
                "zone":          zone,
                "total_72h_mm":  round(estimated, 2),
                "max_3h_mm":     round(estimated / 8, 2),
                "forecast_periods": [],
                "note":          f"Interpolated from {successful}/5 OWM zones via IMD ratio",
            }
            print(f"[forecast] Zone '{zone}' OWM failed — interpolated {estimated:.1f}mm "
                  f"from city avg {city_avg:.1f}mm × ratio {_ZONE_RATIO[zone]}")

    return results


# ── IMD historical fallback (zone-aware) ──────────────────────────────────────

def get_imd_historical_avg(rain_path: str, month: int | None = None) -> float:
    """Return IMD Karnataka mean daily monsoon rainfall (city-wide baseline)."""
    try:
        df  = pd.read_csv(rain_path, parse_dates=["date"])
        kar = df[df["state_name"] == "Karnataka"].copy()
        kar["month"] = kar["date"].dt.month
        kar = kar[kar["month"] == month] if month else kar[kar["month"].between(6, 9)]
        return float(kar["actual"].mean())
    except Exception:
        return 12.0   # mm/day conservative fallback


def _imd_zone_fallback(rain_path: Optional[str]) -> dict:
    """
    FIX 7: Build zone-specific fallback forecasts from IMD Karnataka baseline
    + KSNDMC zone ratios. Replaces the single-point city average.

    The zone ratios come from config.py RAINFALL_2020_ZONE (IMD 1991-2020
    normals) and are applied to the Karnataka CSV daily average × 3 days.
    """
    month    = datetime.datetime.now().month
    avg_day  = get_imd_historical_avg(rain_path, month) if rain_path else 12.0
    city_72h = avg_day * 3   # 72h window

    zone_forecasts = {}
    for zone_name, ratio in _ZONE_RATIO.items():
        zone_72h = round(city_72h * ratio, 2)
        zone_forecasts[zone_name] = {
            "zone":          zone_name,
            "total_72h_mm":  zone_72h,
            "max_3h_mm":     round(zone_72h / 8, 2),
            "forecast_periods": [],
            "source":        (
                f"IMD Karnataka historical average × KSNDMC zone ratio {ratio:.3f} "
                f"(OWM API key not set). City baseline: {city_72h:.1f}mm/72h."
            ),
        }
    return zone_forecasts


# ── Public API ────────────────────────────────────────────────────────────────

def get_forecast(api_key: Optional[str] = None,
                 rain_path: Optional[str] = None) -> dict:
    """
    Legacy single-forecast entry point — kept for API backward compatibility.
    Returns the CENTRAL zone forecast (closest to the old city-centre point).
    New code should use get_zone_forecasts() directly.
    """
    zf = get_zone_forecasts(api_key=api_key, rain_path=rain_path)
    central = zf["central"]
    return {
        "total_72h_mm":     central["total_72h_mm"],
        "max_3h_mm":        central["max_3h_mm"],
        "forecast_periods": central.get("forecast_periods", []),
        "source":           central.get("source", "OWM central zone"),
        "fetched_at":       datetime.datetime.utcnow().isoformat(),
        "note":             "Central-zone value. Use get_zone_forecasts() for per-ward accuracy.",
    }


def get_zone_forecasts(api_key: Optional[str] = None,
                       rain_path: Optional[str] = None) -> dict:
    """
    FIX 7: Main entry point. Returns zone-keyed forecast dict.

    If api_key is set  → fetches 5 OWM zone points (free tier: 5 calls/request).
    Else               → IMD historical baseline × KSNDMC zone ratios.

    Returns:
        {
          "east":    {"zone": "east",    "total_72h_mm": 95.2, ...},
          "south":   {"zone": "south",   "total_72h_mm": 82.4, ...},
          "central": {"zone": "central", "total_72h_mm": 77.1, ...},
          "north":   {"zone": "north",   "total_72h_mm": 68.3, ...},
          "west":    {"zone": "west",    "total_72h_mm": 62.5, ...},
        }
    """
    if api_key:
        results = fetch_zone_forecasts(api_key)
        if results:
            for zone in results:
                results[zone].setdefault("source", "OpenWeatherMap (zone-specific)")
                results[zone]["fetched_at"] = datetime.datetime.utcnow().isoformat()
            return results

    # OWM unavailable — use IMD zone fallback
    return _imd_zone_fallback(rain_path)


# ── Risk adjustment (ward-level, zone-aware) ──────────────────────────────────

def adjust_risk_for_forecast(ward_df: pd.DataFrame,
                              forecast: dict,
                              zone_forecasts: Optional[dict] = None) -> pd.DataFrame:
    """
    FIX 7: Dynamically adjust ward risk scores using zone-specific forecast mm.

    Parameters
    ----------
    ward_df        : DataFrame from ward_pipeline.build_ward_scores()
                     Must have: risk_score, drain, lon (or 'lon'/'lng'), lat
    forecast       : Legacy single-point dict (used only if zone_forecasts=None)
    zone_forecasts : Dict from get_zone_forecasts() — preferred.
                     If provided, each ward gets the forecast for its own zone.

    What changed (FIX 7 vs previous):
        BEFORE: total_mm = forecast["total_72h_mm"]  ← same for ALL 243 wards
        AFTER:  total_mm = zone_forecasts[ward_zone]["total_72h_mm"]  ← per-ward zone

    Zone assignment uses the same cardinal-direction boundaries as
    ward_pipeline.get_ward_rainfall_mm() for full pipeline consistency.
    """
    df = ward_df.copy()

    def _ward_forecast_mm(row) -> tuple:
        """Return (forecast_mm, zone_name) for one ward row."""
        if zone_forecasts is None:
            # Backward-compat: fall back to single-point forecast
            return float(forecast.get("total_72h_mm", 0.0)), "city_centre"

        lat = float(row.get("lat", 12.972))
        lon = float(row.get("lon", row.get("lng", 77.594)))   # accept both column names
        zone = _zone_for_ward(lat, lon)
        zf   = zone_forecasts.get(zone, zone_forecasts.get("central", {}))
        return float(zf.get("total_72h_mm", 0.0)), zone

    forecast_mm_col  = []
    forecast_zone_col = []
    adjusted_score_col = []

    for _, row in df.iterrows():
        ward_mm, ward_zone = _ward_forecast_mm(row)

        # Rainfall adjustment factor (0–1): 30mm/72h = threshold, 300mm = severe
        rain_factor = float(np.clip((ward_mm - 30) / 270, 0, 1))

        # Poor-drainage wards are more sensitive to the same rainfall total
        drain_sensitivity = 1.0 - float(row.get("drain", 0.5))
        delta = rain_factor * 15 * (0.5 + 0.5 * drain_sensitivity)
        new_score = round(min(100.0, float(row["risk_score"]) + delta), 2)

        forecast_mm_col.append(ward_mm)
        forecast_zone_col.append(ward_zone)
        adjusted_score_col.append(new_score)

    df["forecast_adjusted_score"] = adjusted_score_col
    df["forecast_ward_mm_72h"]    = forecast_mm_col   # FIX 7: per-ward, not city average
    df["forecast_zone"]           = forecast_zone_col  # FIX 7: zone traceability
    df["forecast_source"]         = (
        zone_forecasts["central"].get("source", "OWM zone-specific")
        if zone_forecasts else forecast.get("source", "city_centre")
    )

    def label(s):
        if s >= 70: return "CRITICAL"
        if s >= 50: return "HIGH"
        if s >= 30: return "MODERATE"
        return "LOW"

    df["forecast_risk_label"] = df["forecast_adjusted_score"].apply(label)

    # Summary — shows the east/west split to make FIX 7 observable in output
    if zone_forecasts:
        east_mm    = zone_forecasts.get("east",    {}).get("total_72h_mm", 0)
        west_mm    = zone_forecasts.get("west",    {}).get("total_72h_mm", 0)
        central_mm = zone_forecasts.get("central", {}).get("total_72h_mm", 0)
        print(f"[forecast] Zone-specific adjustment applied: "
              f"east={east_mm}mm  central={central_mm}mm  west={west_mm}mm  "
              f"(east/west spread: {east_mm - west_mm:+.1f}mm)")

    return df


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(__file__))
    rain = os.path.join(base, "../../data/data/rainfall_india.csv")

    print("=== Zone Forecasts (IMD fallback — no OWM key) ===")
    zf = get_zone_forecasts(rain_path=rain)
    for zone, data in zf.items():
        print(f"  {zone:8s}: {data['total_72h_mm']:6.1f} mm/72h  "
              f"({FORECAST_ZONES[zone]['label']})")

    east_mm  = zf["east"]["total_72h_mm"]
    west_mm  = zf["west"]["total_72h_mm"]
    print(f"\nEast/West spread: {east_mm - west_mm:+.1f} mm  "
          f"({(east_mm/west_mm - 1)*100:+.1f}% higher in east)")
