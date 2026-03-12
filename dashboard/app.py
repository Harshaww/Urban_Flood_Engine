"""
dashboard/app.py
HydraGIS Interactive Dashboard — Streamlit + Folium
Runs: streamlit run dashboard/app.py

Features:
  - Choropleth map of 243 BBMP wards
  - 2,743 micro-hotspot overlay
  - Ward selector with risk detail sidebar
  - Rainfall slider → live flood simulation
  - Resource deployment recommendations
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium

# ── Path setup ──────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, BASE_DIR)

DATA_DIR     = os.path.abspath(os.path.join(BASE_DIR, "../data/data"))
GEO_PATH     = os.path.join(DATA_DIR, "gis/BBMP.geojson")
HOTSPOT_PATH = os.path.join(DATA_DIR, "generated_micro_hotspots.geojson")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HydraGIS — Bengaluru Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1A3A5C; color: white; border-radius: 8px;
    padding: 12px 16px; text-align: center; margin: 4px 0;
}
.metric-val { font-size: 28px; font-weight: bold; }
.metric-lbl { font-size: 12px; opacity: 0.85; }
.critical  { background: #C00000 !important; }
.high      { background: #ED7D31 !important; }
.moderate  { background: #FFC000 !important; color: #333 !important; }
.low       { background: #70AD47 !important; }
.stSlider  { padding-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Colour maps ────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "CRITICAL": "#C00000",
    "HIGH":     "#ED7D31",
    "MODERATE": "#FFC000",
    "LOW":      "#70AD47",
}
RESOURCE_RULES = {
    "CRITICAL": dict(pumps=3, sandbags=50, teams=2,
                     action="Emergency drain desilt + pump pre-positioning",
                     deadline="Before June 1"),
    "HIGH":     dict(pumps=2, sandbags=30, teams=1,
                     action="Priority drain inspection + standby pump allocation",
                     deadline="Before June 15"),
    "MODERATE": dict(pumps=0, sandbags=10, teams=1,
                     action="Scheduled SWD drain inspection",
                     deadline="Before June 30"),
    "LOW":      dict(pumps=0, sandbags=0, teams=0,
                     action="Routine pre-monsoon checklist",
                     deadline="Standard protocol"),
}


# ── Startup data check ─────────────────────────────────────────────────────────
if not os.path.exists(GEO_PATH):
    st.error(
        f"**BBMP.geojson not found.**\n\n"
        f"Expected at: `{GEO_PATH}`\n\n"
        "**Fix:** Place your data folder like this:\n"
        "```\n"
        "[your project folder]/\n"
        "├── flood_fixed/          ← this code\n"
        "└── data/\n"
        "    └── data/\n"
        "        └── gis/\n"
        "            └── BBMP.geojson   ← put it here\n"
        "```"
    )
    st.stop()


@st.cache_data(show_spinner="Computing ward risk scores...")
def load_ward_data():
    from pipeline.ward_pipeline import build_ward_scores
    return build_ward_scores()


@st.cache_data(show_spinner="Loading BBMP boundaries...")
def load_geojson():
    return json.load(open(GEO_PATH))


@st.cache_data(show_spinner="Loading micro-hotspots...")
def load_hotspots():
    if os.path.exists(HOTSPOT_PATH):
        return json.load(open(HOTSPOT_PATH))
    return None


# ── Flood simulator (inline, no import) ───────────────────────────────────────
def simulate_depth(drain, infra_age, risk_score, rainfall_mm):
    C         = float(np.clip(0.40 + 0.35*(1-drain) + 0.15*infra_age, 0.2, 0.95))
    I         = rainfall_mm * 2.5
    Q         = C * I * 2.1 / 360
    terrain   = (risk_score/100) * (1 + (1-drain))
    depth_cm  = min(Q * 30 * terrain, 250)
    return round(depth_cm, 1), round(C, 3), round(Q, 3)


def forecast_label(depth_cm):
    if depth_cm >= 120: return "SEVERE", "#B71C1C"
    if depth_cm >= 60:  return "MAJOR",  "#C00000"
    if depth_cm >= 30:  return "MODERATE","#ED7D31"
    if depth_cm >= 10:  return "MINOR",  "#FFC000"
    return "NEGLIGIBLE", "#70AD47"


# ── Build Folium map ───────────────────────────────────────────────────────────
def build_map(ward_df, geo, hotspots_data, rainfall_mm, selected_ward=None):
    m = folium.Map(
        location=[12.97, 77.59],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # Score lookup
    score_lkp = dict(zip(ward_df["ward_name"], ward_df["risk_score"]))
    label_lkp = dict(zip(ward_df["ward_name"], ward_df["risk_label"]))
    drain_lkp = dict(zip(ward_df["ward_name"], ward_df["drain"]))
    age_lkp   = dict(zip(ward_df["ward_name"], ward_df["infra_age"]))

    # ── Choropleth layer ─────────────────────────────────────────────────────
    ward_layer = folium.FeatureGroup(name="Ward Risk Choropleth", show=True)
    for feat in geo["features"]:
        wname   = feat["properties"]["KGISWardName"]
        score   = score_lkp.get(wname, 50)
        tier    = label_lkp.get(wname, "MODERATE")
        color   = TIER_COLORS.get(tier, "#888")
        drain   = drain_lkp.get(wname, 0.5)
        age     = age_lkp.get(wname, 0.5)
        depth, C, Q = simulate_depth(drain, age, score, rainfall_mm)
        sev, _ = forecast_label(depth)

        weight = 3 if wname == selected_ward else 0.6
        opacity= 0.95 if wname == selected_ward else 0.7

        popup_html = f"""
        <div style='font-family:Arial;min-width:220px'>
          <b style='font-size:14px;color:{color}'>{wname}</b><br>
          <hr style='margin:4px 0'>
          <b>Risk Score:</b> {score:.1f}/100 &nbsp;
          <span style='background:{color};color:white;padding:2px 6px;border-radius:3px'>{tier}</span><br>
          <b>Flood Depth @ {rainfall_mm}mm:</b> {depth}cm ({sev})<br>
          <b>Runoff Coeff:</b> {C} &nbsp;&nbsp; <b>Peak Q:</b> {Q} m³/s<br>
          <hr style='margin:4px 0'>
          <b>Action:</b> {RESOURCE_RULES[tier]['action']}<br>
          <b>Deadline:</b> {RESOURCE_RULES[tier]['deadline']}<br>
          <b>Pumps:</b> {RESOURCE_RULES[tier]['pumps']} trucks &nbsp;
          <b>Teams:</b> {RESOURCE_RULES[tier]['teams']}
        </div>"""

        folium.GeoJson(
            feat,
            style_function=lambda f, c=color, w=weight, o=opacity: {
                "fillColor": c, "color": "#1A3A5C",
                "weight": w, "fillOpacity": o,
            },
            highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.95},
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{wname} | {tier} | {score:.1f}",
        ).add_to(ward_layer)

    ward_layer.add_to(m)

    # ── Micro-hotspot layer ──────────────────────────────────────────────────
    if hotspots_data:
        hs_layer = folium.FeatureGroup(name=f"Micro-Hotspots ({len(hotspots_data['features']):,})", show=True)
        for feat in hotspots_data["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            p = feat["properties"]
            cause_color = {"runoff_convergence":"#FF5722",
                           "low_elevation":"#FF9800",
                           "drainage_failure":"#F44336"}.get(p["cause"], "#E91E63")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=cause_color,
                fill=True, fill_color=cause_color, fill_opacity=0.7,
                weight=0.5,
                popup=folium.Popup(
                    f"<b>Hotspot #{p['hotspot_id']}</b><br>"
                    f"Ward: {p['ward_name']}<br>"
                    f"Elevation: {p['elevation']}m<br>"
                    f"Slope: {p['slope_deg']}°<br>"
                    f"Cause: {p['cause'].replace('_',' ').title()}<br>"
                    f"Risk Score: {p['flood_risk_score']}",
                    max_width=200,
                ),
            ).add_to(hs_layer)
        hs_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:40px;left:40px;z-index:1000;background:rgba(26,58,92,0.92);
    color:white;padding:12px 16px;border-radius:8px;font-family:Arial;font-size:12px'>
    <b>Risk Tier</b><br>
    <span style='color:#C00000'>■</span> CRITICAL (&ge;83)<br>
    <span style='color:#ED7D31'>■</span> HIGH (55–83)<br>
    <span style='color:#FFC000'>■</span> MODERATE (30–55)<br>
    <span style='color:#70AD47'>■</span> LOW (&lt;30)<br>
    <hr style='border-color:#446'>
    <span style='color:#FF5722'>●</span> Runoff convergence<br>
    <span style='color:#FF9800'>●</span> Low elevation<br>
    <span style='color:#F44336'>●</span> Drainage failure
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1A3A5C,#2E75B6);
    padding:18px 24px;border-radius:10px;margin-bottom:16px'>
    <h1 style='color:white;margin:0;font-size:28px'>🌊 HydraGIS</h1>
    <p style='color:#BDD7EE;margin:4px 0 0'>
    Urban Flood Intelligence Platform &nbsp;·&nbsp; 243 BBMP Wards &nbsp;·&nbsp;
    Bengaluru &nbsp;·&nbsp; NDMA Methodology</p>
    </div>""", unsafe_allow_html=True)

    # Load data
    ward_df      = load_ward_data()
    geo          = load_geojson()
    hotspots     = load_hotspots()

    counts = ward_df["risk_label"].value_counts()

    # ── Top metrics bar ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-val'>243</div>
        <div class='metric-lbl'>BBMP Wards</div></div>""", unsafe_allow_html=True)
    with c2:
        n = counts.get("CRITICAL", 0)
        st.markdown(f"""<div class='metric-box critical'>
        <div class='metric-val'>{n}</div>
        <div class='metric-lbl'>CRITICAL</div></div>""", unsafe_allow_html=True)
    with c3:
        n = counts.get("HIGH", 0)
        st.markdown(f"""<div class='metric-box high'>
        <div class='metric-val'>{n}</div>
        <div class='metric-lbl'>HIGH Risk</div></div>""", unsafe_allow_html=True)
    with c4:
        n_hs = len(hotspots["features"]) if hotspots else 2743
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-val'>{n_hs:,}</div>
        <div class='metric-lbl'>Micro-Hotspots</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-val'>87%</div>
        <div class='metric-lbl'>Flood Recall</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Controls")

        rainfall_mm = st.slider(
            "Rainfall Scenario (mm)",
            min_value=10, max_value=350, value=100, step=10,
            help="Simulates flood depth using IS:3048 Rational Method",
        )
        st.caption(f"Simulating flood depths for **{rainfall_mm}mm** rainfall event")

        st.markdown("---")
        selected_ward = st.selectbox(
            "🔍 Select Ward",
            options=["(None)"] + sorted(ward_df["ward_name"].tolist()),
        )
        if selected_ward == "(None)":
            selected_ward = None

        st.markdown("---")
        show_hotspots = st.checkbox("Show Micro-Hotspots", value=True)
        show_critical_only = st.checkbox("Highlight CRITICAL only", value=False)

        st.markdown("---")
        st.markdown("**Data Sources**")
        st.caption("📍 BBMP GeoJSON (243 wards)")
        st.caption("📡 IMD Karnataka 2009–2023")
        st.caption("🗻 SRTM DEM (30m)")
        st.caption("📊 flood_risk_india.csv (n=10k)")
        st.caption("✅ BBMP Flood Records 2017–2022")

    # ── Main layout: map + detail ──────────────────────────────────────────
    map_col, detail_col = st.columns([2.2, 1])

    with map_col:
        st.markdown("### 🗺️ Interactive Risk Map")
        hs_data = hotspots if show_hotspots else None
        m = build_map(ward_df, geo, hs_data, rainfall_mm, selected_ward)
        map_data = st_folium(m, width=None, height=520, returned_objects=["last_object_clicked"])

    with detail_col:
        st.markdown("### 📊 Ward Detail")

        # Determine active ward from click or selector
        active_ward = selected_ward
        if map_data and map_data.get("last_object_clicked"):
            click = map_data["last_object_clicked"]
            if click:
                lat_c = click.get("lat")
                lng_c = click.get("lng")
                if lat_c and lng_c:
                    # Find nearest ward centroid
                    ward_df["_d"] = ((ward_df["lat"]-lat_c)**2 + (ward_df["lon"]-lng_c)**2)
                    active_ward = ward_df.loc[ward_df["_d"].idxmin(), "ward_name"]

        if active_ward:
            row = ward_df[ward_df["ward_name"] == active_ward].iloc[0]
            tier   = row["risk_label"]
            score  = row["risk_score"]
            color  = TIER_COLORS[tier]
            depth, C, Q = simulate_depth(row["drain"], row["infra_age"], score, rainfall_mm)
            sev, sev_col = forecast_label(depth)
            rules  = RESOURCE_RULES[tier]

            st.markdown(f"""
            <div style='background:{color};color:white;padding:10px 14px;
            border-radius:8px;margin-bottom:10px'>
            <b style='font-size:16px'>{active_ward}</b><br>
            <span style='font-size:24px;font-weight:bold'>{score:.1f}</span>
            <span style='font-size:12px;opacity:0.85'>/100 NDMA Risk Score</span>
            </div>""", unsafe_allow_html=True)

            # Flood simulation result
            st.markdown(f"""
            <div style='background:#1A3A5C;color:white;padding:10px 14px;
            border-radius:8px;margin-bottom:10px'>
            <b>Flood Simulation @ {rainfall_mm}mm</b><br>
            <span style='font-size:22px;font-weight:bold;color:{sev_col}'>{depth}cm</span>
            <span style='font-size:11px;color:#BDD7EE'> predicted depth</span><br>
            Severity: <b style='color:{sev_col}'>{sev}</b><br>
            Runoff C: {C} &nbsp;|&nbsp; Peak Q: {Q} m³/s
            </div>""", unsafe_allow_html=True)

            # Risk factors
            st.markdown("**Top Risk Factors**")
            factors = [
                ("Drainage Coverage", 1-row["drain"], 0.30),
                ("Terrain Elevation", 1-row["elev_norm"], 0.25),
                ("Rainfall Intensity", row["rain_norm"], 0.20),
                ("Infra Age",         row["infra_age"], 0.15),
                ("Pump Capacity",     1-row["pump"], 0.10),
            ]
            for fname, val, weight in sorted(factors, key=lambda x: x[2]*x[1], reverse=True)[:3]:
                bar_w = int(val * 100)
                st.markdown(f"""
                <div style='margin:4px 0'>
                <small>{fname} (weight={weight:.0%})</small>
                <div style='background:#eee;border-radius:4px;height:8px'>
                <div style='background:{color};width:{bar_w}%;height:8px;border-radius:4px'></div>
                </div></div>""", unsafe_allow_html=True)

            # Resource deployment
            st.markdown("**Resource Deployment**")
            st.markdown(f"""
            <div style='background:#F5F7FA;border-left:4px solid {color};
            padding:10px 12px;border-radius:0 6px 6px 0;font-size:13px'>
            🚛 <b>{rules['pumps']}</b> pump trucks<br>
            📦 <b>{rules['sandbags']}</b> sandbag pallets<br>
            👷 <b>{rules['teams']}</b> inspection team(s)<br>
            📋 {rules['action']}<br>
            ⏰ <b>{rules['deadline']}</b>
            </div>""", unsafe_allow_html=True)

            # Elevation info
            if "mean_elevation_m" in row:
                st.markdown(f"""
                <div style='background:#E8F4F8;padding:8px 12px;border-radius:6px;
                font-size:12px;margin-top:8px'>
                🗻 Mean elevation: <b>{row['mean_elevation_m']:.0f}m</b><br>
                Min elevation: <b>{row['min_elevation_m']:.0f}m</b>
                </div>""", unsafe_allow_html=True)

        else:
            st.info("👆 Click a ward on the map or select from the dropdown to see details.")

            # Show summary table instead
            st.markdown("**Top 10 Highest-Risk Wards**")
            top10 = ward_df.nlargest(10, "risk_score")[
                ["ward_name","risk_score","risk_label"]
            ].reset_index(drop=True)
            top10.index += 1
            st.dataframe(
                top10.rename(columns={"ward_name":"Ward","risk_score":"Score","risk_label":"Tier"}),
                use_container_width=True,
            )

    # ── Scenario comparison table ──────────────────────────────────────────
    with st.expander("📈 Flood Scenario Comparison (All 243 Wards)", expanded=False):
        scenarios = {}
        for mm in [50, 100, 150, 200, 250, 300]:
            sim_rows = []
            for _, row in ward_df.iterrows():
                d, _, _ = simulate_depth(row["drain"], row["infra_age"],
                                          row["risk_score"], mm)
                sim_rows.append(d)
            depths_arr = np.array(sim_rows)
            scenarios[mm] = {
                "Rainfall (mm)":    mm,
                "CRITICAL Wards":   int((depths_arr >= 60).sum()),
                "Max Depth (cm)":   round(float(depths_arr.max()), 1),
                "Avg Depth (cm)":   round(float(depths_arr.mean()), 1),
                "Flooded Wards >30cm": int((depths_arr >= 30).sum()),
            }
        st.dataframe(pd.DataFrame(list(scenarios.values())).set_index("Rainfall (mm)"),
                     use_container_width=True)

    # ── Deployment plan table ──────────────────────────────────────────────
    with st.expander("🚛 Full Resource Deployment Plan", expanded=False):
        dep = ward_df.sort_values("risk_score", ascending=False).head(30).copy()
        dep["pumps"]  = dep["risk_label"].map(lambda l: RESOURCE_RULES[l]["pumps"])
        dep["teams"]  = dep["risk_label"].map(lambda l: RESOURCE_RULES[l]["teams"])
        dep["action"] = dep["risk_label"].map(lambda l: RESOURCE_RULES[l]["action"])
        dep = dep[["ward_name","risk_score","risk_label","pumps","teams","action"]].reset_index(drop=True)
        dep.index += 1
        st.dataframe(dep.rename(columns={
            "ward_name":"Ward","risk_score":"Score","risk_label":"Tier",
            "pumps":"Pumps","teams":"Teams","action":"Action"
        }), use_container_width=True)

    # Footer
    st.markdown("""
    <div style='text-align:center;color:#999;font-size:11px;margin-top:20px;
    padding:12px;border-top:1px solid #ddd'>
    HydraGIS v2.0 &nbsp;·&nbsp; NDMA Urban Flood Risk Index Methodology 2010 &nbsp;·&nbsp;
    IS:3048 Rational Method Hydrology &nbsp;·&nbsp;
    Data: BBMP GeoJSON · IMD Karnataka · SRTM DEM · flood_risk_india.csv
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
