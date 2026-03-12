import streamlit as st
import requests
import pandas as pd
import folium
import json
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium


# =========================
# API URLs
# =========================

WARD_API = "http://localhost:8000/wards/scores"
HOTSPOT_API = "http://localhost:8000/hotspots"


st.set_page_config(page_title="Urban Flood Intelligence", layout="wide")

st.title("🌧 Urban Flood Intelligence Dashboard")


# =========================
# Fetch Ward Data
# =========================

ward_response = requests.get(WARD_API)
ward_data = ward_response.json()

wards = pd.DataFrame(ward_data["wards"])


# =========================
# Fetch Hotspot Data
# =========================

hotspot_response = requests.get(HOTSPOT_API)
hotspot_data = hotspot_response.json()

hotspots = pd.DataFrame(hotspot_data["hotspots"])


# Normalize coordinate column names
if "longitude" in hotspots.columns:
    hotspots.rename(columns={"longitude": "lon"}, inplace=True)

if "latitude" in hotspots.columns:
    hotspots.rename(columns={"latitude": "lat"}, inplace=True)

if "lng" in hotspots.columns:
    hotspots.rename(columns={"lng": "lon"}, inplace=True)


# =========================
# City Readiness Score
# =========================

city_score = wards["readiness_score"].mean()

st.metric("City Flood Readiness Index", round(city_score, 1))


# =========================
# Show total predicted hotspots
# =========================

st.metric("Predicted Flood Hotspots", len(hotspots))


# =========================
# Flood Risk Distribution
# =========================

st.subheader("Flood Risk Distribution")

risk_counts = wards["risk_level"].value_counts()

st.bar_chart(risk_counts)


# =========================
# Highest Flood Risk Wards
# =========================

st.subheader("🚨 Highest Flood Risk Wards")

top_risk = wards.sort_values("readiness_score").head(5)

st.table(
    top_risk[
        [
            "name",
            "readiness_score",
            "risk_level",
            "deployment_priority",
            "hotspot_count"
        ]
    ]
)


# =========================
# Emergency Deployment Plan
# =========================

st.subheader("🚑 Emergency Deployment Plan")

deploy = wards.sort_values("deployment_priority").head(5)

st.table(
    deploy[
        [
            "name",
            "deployment_priority",
            "hotspot_count"
        ]
    ]
)


# =========================
# Flood Risk Map
# =========================

st.subheader("🗺 Bengaluru Flood Risk Map")


# Load GeoJSON
with open("data/gis/BBMP.geojson") as f:
    wards_geo = json.load(f)


# Create map
m = folium.Map(
    location=[12.9716, 77.5946],
    zoom_start=11,
    tiles="cartodbpositron"
)


# =========================
# Ward Risk Polygons
# =========================

def get_color(risk):

    colors = {
        "CRITICAL": "#8B0000",
        "HIGH": "#FF3B30",
        "MODERATE": "#FFA500",
        "LOW": "#2ECC71"
    }

    return colors.get(risk, "#2ECC71")


ward_layer = folium.FeatureGroup(name="Ward Flood Risk")

for feature in wards_geo["features"]:

    props = feature["properties"]

    ward_name = (
        props.get("WARD_NAME")
        or props.get("WARDNAME")
        or props.get("name")
        or props.get("WARD")
    )

    if ward_name is None:
        continue

    ward = wards[wards["name"] == ward_name]

    if len(ward) == 0:
        continue

    ward = ward.iloc[0]

    popup_text = f"""
    <b>{ward_name}</b><br>
    Readiness Score: {ward['readiness_score']}<br>
    Risk Level: {ward['risk_level']}<br>
    Flood Hotspots: {ward['hotspot_count']}
    """

    folium.GeoJson(
        feature,
        style_function=lambda x, risk=ward["risk_level"]: {
            "fillColor": get_color(risk),
            "color": "#333333",
            "weight": 1,
            "fillOpacity": 0.6
        },
        highlight_function=lambda x: {
            "weight": 3,
            "color": "#000000"
        },
        tooltip=folium.Tooltip(popup_text)
    ).add_to(ward_layer)

ward_layer.add_to(m)


# =========================
# Flood Hotspot Markers
# =========================

hotspot_cluster = MarkerCluster(name="Flood Hotspots")

for _, row in hotspots.iterrows():

    if row["severity"] == "CRITICAL":
        color = "red"
    elif row["severity"] == "HIGH":
        color = "orange"
    else:
        color = "yellow"

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"""
        Flood Hotspot<br>
        Severity: {row['severity']}<br>
        Ward: {row['ward_id']}
        """
    ).add_to(hotspot_cluster)

hotspot_cluster.add_to(m)


# =========================
# Flood Heatmap Layer
# =========================

heat_data = hotspots[["lat", "lon"]].values.tolist()

HeatMap(
    heat_data,
    name="Flood Intensity Heatmap",
    radius=10,
    blur=15,
    min_opacity=0.4
).add_to(m)


# =========================
# Map Legend
# =========================

legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 200px;
background-color: white;
border-radius: 8px;
z-index:9999;
font-size:14px;
padding:12px;
box-shadow: 0 0 15px rgba(0,0,0,0.2);
">

<b>Flood Risk Levels</b><br>

<span style="color:#8B0000;">⬤</span> Critical<br>
<span style="color:#FF3B30;">⬤</span> High<br>
<span style="color:#FFA500;">⬤</span> Moderate<br>
<span style="color:#2ECC71;">⬤</span> Low<br><br>

<b>Hotspot Severity</b><br>

<span style="color:red;">⬤</span> Critical<br>
<span style="color:orange;">⬤</span> High<br>
<span style="color:yellow;">⬤</span> Moderate

</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))


# =========================
# Layer Control
# =========================

folium.LayerControl().add_to(m)


# =========================
# Display Map
# =========================

st_folium(m, width=1100, height=650)