# config.py — Central configuration for HydraGIS Flood Backend

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    APP_NAME: str = "HydraGIS Flood Prediction Engine"
    VERSION:  str = "1.0.0"
    DEBUG:    bool = True

    # ── Paths ─────────────────────────────────────────────────────────────
    DATA_DIR:    Path = BASE_DIR / "data"
    MODELS_DIR:  Path = BASE_DIR / "saved_models"

    # ── Kaggle dataset filenames (put these in /data after downloading) ──
    # Dataset #5 — Kaggle Playground S4E5 Flood Prediction
    KAGGLE_FLOOD_TRAIN: str = "flood_train.csv"        # ~35MB
    KAGGLE_FLOOD_TEST:  str = "flood_test.csv"

    # Dataset #2 — Daily Rainfall India 2009–2024
    KAGGLE_RAINFALL:    str = "rainfall_india.csv"     # ~85MB

    # Dataset #1 — Flood Risk in India
    KAGGLE_RISK_INDIA:  str = "flood_risk_india.csv"   # ~12MB

    # ── Model hyper-parameters ────────────────────────────────────────────
    RF_N_ESTIMATORS:  int   = 200
    RF_MAX_DEPTH:     int   = 12
    RF_RANDOM_STATE:  int   = 42

    XGB_N_ESTIMATORS: int   = 50
    XGB_RANDOM_STATE: int   = 42
    XGB_LEARNING_RATE: float = 0.05
    XGB_MAX_DEPTH:    int   = 6

    ENSEMBLE_RF_WEIGHT:  float = 0.45
    ENSEMBLE_XGB_WEIGHT: float = 0.55

    # ── Ward scoring weights (must sum to 1.0) ────────────────────────────
    W_DRAINAGE:   float = 0.30
    W_ELEVATION:  float = 0.25
    W_RAINFALL:   float = 0.20
    W_INFRA_AGE:  float = 0.15
    W_PUMP_CAP:   float = 0.10

    # ── OpenWeatherMap API (ADD 2 — Live rainfall integration) ───────────────
    # Register free key at: https://openweathermap.org/api
    # Set via env var: OWM_API_KEY=your_key  or in .env file
    OWM_API_KEY: str = ""

    # ── Historical rainfall snapshots for trend analysis (ADD 4) ─────────────
    # IMD Bengaluru Subdivision normals (1991-2020) and observed 2024 season
    # Source: IMD Pune, Bengaluru Sub-division seasonal normals
    RAINFALL_2020_ZONE: dict = {
        "north": 845.0,   # Yelahanka, Hebbal, RT Nagar
        "south": 920.0,   # Jayanagar, BTM, Banashankari
        "east":  970.0,   # Bellandur, Whitefield, KR Puram
        "west":  810.0,   # Rajajinagar, Malleshwaram, Yeshwanthpur
        "city":  886.0,   # City-wide average
    }
    RAINFALL_2024_ZONE: dict = {
        "north": 1020.0,  # 20.7% above normal (IMD ENSO-active 2024)
        "south": 1105.0,  # 20.1% above normal
        "east":  1195.0,  # 23.2% above normal — IT corridor worst hit
        "west":   985.0,  # 21.6% above normal
        "city":  1076.0,  # City-wide 2024 season actual
    }

    # ── Flood thresholds (mm) ─────────────────────────────────────────────
    FLOOD_THRESHOLD_MM: float = 300.0
    ALERT_THRESHOLD_MM: float = 200.0

    # ── Redis (caching + Celery broker) ───────────────────────────────────
    REDIS_URL:          str = "redis://localhost:6379"
    CACHE_TTL_SECONDS:  int = 3600          # 1 hour for ward_scores cache

    # ── Celery ────────────────────────────────────────────────────────────
    CELERY_BROKER_URL:  str = "redis://localhost:6379/0"
    CELERY_RESULT_URL:  str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Create directories on import
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)