# ingestion/fetch_weather_data.py

import requests
import pandas as pd
import duckdb
import time
from datetime import datetime, timedelta

DB_PATH = "data/opticalflow.duckdb"

# ──────────────────────────────────────────
# SUPPLIER PORT CITIES
# Mapped to main export port per country
# ──────────────────────────────────────────
SUPPLIER_PORT_CITIES = {
    "China"       : {"city": "Shanghai",   "lat": 31.2304,  "lon": 121.4737},
    "Vietnam"     : {"city": "Ho Chi Minh","lat": 10.8231,  "lon": 106.6297},
    "Italy"       : {"city": "Genoa",      "lat": 44.4056,  "lon": 8.9463 },
    "Germany"     : {"city": "Hamburg",    "lat": 53.5753,  "lon": 10.0153},
    "Kenya"       : {"city": "Mombasa",    "lat": -4.0435,  "lon": 39.6682},
    "India"       : {"city": "Mumbai",     "lat": 19.0760,  "lon": 72.8777},
    "USA"         : {"city": "Los Angeles","lat": 34.0522,  "lon": -118.2437},
    "France"      : {"city": "Marseille",  "lat": 43.2965,  "lon": 5.3698 },
    "Japan"       : {"city": "Osaka",      "lat": 34.6937,  "lon": 135.5023},
    "South Korea" : {"city": "Busan",      "lat": 35.1796,  "lon": 129.0756},
}

# ──────────────────────────────────────────
# WEATHER SEVERITY THRESHOLDS
# ──────────────────────────────────────────
THRESHOLDS = {
    "heavy_rain_mm"        : 20,   # mm/day
    "strong_wind_kmh"      : 50,   # km/h
    "extreme_temp_high_c"  : 38,   # celsius
    "extreme_temp_low_c"   : -5,   # celsius
    "heavy_snow_cm"        : 5,    # cm/day
}


# ──────────────────────────────────────────
# 1. FETCH WEATHER DATA PER CITY
# ──────────────────────────────────────────
def fetch_weather_for_city(country, city_info, days_back=30):
    """
    Fetches historical daily weather from Open-Meteo API.
    Free, no API key required.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"              : city_info["lat"],
        "longitude"             : city_info["lon"],
        "daily"                 : [
            "precipitation_sum",
            "windspeed_10m_max",
            "temperature_2m_max",
            "temperature_2m_min",
            "snowfall_sum",
            "weathercode",
        ],
        "timezone"              : "auto",
        "past_days"             : days_back,
        "forecast_days"         : 7,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        daily = data["daily"]
        df = pd.DataFrame({
            "date"              : daily["time"],
            "country"           : country,
            "city"              : city_info["city"],
            "precipitation_mm"  : daily["precipitation_sum"],
            "windspeed_kmh"     : daily["windspeed_10m_max"],
            "temp_max_c"        : daily["temperature_2m_max"],
            "temp_min_c"        : daily["temperature_2m_min"],
            "snowfall_cm"       : daily["snowfall_sum"],
            "weather_code"      : daily["weathercode"],
        })

        # Fill nulls
        df = df.fillna(0)
        return df

    except Exception as e:
        print(f"      Warning: Could not fetch weather for {city_info['city']}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────
# 2. CALCULATE WEATHER DISRUPTION SCORE
# ──────────────────────────────────────────
def calculate_disruption_score(df):
    """
    Scores each day 0-100 based on weather severity.
    Multiple severe conditions compound the score.
    """
    df = df.copy()

    # Individual condition flags
    df["heavy_rain"]    = (df["precipitation_mm"] >= THRESHOLDS["heavy_rain_mm"]).astype(int)
    df["strong_wind"]   = (df["windspeed_kmh"]    >= THRESHOLDS["strong_wind_kmh"]).astype(int)
    df["extreme_heat"]  = (df["temp_max_c"]        >= THRESHOLDS["extreme_temp_high_c"]).astype(int)
    df["extreme_cold"]  = (df["temp_min_c"]        <= THRESHOLDS["extreme_temp_low_c"]).astype(int)
    df["heavy_snow"]    = (df["snowfall_cm"]       >= THRESHOLDS["heavy_snow_cm"]).astype(int)

    # Severity score — weighted sum
    df["weather_disruption_score"] = (
        df["heavy_rain"]  * 30 +
        df["strong_wind"] * 25 +
        df["extreme_heat"]* 20 +
        df["extreme_cold"]* 20 +
        df["heavy_snow"]  * 25 +
        # Continuous component
        (df["precipitation_mm"] / 5).clip(upper=20) +
        (df["windspeed_kmh"] / 10).clip(upper=15)
    ).clip(upper=100).round(2)

    # Weather risk tier
    df["weather_risk_tier"] = pd.cut(
        df["weather_disruption_score"],
        bins        = [0, 10, 30, 60, 100],
        labels      = ["CLEAR", "WATCH", "WARNING", "SEVERE"],
        include_lowest = True
    )

    return df


# ──────────────────────────────────────────
# 3. BUILD COUNTRY WEATHER SUMMARY
# ──────────────────────────────────────────
def build_country_summary(all_weather_df):
    """
    Aggregates daily weather into a country-level
    disruption risk summary for the last 30 days.
    """
    summary = all_weather_df.groupby("country").agg(
        city                    = ("city", "first"),
        avg_disruption_score    = ("weather_disruption_score", "mean"),
        max_disruption_score    = ("weather_disruption_score", "max"),
        severe_days             = ("weather_risk_tier",
                                   lambda x: (x == "SEVERE").sum()),
        warning_days            = ("weather_risk_tier",
                                   lambda x: (x == "WARNING").sum()),
        avg_precipitation_mm    = ("precipitation_mm", "mean"),
        avg_windspeed_kmh       = ("windspeed_kmh", "mean"),
        days_measured           = ("date", "count"),
    ).reset_index()

    summary["avg_disruption_score"] = summary["avg_disruption_score"].round(2)
    summary["max_disruption_score"] = summary["max_disruption_score"].round(2)
    summary["avg_precipitation_mm"] = summary["avg_precipitation_mm"].round(2)
    summary["avg_windspeed_kmh"]    = summary["avg_windspeed_kmh"].round(2)

    # Overall country weather risk tier
    summary["country_weather_risk"] = pd.cut(
        summary["avg_disruption_score"],
        bins        = [0, 10, 25, 50, 100],
        labels      = ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest = True
    )

    summary["fetched_at"] = datetime.utcnow().isoformat()
    return summary


# ──────────────────────────────────────────
# 4. SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_to_duckdb(daily_df, summary_df):
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    con.execute("DROP TABLE IF EXISTS raw.weather_daily;")
    con.execute("CREATE TABLE raw.weather_daily AS SELECT * FROM daily_df")

    con.execute("DROP TABLE IF EXISTS raw.weather_country_summary;")
    con.execute("CREATE TABLE raw.weather_country_summary AS SELECT * FROM summary_df")

    print(f"  raw.weather_daily           → {len(daily_df)} rows")
    print(f"  raw.weather_country_summary → {len(summary_df)} rows")
    con.close()


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - Weather Risk Integration\n")
    print("=" * 50)

    all_frames = []

    print("[1/3] Fetching weather for all supplier port cities...")
    for country, city_info in SUPPLIER_PORT_CITIES.items():
        print(f"      Fetching {city_info['city']}, {country}...")
        df = fetch_weather_for_city(country, city_info, days_back=30)
        if not df.empty:
            df = calculate_disruption_score(df)
            all_frames.append(df)
        time.sleep(0.5)   # Be respectful to the free API

    all_weather_df = pd.concat(all_frames, ignore_index=True)
    print(f"\n[2/3] Processed {len(all_weather_df)} daily weather records")

    print("\n[3/3] Building country weather risk summary...")
    summary_df = build_country_summary(all_weather_df)

    print("\nCountry Weather Risk Summary (last 30 days):")
    print("─" * 60)
    print(summary_df[[
        "country", "city", "avg_disruption_score",
        "severe_days", "warning_days", "country_weather_risk"
    ]].sort_values("avg_disruption_score", ascending=False).to_string(index=False))

    print("\nSaving to DuckDB...")
    save_to_duckdb(all_weather_df, summary_df)

    print("\nWeather data integration complete.\n")