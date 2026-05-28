# ingestion/fetch_real_data.py

import requests
import pandas as pd
import duckdb
import time
import os

DB_PATH = "data/opticalflow.duckdb"

# ──────────────────────────────────────────
# 1. WORLD BANK — Logistics Performance Index
# ──────────────────────────────────────────
def fetch_world_bank_lpi():
    """
    Fetches the World Bank Logistics Performance Index (LPI)
    for all countries. LPI measures:
    - Customs efficiency
    - Infrastructure quality
    - Ease of arranging shipments
    - Logistics competence
    - Tracking & tracing
    - Timeliness of shipments

    Score: 1 (worst) to 5 (best)
    API docs: https://data.worldbank.org/indicator/LP.LPI.OVRL.XQ
    """
    print("[1/3] Fetching World Bank Logistics Performance Index...")

    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        "LP.LPI.OVRL.XQ?format=json&mrv=1&per_page=300"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "country_code"  : item["country"]["id"],
                    "country_name"  : item["country"]["value"],
                    "lpi_score"     : round(float(item["value"]), 3),
                    "year"          : item["date"],
                })

        df = pd.DataFrame(records)

        # Normalize LPI to 0-1 for use as risk modifier
        df["lpi_normalized"] = (
            (df["lpi_score"] - df["lpi_score"].min()) /
            (df["lpi_score"].max() - df["lpi_score"].min())
        ).round(3)

        # Country logistics risk (inverse of LPI)
        df["logistics_risk_score"] = (1 - df["lpi_normalized"]).round(3)

        print(f"      {len(df)} countries fetched")
        print(f"      LPI range: {df['lpi_score'].min()} - {df['lpi_score'].max()}")
        return df

    except Exception as e:
        print(f"      World Bank API error: {e}")
        print("      Using fallback LPI data...")
        return get_fallback_lpi()


def get_fallback_lpi():
    """Fallback LPI data for key supplier countries if API fails."""
    data = {
        "country_name"       : ["China", "Germany", "Japan", "United States",
                                 "Italy", "France", "South Korea", "India",
                                 "Vietnam", "Kenya", "Nigeria", "Ghana",
                                 "South Africa", "Egypt", "United Arab Emirates"],
        "lpi_score"          : [3.65, 4.20, 4.03, 3.89, 3.72, 3.84, 3.77,
                                 3.18, 3.27, 2.81, 2.53, 2.66, 3.38, 2.82, 3.96],
        "country_code"       : ["CN", "DE", "JP", "US", "IT", "FR", "KR",
                                 "IN", "VN", "KE", "NG", "GH", "ZA", "EG", "AE"],
    }
    df = pd.DataFrame(data)
    df["year"] = "2023"
    df["lpi_normalized"] = (
        (df["lpi_score"] - df["lpi_score"].min()) /
        (df["lpi_score"].max() - df["lpi_score"].min())
    ).round(3)
    df["logistics_risk_score"] = (1 - df["lpi_normalized"]).round(3)
    print(f"      Fallback: {len(df)} key countries loaded")
    return df


# ──────────────────────────────────────────
# 2. WORLD BANK — GDP per capita
#    (proxy for market size / demand)
# ──────────────────────────────────────────
def fetch_world_bank_gdp():
    """
    Fetches GDP per capita for all countries.
    Used as a proxy for eyewear market demand size.
    """
    print("[2/3] Fetching World Bank GDP per capita...")

    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        "NY.GDP.PCAP.CD?format=json&mrv=1&per_page=300"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "country_code" : item["country"]["id"],
                    "country_name" : item["country"]["value"],
                    "gdp_per_capita_usd" : round(float(item["value"]), 2),
                    "year"         : item["date"],
                })

        df = pd.DataFrame(records)
        print(f"      {len(df)} countries fetched")
        return df

    except Exception as e:
        print(f"      GDP API error: {e}")
        return pd.DataFrame(columns=[
            "country_code", "country_name", "gdp_per_capita_usd", "year"
        ])


# ──────────────────────────────────────────
# 3. BUILD COUNTRY RISK PROFILE
# ──────────────────────────────────────────
def build_country_risk_profile(lpi_df, gdp_df):
    """
    Merges LPI and GDP data into a unified country risk profile
    used to enrich supplier risk scoring.
    """
    print("[3/3] Building country risk profiles...")

    # Map our supplier country names to World Bank names
    country_name_map = {
        "China"       : "China",
        "Vietnam"     : "Vietnam",
        "Italy"       : "Italy",
        "Germany"     : "Germany",
        "Kenya"       : "Kenya",
        "India"       : "India",
        "USA"         : "United States",
        "France"      : "France",
        "Japan"       : "Japan",
        "South Korea" : "Korea, Rep.",
    }

    # Filter LPI to our supplier countries
    our_countries = list(country_name_map.values())
    lpi_filtered  = lpi_df[lpi_df["country_name"].isin(our_countries)].copy()
    lpi_filtered["our_country_name"] = lpi_filtered["country_name"].map(
        {v: k for k, v in country_name_map.items()}
    )

    # Merge with GDP if available
    if not gdp_df.empty:
        gdp_filtered = gdp_df[gdp_df["country_name"].isin(our_countries)][[
            "country_name", "gdp_per_capita_usd"
        ]]
        profile = lpi_filtered.merge(gdp_filtered, on="country_name", how="left")
    else:
        profile = lpi_filtered.copy()
        profile["gdp_per_capita_usd"] = None

    # Assign logistics risk tier
    profile["logistics_risk_tier"] = pd.cut(
        profile["logistics_risk_score"],
        bins   = [0, 0.25, 0.5, 0.75, 1.0],
        labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    )

    print(f"      {len(profile)} country risk profiles built")
    print("\n      Country Risk Profiles:")
    print("      " + "─" * 60)
    display_cols = [
        "our_country_name", "lpi_score",
        "logistics_risk_score", "logistics_risk_tier"
    ]
    available = [c for c in display_cols if c in profile.columns]
    print(profile[available].sort_values(
        "logistics_risk_score", ascending=False
    ).to_string(index=False))

    return profile


# ──────────────────────────────────────────
# SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_to_duckdb(lpi_df, gdp_df, profile_df):
    print("\nSaving real data to DuckDB...")

    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    # LPI data
    con.execute("DROP TABLE IF EXISTS raw.country_lpi;")
    con.execute("CREATE TABLE raw.country_lpi AS SELECT * FROM lpi_df")
    print(f"  raw.country_lpi          → {len(lpi_df)} rows")

    # GDP data
    if not gdp_df.empty:
        con.execute("DROP TABLE IF EXISTS raw.country_gdp;")
        con.execute("CREATE TABLE raw.country_gdp AS SELECT * FROM gdp_df")
        print(f"  raw.country_gdp          → {len(gdp_df)} rows")

    # Country risk profiles
    con.execute("DROP TABLE IF EXISTS raw.country_risk_profiles;")
    con.execute("CREATE TABLE raw.country_risk_profiles AS SELECT * FROM profile_df")
    print(f"  raw.country_risk_profiles → {len(profile_df)} rows")

    con.close()


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - Real World Data Ingestion\n")
    print("=" * 50)

    lpi_df     = fetch_world_bank_lpi()
    time.sleep(1)
    gdp_df     = fetch_world_bank_gdp()
    profile_df = build_country_risk_profile(lpi_df, gdp_df)
    save_to_duckdb(lpi_df, gdp_df, profile_df)

    print("\nReal data ingestion complete.\n")