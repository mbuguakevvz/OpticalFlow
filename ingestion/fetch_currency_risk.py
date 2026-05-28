# ingestion/fetch_currency_risk.py

import requests
import pandas as pd
import duckdb
import time
from datetime import datetime, timedelta

DB_PATH = "data/opticalflow.duckdb"

# ──────────────────────────────────────────
# SUPPLIER COUNTRY CURRENCIES
# ──────────────────────────────────────────
SUPPLIER_CURRENCIES = {
    "China"       : "CNY",
    "Vietnam"     : "VND",
    "Italy"       : "EUR",
    "Germany"     : "EUR",
    "Kenya"       : "KES",
    "India"       : "INR",
    "USA"         : "USD",
    "France"      : "EUR",
    "Japan"       : "JPY",
    "South Korea" : "KRW",
}

BASE_CURRENCY = "USD"


# ──────────────────────────────────────────
# 1. FETCH CURRENT RATES
# ──────────────────────────────────────────
def fetch_current_rates():
    """
    Fetches current exchange rates vs USD
    from Frankfurter API (free, no key needed).
    """
    print("[1/4] Fetching current exchange rates...")

    url = f"https://api.frankfurter.app/latest?from={BASE_CURRENCY}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        rates = []
        currencies = set(SUPPLIER_CURRENCIES.values()) - {"USD"}

        for currency, country in [
            (v, k) for k, v in SUPPLIER_CURRENCIES.items()
        ]:
            if currency == "USD":
                rate = 1.0
            elif currency in data["rates"]:
                rate = data["rates"][currency]
            else:
                rate = None

            rates.append({
                "country"          : country,
                "currency_code"    : currency,
                "rate_vs_usd"      : rate,
                "date"             : data["date"],
            })

        df = pd.DataFrame(rates).dropna(subset=["rate_vs_usd"])
        print(f"      {len(df)} currency rates fetched (base: USD)")
        return df

    except Exception as e:
        print(f"      Frankfurter API error: {e}")
        return get_fallback_rates()


def get_fallback_rates():
    """Fallback rates if API is unavailable."""
    print("      Using fallback exchange rates...")
    data = {
        "country"       : list(SUPPLIER_CURRENCIES.keys()),
        "currency_code" : list(SUPPLIER_CURRENCIES.values()),
        "rate_vs_usd"   : [7.24, 24485, 0.92, 0.92, 129.50,
                            83.12, 1.0, 0.92, 149.50, 1325.0],
        "date"          : [datetime.today().strftime("%Y-%m-%d")] * 10,
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────
# 2. FETCH 30-DAY HISTORICAL RATES
# ──────────────────────────────────────────
def fetch_historical_rates():
    """
    Fetches 30-day historical rates to calculate volatility.
    """
    print("[2/4] Fetching 30-day historical rates for volatility...")

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # Get unique currencies excluding USD
    currencies = list(set(SUPPLIER_CURRENCIES.values()) - {"USD"})
    symbols    = ",".join(currencies)

    url = (
        f"https://api.frankfurter.app/{start_str}..{end_str}"
        f"?from={BASE_CURRENCY}&to={symbols}"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        records = []
        for date, rates in data["rates"].items():
            for currency, rate in rates.items():
                records.append({
                    "date"          : date,
                    "currency_code" : currency,
                    "rate_vs_usd"   : rate,
                })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        print(f"      {len(df)} historical rate records fetched")
        return df

    except Exception as e:
        print(f"      Historical rates error: {e}")
        return pd.DataFrame(columns=["date", "currency_code", "rate_vs_usd"])


# ──────────────────────────────────────────
# 3. CALCULATE VOLATILITY SCORES
# ──────────────────────────────────────────
def calculate_volatility(current_df, historical_df):
    """
    Calculates currency volatility as:
    - 30-day % change (trend)
    - Standard deviation of daily returns (volatility)
    - Combined currency risk score
    """
    print("[3/4] Calculating currency volatility scores...")

    volatility_records = []

    for _, row in current_df.iterrows():
        currency = row["currency_code"]
        country  = row["country"]

        if currency == "USD" or historical_df.empty:
            volatility_records.append({
                "country"              : country,
                "currency_code"        : currency,
                "current_rate_vs_usd"  : row["rate_vs_usd"],
                "rate_30d_ago"         : row["rate_vs_usd"],
                "pct_change_30d"       : 0.0,
                "daily_volatility"     : 0.0,
                "currency_risk_score"  : 0.0,
            })
            continue

        hist = historical_df[
            historical_df["currency_code"] == currency
        ].sort_values("date")

        if len(hist) < 2:
            pct_change    = 0.0
            daily_vol     = 0.0
            rate_30d_ago  = row["rate_vs_usd"]
        else:
            rate_30d_ago  = hist.iloc[0]["rate_vs_usd"]
            rate_now      = hist.iloc[-1]["rate_vs_usd"]
            pct_change    = round(
                (rate_now - rate_30d_ago) / rate_30d_ago * 100, 4
            )
            daily_returns = hist["rate_vs_usd"].pct_change().dropna()
            daily_vol     = round(float(daily_returns.std() * 100), 4)

        # Currency risk score — combines volatility and trend magnitude
        risk_score = round(
            min(100, abs(pct_change) * 2 + daily_vol * 10), 2
        )

        volatility_records.append({
            "country"              : country,
            "currency_code"        : currency,
            "current_rate_vs_usd"  : row["rate_vs_usd"],
            "rate_30d_ago"         : rate_30d_ago,
            "pct_change_30d"       : pct_change,
            "daily_volatility_pct" : daily_vol,
            "currency_risk_score"  : risk_score,
        })

    vol_df = pd.DataFrame(volatility_records)

    # Currency risk tier
    vol_df["currency_risk_tier"] = pd.cut(
        vol_df["currency_risk_score"],
        bins        = [0, 5, 15, 30, 100],
        labels      = ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest = True
    )

    vol_df["fetched_at"] = datetime.utcnow().isoformat()
    return vol_df


# ──────────────────────────────────────────
# 4. SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_to_duckdb(current_df, historical_df, volatility_df):
    print("[4/4] Saving currency data to DuckDB...")

    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    con.execute("DROP TABLE IF EXISTS raw.currency_rates_current;")
    con.execute("""
        CREATE TABLE raw.currency_rates_current AS
        SELECT * FROM current_df
    """)

    if not historical_df.empty:
        historical_df["date"] = historical_df["date"].astype(str)
        con.execute("DROP TABLE IF EXISTS raw.currency_rates_historical;")
        con.execute("""
            CREATE TABLE raw.currency_rates_historical AS
            SELECT * FROM historical_df
        """)

    con.execute("DROP TABLE IF EXISTS raw.currency_volatility;")
    con.execute("""
        CREATE TABLE raw.currency_volatility AS
        SELECT * FROM volatility_df
    """)

    print(f"  raw.currency_rates_current    → {len(current_df)} rows")
    if not historical_df.empty:
        print(f"  raw.currency_rates_historical → {len(historical_df)} rows")
    print(f"  raw.currency_volatility       → {len(volatility_df)} rows")
    con.close()


# ──────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────
def print_summary(volatility_df):
    print("\nCurrency Risk Summary (supplier countries):")
    print("─" * 65)
    display = volatility_df[[
        "country", "currency_code", "current_rate_vs_usd",
        "pct_change_30d", "daily_volatility_pct",
        "currency_risk_score", "currency_risk_tier"
    ]].sort_values("currency_risk_score", ascending=False)
    print(display.to_string(index=False))

    print("\nHighest Currency Risk Countries:")
    print("─" * 65)
    high_risk = volatility_df[
        volatility_df["currency_risk_tier"].isin(["HIGH", "CRITICAL"])
    ]
    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            print(f"  {row['country']:<15} {row['currency_code']}  "
                  f"risk score: {row['currency_risk_score']}  "
                  f"30d change: {row['pct_change_30d']}%")
    else:
        print("  No HIGH or CRITICAL currency risk countries currently")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - Currency Exchange Rate Risk\n")
    print("=" * 55)

    current_df    = fetch_current_rates()
    time.sleep(1)
    historical_df = fetch_historical_rates()
    volatility_df = calculate_volatility(current_df, historical_df)

    print_summary(volatility_df)
    save_to_duckdb(current_df, historical_df, volatility_df)

    print("\nCurrency risk integration complete.\n")