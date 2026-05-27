# ml_pipeline/demand_forecaster.py

import duckdb
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

DB_PATH      = "data/opticalflow.duckdb"
FORECAST_DAYS = 90  # Forecast 90 days ahead

# ──────────────────────────────────────────
# 1. LOAD DEMAND HISTORY
# ──────────────────────────────────────────
def load_demand_history(con):
    print("[1/4] Loading demand history...")
    df = con.execute("""
        SELECT
            date,
            sku,
            warehouse,
            SUM(demand) AS demand
        FROM raw.demand_history
        GROUP BY date, sku, warehouse
        ORDER BY sku, warehouse, date
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    print(f"      {len(df):,} records loaded across "
          f"{df['sku'].nunique()} SKUs and "
          f"{df['warehouse'].nunique()} warehouses")
    return df


# ──────────────────────────────────────────
# 2. FORECAST PER SKU + WAREHOUSE
# ──────────────────────────────────────────
def run_forecasts(df):
    print(f"[2/4] Running Prophet forecasts ({FORECAST_DAYS} days ahead)...")

    all_forecasts = []
    combinations = df.groupby(["sku", "warehouse"])
    total        = len(combinations)

    for i, ((sku, warehouse), group) in enumerate(combinations, 1):
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = group[["date", "demand"]].rename(
            columns={"date": "ds", "demand": "y"}
        )

        # Train Prophet model
        model = Prophet(
            yearly_seasonality  = True,
            weekly_seasonality  = True,
            daily_seasonality   = False,
            seasonality_mode    = "multiplicative",
            interval_width      = 0.95,
        )
        model.fit(prophet_df)

        # Generate future dates
        future   = model.make_future_dataframe(periods=FORECAST_DAYS)
        forecast = model.predict(future)

        # Keep only future predictions
        future_forecast = forecast[forecast["ds"] > prophet_df["ds"].max()][[
            "ds", "yhat", "yhat_lower", "yhat_upper"
        ]].copy()

        future_forecast["sku"]       = sku
        future_forecast["warehouse"] = warehouse
        future_forecast["yhat"]      = future_forecast["yhat"].clip(lower=0).round(1)
        future_forecast["yhat_lower"]= future_forecast["yhat_lower"].clip(lower=0).round(1)
        future_forecast["yhat_upper"]= future_forecast["yhat_upper"].clip(lower=0).round(1)

        all_forecasts.append(future_forecast)

        if i % 10 == 0 or i == total:
            print(f"      Forecasted {i}/{total} SKU-warehouse combinations")

    return pd.concat(all_forecasts, ignore_index=True)


# ──────────────────────────────────────────
# 3. CALCULATE STOCKOUT RISK
# ──────────────────────────────────────────
def calculate_stockout_risk(forecasts_df, con):
    print("[3/4] Calculating stockout risk...")

    # Get current inventory levels
    inventory = con.execute("""
        SELECT
            sku,
            warehouse,
            SUM(quantity_on_hand) AS current_stock,
            AVG(reorder_point)    AS reorder_point
        FROM transformed_staging.stg_inventory
        GROUP BY sku, warehouse
    """).df()

    # Aggregate total forecasted demand over next 90 days
    demand_summary = forecasts_df.groupby(["sku", "warehouse"]).agg(
        forecasted_demand_90d = ("yhat", "sum"),
        peak_daily_demand     = ("yhat", "max"),
        avg_daily_demand      = ("yhat", "mean"),
    ).reset_index()

    # Join with current inventory
    risk_df = demand_summary.merge(inventory, on=["sku", "warehouse"], how="left")
    risk_df["current_stock"]  = risk_df["current_stock"].fillna(0)
    risk_df["reorder_point"]  = risk_df["reorder_point"].fillna(100)

    # Days until stockout = current stock / avg daily demand
    risk_df["days_until_stockout"] = (
        risk_df["current_stock"] / risk_df["avg_daily_demand"].replace(0, 1)
    ).round(1)

    # Stockout risk classification
    risk_df["stockout_risk"] = pd.cut(
        risk_df["days_until_stockout"],
        bins        = [0, 7, 14, 30, float("inf")],
        labels      = ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        include_lowest = True
    )

    risk_df["will_stockout_90d"] = risk_df["days_until_stockout"] < 90

    return risk_df


# ──────────────────────────────────────────
# 4. SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_results(con, forecasts_df, risk_df):
    print("[4/4] Saving forecasts to DuckDB...")

    con.execute("CREATE SCHEMA IF NOT EXISTS predictions;")

    # Save raw forecasts
    con.execute("DROP TABLE IF EXISTS predictions.demand_forecasts;")
    con.execute("""
        CREATE TABLE predictions.demand_forecasts AS
        SELECT * FROM forecasts_df
    """)

    # Save stockout risk
    con.execute("DROP TABLE IF EXISTS predictions.stockout_risk;")
    con.execute("""
        CREATE TABLE predictions.stockout_risk AS
        SELECT * FROM risk_df
    """)

    fc_count = con.execute(
        "SELECT COUNT(*) FROM predictions.demand_forecasts"
    ).fetchone()[0]
    sr_count = con.execute(
        "SELECT COUNT(*) FROM predictions.stockout_risk"
    ).fetchone()[0]

    print(f"      {fc_count:,} forecast rows saved")
    print(f"      {sr_count:,} stockout risk rows saved")


# ──────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────
def print_summary(con):
    print("\nStockout Risk Summary:")
    print("─" * 50)

    summary = con.execute("""
        SELECT
            stockout_risk,
            COUNT(*)                             AS combinations,
            ROUND(AVG(days_until_stockout), 1)   AS avg_days_to_stockout,
            SUM(CASE WHEN will_stockout_90d
                THEN 1 ELSE 0 END)               AS will_stockout
        FROM predictions.stockout_risk
        GROUP BY stockout_risk
        ORDER BY avg_days_to_stockout
    """).df()
    print(summary.to_string(index=False))

    print("\nTop 5 Most Critical SKU-Warehouse Combinations:")
    print("─" * 50)
    critical = con.execute("""
        SELECT sku, warehouse, current_stock,
               ROUND(avg_daily_demand, 1) AS avg_daily_demand,
               days_until_stockout, stockout_risk
        FROM predictions.stockout_risk
        ORDER BY days_until_stockout ASC
        LIMIT 5
    """).df()
    print(critical.to_string(index=False))


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - Prophet Demand Forecasting\n")
    print("=" * 50)

    con = duckdb.connect(DB_PATH)

    df           = load_demand_history(con)
    forecasts_df = run_forecasts(df)
    risk_df      = calculate_stockout_risk(forecasts_df, con)
    save_results(con, forecasts_df, risk_df)
    print_summary(con)

    con.close()
    print("\nForecasting complete.\n")