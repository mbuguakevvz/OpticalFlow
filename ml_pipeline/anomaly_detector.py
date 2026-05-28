# ml_pipeline/anomaly_detector.py

import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DB_PATH             = "data/opticalflow.duckdb"
CONTAMINATION_RATE  = 0.05  # Expect ~5% of shipments to be anomalous


# ──────────────────────────────────────────
# 1. LOAD SHIPMENT DATA
# ──────────────────────────────────────────
def load_shipments(con):
    print("[1/5] Loading shipment data...")
    df = con.execute("""
        SELECT
            s.shipment_id,
            s.supplier_id,
            s.sku,
            s.origin_country,
            s.destination_warehouse,
            s.quantity_shipped,
            s.delay_days,
            s.freight_cost_usd,
            s.carrier,
            s.is_disrupted,
            s.status,
            sup.reliability_score,
            sup.lead_time_days        AS supplier_lead_time,
            sup.risk_level            AS supplier_risk_level,
            sup.annual_spend_usd
        FROM transformed_staging.stg_shipments s
        LEFT JOIN transformed_staging.stg_suppliers sup
            ON s.supplier_id = sup.supplier_id
    """).df()
    print(f"      {len(df):,} shipments loaded")
    return df


# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────
def engineer_features(df):
    print("[2/5] Engineering anomaly detection features...")

    le_country   = LabelEncoder()
    le_warehouse = LabelEncoder()
    le_carrier   = LabelEncoder()
    le_sku       = LabelEncoder()
    le_risk      = LabelEncoder()

    df["country_enc"]    = le_country.fit_transform(df["origin_country"])
    df["warehouse_enc"]  = le_warehouse.fit_transform(df["destination_warehouse"])
    df["carrier_enc"]    = le_carrier.fit_transform(df["carrier"])
    df["sku_enc"]        = le_sku.fit_transform(df["sku"])
    df["risk_enc"]       = le_risk.fit_transform(df["supplier_risk_level"])
    df["is_disrupted_int"] = df["is_disrupted"].astype(int)

    # Cost per unit shipped — unusually high = anomaly signal
    df["cost_per_unit"] = (
        df["freight_cost_usd"] / df["quantity_shipped"].replace(0, 1)
    ).round(4)

    # Delay ratio vs supplier lead time
    df["delay_ratio"] = (
        df["delay_days"] / df["supplier_lead_time"].replace(0, 1)
    ).round(4)

    feature_cols = [
        "delay_days",
        "freight_cost_usd",
        "quantity_shipped",
        "reliability_score",
        "supplier_lead_time",
        "cost_per_unit",
        "delay_ratio",
        "is_disrupted_int",
        "country_enc",
        "warehouse_enc",
        "carrier_enc",
        "sku_enc",
        "risk_enc",
    ]

    print(f"      {len(feature_cols)} features engineered")
    return df, feature_cols


# ──────────────────────────────────────────
# 3. RUN ISOLATION FOREST
# ──────────────────────────────────────────
def detect_anomalies(df, feature_cols):
    print("[3/5] Running Isolation Forest anomaly detection...")

    X = df[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators  = 200,
        contamination = CONTAMINATION_RATE,
        random_state  = 42,
        n_jobs        = -1,
    )
    model.fit(X)

    # -1 = anomaly, 1 = normal
    df["anomaly_label"]  = model.predict(X)
    df["anomaly_score"]  = model.decision_function(X).round(6)

    # Convert to readable format
    df["is_anomaly"]     = df["anomaly_label"] == -1

    # Normalize anomaly score to 0-1 (higher = more anomalous)
    score_min = df["anomaly_score"].min()
    score_max = df["anomaly_score"].max()
    df["anomaly_severity"] = (
        1 - (df["anomaly_score"] - score_min) / (score_max - score_min)
    ).round(4)

    # Severity tier
    df["severity_tier"] = pd.cut(
        df["anomaly_severity"],
        bins        = [0, 0.4, 0.6, 0.8, 1.0],
        labels      = ["NORMAL", "WATCH", "WARNING", "CRITICAL"],
        include_lowest = True
    )

    anomaly_count = df["is_anomaly"].sum()
    print(f"      {anomaly_count} anomalies detected "
          f"({anomaly_count/len(df)*100:.1f}% of shipments)")
    return df, model


# ──────────────────────────────────────────
# 4. IDENTIFY ANOMALY REASONS
# ──────────────────────────────────────────
def identify_anomaly_reasons(df):
    print("[4/5] Identifying anomaly patterns...")

    reasons = []
    for _, row in df[df["is_anomaly"]].iterrows():
        flags = []

        if row["delay_days"] > 20:
            flags.append("SEVERE_DELAY")
        if row["cost_per_unit"] > df["cost_per_unit"].quantile(0.95):
            flags.append("HIGH_FREIGHT_COST")
        if row["quantity_shipped"] > df["quantity_shipped"].quantile(0.97):
            flags.append("UNUSUALLY_LARGE_SHIPMENT")
        if row["delay_ratio"] > 2:
            flags.append("DELAY_EXCEEDS_LEAD_TIME_2X")
        if row["reliability_score"] < 0.6 and row["is_disrupted"]:
            flags.append("LOW_RELIABILITY_DISRUPTION")
        if not flags:
            flags.append("MULTIVARIATE_PATTERN")

        reasons.append({
            "shipment_id"      : row["shipment_id"],
            "anomaly_reasons"  : " | ".join(flags),
        })

    reasons_df = pd.DataFrame(reasons)
    df = df.merge(reasons_df, on="shipment_id", how="left")
    df["anomaly_reasons"] = df["anomaly_reasons"].fillna("NORMAL")
    return df


# ──────────────────────────────────────────
# 5. SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_results(con, df):
    print("[5/5] Saving anomaly results to DuckDB...")

    output_df = df[[
        "shipment_id",
        "supplier_id",
        "sku",
        "origin_country",
        "destination_warehouse",
        "carrier",
        "delay_days",
        "freight_cost_usd",
        "cost_per_unit",
        "delay_ratio",
        "status",
        "is_disrupted",
        "is_anomaly",
        "anomaly_score",
        "anomaly_severity",
        "severity_tier",
        "anomaly_reasons",
    ]].copy()

    con.execute("CREATE SCHEMA IF NOT EXISTS predictions;")
    con.execute("DROP TABLE IF EXISTS predictions.shipment_anomalies;")
    con.execute("""
        CREATE TABLE predictions.shipment_anomalies AS
        SELECT * FROM output_df
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM predictions.shipment_anomalies"
    ).fetchone()[0]
    print(f"      {count:,} rows saved to predictions.shipment_anomalies")


# ──────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────
def print_summary(con):
    print("\nAnomaly Detection Summary:")
    print("─" * 55)

    summary = con.execute("""
        SELECT
            severity_tier,
            COUNT(*)                            AS shipment_count,
            ROUND(AVG(delay_days), 1)           AS avg_delay_days,
            ROUND(AVG(freight_cost_usd), 0)     AS avg_freight_cost
        FROM predictions.shipment_anomalies
        GROUP BY severity_tier
        ORDER BY avg_delay_days DESC
    """).df()
    print(summary.to_string(index=False))

    print("\nTop 10 Most Anomalous Shipments:")
    print("─" * 55)
    top = con.execute("""
        SELECT
            shipment_id,
            supplier_id,
            origin_country,
            carrier,
            delay_days,
            severity_tier,
            anomaly_reasons
        FROM predictions.shipment_anomalies
        WHERE is_anomaly = true
        ORDER BY anomaly_severity DESC
        LIMIT 10
    """).df()
    print(top.to_string(index=False))

    print("\nTop Anomaly Reason Breakdown:")
    print("─" * 55)
    reasons = con.execute("""
        SELECT
            anomaly_reasons,
            COUNT(*) AS count
        FROM predictions.shipment_anomalies
        WHERE is_anomaly = true
        GROUP BY anomaly_reasons
        ORDER BY count DESC
    """).df()
    print(reasons.to_string(index=False))


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - Shipment Anomaly Detection\n")
    print("=" * 55)

    con = duckdb.connect(DB_PATH)

    df              = load_shipments(con)
    df, feature_cols = engineer_features(df)
    df, model       = detect_anomalies(df, feature_cols)
    df              = identify_anomaly_reasons(df)
    save_results(con, df)
    print_summary(con)

    con.close()
    print("\nAnomaly detection complete.\n")