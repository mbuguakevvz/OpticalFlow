# ml_pipeline/risk_explainer.py

import duckdb
import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

DB_PATH = "data/opticalflow.duckdb"


# ──────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ──────────────────────────────────────────
def load_and_prepare(con):
    print("[1/5] Loading supplier data...")
    df = con.execute("""
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.country,
            s.product_category,
            s.risk_level,
            s.reliability_score,
            s.lead_time_days,
            s.is_active,
            s.annual_spend_usd,
            COALESCE(sh.total_shipments, 0)       AS total_shipments,
            COALESCE(sh.disrupted_shipments, 0)   AS disrupted_shipments,
            COALESCE(sh.avg_delay_days, 0)        AS avg_delay_days,
            COALESCE(sh.max_delay_days, 0)        AS max_delay_days,
            COALESCE(sh.disruption_rate_pct, 0)   AS disruption_rate_pct
        FROM transformed_staging.stg_suppliers s
        LEFT JOIN (
            SELECT
                supplier_id,
                COUNT(*)                                        AS total_shipments,
                SUM(CASE WHEN is_disrupted THEN 1 ELSE 0 END)  AS disrupted_shipments,
                AVG(delay_days)                                 AS avg_delay_days,
                MAX(delay_days)                                 AS max_delay_days,
                ROUND(
                    SUM(CASE WHEN is_disrupted THEN 1 ELSE 0 END) * 100.0
                    / NULLIF(COUNT(*), 0), 2
                )                                               AS disruption_rate_pct
            FROM transformed_staging.stg_shipments
            GROUP BY supplier_id
        ) sh ON s.supplier_id = sh.supplier_id
    """).df()

    print(f"      {len(df)} suppliers loaded")
    return df


# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────
def engineer_features(df):
    print("[2/5] Engineering features...")

    le_country  = LabelEncoder()
    le_category = LabelEncoder()
    le_risk     = LabelEncoder()

    df["country_enc"]   = le_country.fit_transform(df["country"])
    df["category_enc"]  = le_category.fit_transform(df["product_category"])
    df["risk_enc"]      = le_risk.fit_transform(df["risk_level"])
    df["is_active_int"] = df["is_active"].astype(int)

    df["target"] = (
        (df["disruption_rate_pct"] > 20) |
        (df["risk_level"].isin(["HIGH", "CRITICAL"]))
    ).astype(int)

    feature_cols = [
        "reliability_score",
        "lead_time_days",
        "is_active_int",
        "annual_spend_usd",
        "total_shipments",
        "disrupted_shipments",
        "avg_delay_days",
        "max_delay_days",
        "disruption_rate_pct",
        "country_enc",
        "category_enc",
        "risk_enc",
    ]

    # Human-readable feature names for SHAP plots
    feature_labels = {
        "reliability_score"   : "Reliability Score",
        "lead_time_days"      : "Lead Time (Days)",
        "is_active_int"       : "Is Active",
        "annual_spend_usd"    : "Annual Spend (USD)",
        "total_shipments"     : "Total Shipments",
        "disrupted_shipments" : "Disrupted Shipments",
        "avg_delay_days"      : "Avg Delay (Days)",
        "max_delay_days"      : "Max Delay (Days)",
        "disruption_rate_pct" : "Disruption Rate %",
        "country_enc"         : "Country",
        "category_enc"        : "Product Category",
        "risk_enc"            : "Risk Level",
    }

    return df, feature_cols, feature_labels


# ──────────────────────────────────────────
# 3. TRAIN MODEL
# ──────────────────────────────────────────
def train_model(df, feature_cols):
    print("[3/5] Training model...")

    X = df[feature_cols]
    y = df["target"]

    model = GradientBoostingClassifier(
        n_estimators  = 100,
        learning_rate = 0.1,
        max_depth     = 4,
        random_state  = 42,
    )
    model.fit(X, y)
    print(f"      Model trained on {len(X)} suppliers")
    return model


# ──────────────────────────────────────────
# 4. COMPUTE SHAP VALUES
# ──────────────────────────────────────────
def compute_shap(df, model, feature_cols, feature_labels):
    print("[4/5] Computing SHAP values...")

    X = df[feature_cols]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Build SHAP dataframe — one row per supplier
    shap_df = pd.DataFrame(
        shap_values,
        columns=[feature_labels[c] for c in feature_cols]
    )
    shap_df.insert(0, "supplier_id",   df["supplier_id"].values)
    shap_df.insert(1, "supplier_name", df["supplier_name"].values)
    shap_df.insert(2, "risk_tier",     pd.cut(
        model.predict_proba(X)[:, 1],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    ))
    shap_df.insert(3, "risk_probability",
        model.predict_proba(X)[:, 1].round(4)
    )

    # Top driving factor per supplier
    factor_cols = [feature_labels[c] for c in feature_cols]
    shap_df["top_risk_driver"] = shap_df[factor_cols].abs().idxmax(axis=1)
    shap_df["top_driver_impact"] = shap_df.apply(
        lambda row: round(row[row["top_risk_driver"]], 4), axis=1
    )

    # Build human-readable explanation per supplier
    def build_explanation(row):
        impacts = {col: row[col] for col in factor_cols}
        sorted_factors = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_factors[:3]
        parts = []
        for factor, impact in top3:
            direction = "increases" if impact > 0 else "reduces"
            parts.append(f"{factor} {direction} risk by {abs(impact):.3f}")
        return " | ".join(parts)

    shap_df["explanation"] = shap_df.apply(build_explanation, axis=1)

    print(f"      SHAP values computed for {len(shap_df)} suppliers")
    return shap_df, shap_values, X, feature_labels


# ──────────────────────────────────────────
# 5. SAVE TO DUCKDB
# ──────────────────────────────────────────
def save_results(con, shap_df):
    print("[5/5] Saving SHAP explanations to DuckDB...")

    con.execute("CREATE SCHEMA IF NOT EXISTS predictions;")
    con.execute("DROP TABLE IF EXISTS predictions.supplier_shap_explanations;")
    con.execute("""
        CREATE TABLE predictions.supplier_shap_explanations AS
        SELECT * FROM shap_df
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM predictions.supplier_shap_explanations"
    ).fetchone()[0]
    print(f"      {count} supplier explanations saved")


# ──────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────
def print_summary(con):
    print("\nGlobal Feature Importance (mean |SHAP|):")
    print("─" * 50)

    # Get mean absolute SHAP per feature
    shap_data = con.execute("""
        SELECT * FROM predictions.supplier_shap_explanations
    """).df()

    factor_cols = [
        "Reliability Score", "Lead Time (Days)", "Is Active",
        "Annual Spend (USD)", "Total Shipments", "Disrupted Shipments",
        "Avg Delay (Days)", "Max Delay (Days)", "Disruption Rate %",
        "Country", "Product Category", "Risk Level"
    ]

    importance = shap_data[factor_cols].abs().mean().sort_values(ascending=False)
    for feature, value in importance.items():
        bar = "█" * int(value * 200)
        print(f"  {feature:<25} {bar} {value:.4f}")

    print("\nTop Risk Driver Distribution:")
    print("─" * 50)
    drivers = shap_data["top_risk_driver"].value_counts().reset_index()
    drivers.columns = ["Driver", "Count"]
    print(drivers.to_string(index=False))

    print("\nSample Explanations (Top 5 Critical Suppliers):")
    print("─" * 50)
    critical = con.execute("""
        SELECT supplier_id, supplier_name, risk_tier,
               risk_probability, top_risk_driver, explanation
        FROM predictions.supplier_shap_explanations
        WHERE risk_tier = 'CRITICAL'
        ORDER BY risk_probability DESC
        LIMIT 5
    """).df()
    for _, row in critical.iterrows():
        print(f"\n  {row['supplier_id']} — {row['supplier_name']}")
        print(f"  Risk: {row['risk_tier']} ({row['risk_probability']})")
        print(f"  Top driver: {row['top_risk_driver']}")
        print(f"  Why: {row['explanation']}")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nOpticalFlow - SHAP Risk Explainability\n")
    print("=" * 50)

    con = duckdb.connect(DB_PATH)

    df                              = load_and_prepare(con)
    df, feature_cols, feature_labels = engineer_features(df)
    model                           = train_model(df, feature_cols)
    shap_df, shap_values, X, labels = compute_shap(
                                          df, model, feature_cols, feature_labels
                                      )
    save_results(con, shap_df)
    print_summary(con)

    con.close()
    print("\nSHAP explainability complete.\n")