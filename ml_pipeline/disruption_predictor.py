# ml_pipeline/disruption_predictor.py

import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/opticalflow.duckdb"

# ──────────────────────────────────────────
# 1. LOAD TRAINING DATA FROM DUCKDB
# ──────────────────────────────────────────
def load_features(con):
    print("[1/6] Loading features from DuckDB...")
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

    print(f"    → {len(df)} supplier records loaded")
    return df


# ──────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────
def engineer_features(df):
    print("[2/6] Engineering features...")

    le_country  = LabelEncoder()
    le_category = LabelEncoder()
    le_risk     = LabelEncoder()

    df["country_enc"]   = le_country.fit_transform(df["country"])
    df["category_enc"]  = le_category.fit_transform(df["product_category"])
    df["risk_enc"]      = le_risk.fit_transform(df["risk_level"])
    df["is_active_int"] = df["is_active"].astype(int)

    # Target: high disruption = disruption_rate > 20% OR risk_level is HIGH/CRITICAL
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

    print(f"    → {len(feature_cols)} features ready | Target distribution:")
    print(f"      High risk: {df['target'].sum()} | Low risk: {(df['target']==0).sum()}")
    return df, feature_cols


# ──────────────────────────────────────────
# 3. TRAIN MODEL
# ──────────────────────────────────────────
def train_model(df, feature_cols):
    print("[3/6] Training Gradient Boosting model...")

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n    ── Model Evaluation ──")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"    ROC-AUC Score : {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"    Confusion Matrix:\n{cm}\n")

    return model


# ──────────────────────────────────────────
# 4. GENERATE PREDICTIONS
# ──────────────────────────────────────────
def generate_predictions(df, model, feature_cols):
    print("[4/6] Generating risk predictions for all suppliers...")

    X_all = df[feature_cols]
    df["risk_probability"]  = model.predict_proba(X_all)[:, 1].round(4)
    df["predicted_risk"]    = model.predict(X_all)

    df["risk_tier"] = pd.cut(
        df["risk_probability"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n    ── Top Feature Importances ──")
    for _, row in feature_importance.head(5).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"    {row['feature']:<25} {bar} {row['importance']:.4f}")

    return df


# ──────────────────────────────────────────
# 5. SAVE PREDICTIONS TO DUCKDB
# ──────────────────────────────────────────
def save_predictions(con, df):
    print("\n[5/6] Saving predictions to DuckDB...")

    output_df = df[[
        "supplier_id",
        "supplier_name",
        "country",
        "product_category",
        "risk_level",
        "reliability_score",
        "lead_time_days",
        "total_shipments",
        "disrupted_shipments",
        "avg_delay_days",
        "disruption_rate_pct",
        "risk_probability",
        "predicted_risk",
        "risk_tier",
    ]].copy()

    output_df["predicted_at"] = pd.Timestamp.utcnow()

    con.execute("CREATE SCHEMA IF NOT EXISTS predictions;")
    con.execute("DROP TABLE IF EXISTS predictions.supplier_risk_scores;")
    con.execute("""
        CREATE TABLE predictions.supplier_risk_scores AS
        SELECT * FROM output_df
    """)

    count = con.execute("SELECT COUNT(*) FROM predictions.supplier_risk_scores").fetchone()[0]
    print(f"    → {count} predictions saved to predictions.supplier_risk_scores")


# ──────────────────────────────────────────
# 6. SUMMARY REPORT
# ──────────────────────────────────────────
def print_summary(con):
    print("\n[6/6] Risk Summary Report:")
    print("─" * 45)
    summary = con.execute("""
        SELECT
            risk_tier,
            COUNT(*)                        AS supplier_count,
            ROUND(AVG(risk_probability), 3) AS avg_risk_prob,
            ROUND(AVG(disruption_rate_pct), 1) AS avg_disruption_rate
        FROM predictions.supplier_risk_scores
        GROUP BY risk_tier
        ORDER BY avg_risk_prob DESC
    """).df()
    print(summary.to_string(index=False))
    print("─" * 45)

    print("\n🚨 Top 5 Highest Risk Suppliers:")
    top5 = con.execute("""
        SELECT supplier_id, supplier_name, country,
               risk_tier, risk_probability, disruption_rate_pct
        FROM predictions.supplier_risk_scores
        ORDER BY risk_probability DESC
        LIMIT 5
    """).df()
    print(top5.to_string(index=False))


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 OpticalFlow — Disruption Risk ML Pipeline\n")
    print("=" * 50)

    con = duckdb.connect(DB_PATH)

    df                      = load_features(con)
    df, feature_cols        = engineer_features(df)
    model                   = train_model(df, feature_cols)
    df                      = generate_predictions(df, model, feature_cols)
    save_predictions(con, df)
    print_summary(con)

    con.close()
    print("\n✅ ML pipeline complete.\n")