# dashboard/app.py

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = "data/opticalflow.duckdb"

st.set_page_config(
    page_title            = "OpticalFlow",
    page_icon             = "👁️",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)

    suppliers   = con.execute("SELECT * FROM transformed_staging.stg_suppliers").df()
    shipments   = con.execute("SELECT * FROM transformed_staging.stg_shipments").df()
    inventory   = con.execute("SELECT * FROM transformed_marts.mart_inventory_health").df()
    risk_scores = con.execute("SELECT * FROM predictions.supplier_risk_scores ORDER BY risk_probability DESC").df()

    # New layers — safe load with fallback
    def safe_load(query):
        try:
            return con.execute(query).df()
        except Exception:
            return pd.DataFrame()

    stockout    = safe_load("SELECT * FROM predictions.stockout_risk")
    anomalies   = safe_load("SELECT * FROM predictions.shipment_anomalies")
    shap_df     = safe_load("SELECT * FROM predictions.supplier_shap_explanations")
    lpi_df      = safe_load("SELECT * FROM raw.country_risk_profiles")
    weather_df  = safe_load("SELECT * FROM raw.weather_country_summary")
    currency_df = safe_load("SELECT * FROM raw.currency_volatility")

    con.close()
    return (suppliers, shipments, inventory, risk_scores,
            stockout, anomalies, shap_df, lpi_df, weather_df, currency_df)


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/glasses.png", width=60)
    st.sidebar.title("👁️ OpticalFlow")
    st.sidebar.caption("AI-Driven Supply Chain Resilience")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigate", [
        "🏠 Overview",
        "⚠️ Supplier Risk",
        "🚢 Shipment Monitor",
        "📦 Inventory Health",
        "🌍 External Risk Signals",
        "🔬 AI Explainability",
    ], label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.caption("Built with Python · DuckDB · Streamlit")
    return page


# ──────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ──────────────────────────────────────────
def page_overview(suppliers, shipments, inventory, risk_scores):
    st.title("🏠 Supply Chain Overview")
    st.caption("Real-time snapshot of OpticalFlow supply chain health")
    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    total_suppliers    = len(suppliers)
    active_suppliers   = int(suppliers["is_active"].sum())
    critical_suppliers = len(risk_scores[risk_scores["risk_tier"] == "CRITICAL"])
    disrupted          = int(shipments["is_disrupted"].sum())
    disruption_rate    = round(disrupted / len(shipments) * 100, 1)
    stockouts          = len(inventory[inventory["stock_status"] == "STOCKOUT"])
    critical_stock     = len(inventory[inventory["stock_status"] == "CRITICAL"])

    col1.metric("Total Suppliers",  total_suppliers)
    col2.metric("Active Suppliers", active_suppliers)
    col3.metric("Critical Risk",    critical_suppliers, delta=f"{critical_suppliers} need attention", delta_color="inverse")
    col4.metric("Disruption Rate",  f"{disruption_rate}%", delta="vs 15% benchmark", delta_color="inverse")
    col5.metric("Stock Alerts",     stockouts + critical_stock, delta=f"{stockouts} stockouts", delta_color="inverse")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Supplier Risk Distribution")
        risk_counts = risk_scores["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        color_map = {"CRITICAL": "#ff4b4b", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
        fig = px.pie(risk_counts, values="Count", names="Risk Tier", hole=0.5,
                     color="Risk Tier", color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Shipment Status Breakdown")
        status_counts = shipments["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        status_colors = {"DELIVERED": "#00cc88", "ON_TIME": "#4f8bf9",
                         "IN_TRANSIT": "#ffd700", "DELAYED": "#ff8c00", "CANCELLED": "#ff4b4b"}
        fig2 = px.bar(status_counts, x="Status", y="Count", color="Status",
                      color_discrete_map=status_colors)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", showlegend=False, height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌍 Supplier Geographic Distribution")
    country_counts = suppliers["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]
    fig3 = px.choropleth(country_counts, locations="country",
                         locationmode="country names", color="count",
                         color_continuous_scale="Blues")
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                       height=400, geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────
# PAGE 2 — SUPPLIER RISK
# ──────────────────────────────────────────
def page_supplier_risk(risk_scores):
    st.title("⚠️ Supplier Risk Intelligence")
    st.caption("AI-generated disruption risk scores for all suppliers")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        tier_filter = st.multiselect("Filter by Risk Tier",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"])
    with col2:
        country_filter = st.multiselect("Filter by Country",
            options=sorted(risk_scores["country"].unique()), default=[])

    filtered = risk_scores.copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if country_filter:
        filtered = filtered[filtered["country"].isin(country_filter)]

    st.subheader("Risk Probability vs Disruption Rate")
    fig = px.scatter(filtered, x="disruption_rate_pct", y="risk_probability",
                     color="risk_tier", size="total_shipments",
                     hover_name="supplier_name",
                     hover_data=["country", "product_category", "avg_delay_days"],
                     color_discrete_map={"CRITICAL": "#ff4b4b", "HIGH": "#ff8c00",
                                         "MEDIUM": "#ffd700", "LOW": "#00cc88"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Supplier Risk Scores ({len(filtered)} suppliers)")

    def color_risk(val):
        colors = {"CRITICAL": "background-color: #ff4b4b; color: white",
                  "HIGH":     "background-color: #ff8c00; color: white",
                  "MEDIUM":   "background-color: #ffd700; color: black",
                  "LOW":      "background-color: #00cc88; color: black"}
        return colors.get(val, "")

    display_cols = ["supplier_id", "supplier_name", "country", "product_category",
                    "risk_tier", "risk_probability", "disruption_rate_pct",
                    "avg_delay_days", "total_shipments"]
    styled = filtered[display_cols].style.map(color_risk, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, height=400)


# ──────────────────────────────────────────
# PAGE 3 — SHIPMENT MONITOR
# ──────────────────────────────────────────
def page_shipment_monitor(shipments, anomalies):
    st.title("🚢 Shipment Monitor")
    st.caption("Live tracking of shipment delays, carrier performance and anomalies")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments", len(shipments))
    col2.metric("Disrupted",       int(shipments["is_disrupted"].sum()))
    col3.metric("Avg Delay (days)", round(shipments["delay_days"].mean(), 1))
    col4.metric("Anomalies Detected",
                int(anomalies["is_anomaly"].sum()) if not anomalies.empty else "N/A")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Delay Category Distribution")
        delay_counts = shipments["delay_category"].value_counts().reset_index()
        delay_counts.columns = ["Category", "Count"]
        color_map = {"NO_DELAY": "#00cc88", "MINOR": "#4f8bf9",
                     "MODERATE": "#ffd700", "SEVERE": "#ff4b4b"}
        fig = px.bar(delay_counts, x="Category", y="Count",
                     color="Category", color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Carrier Performance (Avg Delay Days)")
        carrier_perf = shipments.groupby("carrier")["delay_days"].mean().reset_index()
        carrier_perf.columns = ["Carrier", "Avg Delay Days"]
        carrier_perf = carrier_perf.sort_values("Avg Delay Days")
        fig2 = px.bar(carrier_perf, x="Avg Delay Days", y="Carrier",
                      orientation="h", color="Avg Delay Days",
                      color_continuous_scale="RdYlGn_r")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Anomalies section
    if not anomalies.empty:
        st.divider()
        st.subheader("🚨 Detected Shipment Anomalies")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomalies", int(anomalies["is_anomaly"].sum()))
        col2.metric("Critical Anomalies",
                    len(anomalies[anomalies["severity_tier"] == "CRITICAL"]))
        col3.metric("Avg Anomaly Severity",
                    round(anomalies[anomalies["is_anomaly"]]["anomaly_severity"].mean(), 3))

        tier_counts = anomalies["severity_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        color_map2 = {"NORMAL": "#00cc88", "WATCH": "#4f8bf9",
                      "WARNING": "#ffd700", "CRITICAL": "#ff4b4b"}
        fig3 = px.bar(tier_counts, x="Tier", y="Count", color="Tier",
                      color_discrete_map=color_map2,
                      title="Anomaly Severity Distribution")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=280, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Top Anomalous Shipments")
        top_anomalies = anomalies[anomalies["is_anomaly"]].sort_values(
            "anomaly_severity", ascending=False
        ).head(15)
        st.dataframe(top_anomalies[[
            "shipment_id", "supplier_id", "origin_country", "carrier",
            "delay_days", "severity_tier", "anomaly_severity", "anomaly_reasons"
        ]], use_container_width=True)


# ──────────────────────────────────────────
# PAGE 4 — INVENTORY HEALTH
# ──────────────────────────────────────────
def page_inventory_health(inventory, stockout):
    st.title("📦 Inventory Health Monitor")
    st.caption("Stock levels, reorder alerts, and 90-day demand forecasts")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs",       inventory["sku"].nunique())
    col2.metric("Warehouses",       inventory["warehouse"].nunique())
    col3.metric("Stockouts",        len(inventory[inventory["stock_status"] == "STOCKOUT"]))
    col4.metric("Below Reorder",    len(inventory[inventory["stock_status"] == "CRITICAL"]))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Stock Status Distribution")
        status_counts = inventory["stock_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"HEALTHY": "#00cc88", "LOW": "#ffd700",
                     "CRITICAL": "#ff8c00", "STOCKOUT": "#ff4b4b"}
        fig = px.pie(status_counts, values="Count", names="Status",
                     hole=0.4, color="Status", color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Stock Value by Warehouse (USD)")
        wh_value = inventory.groupby("warehouse")["stock_value_usd"].sum().reset_index()
        wh_value.columns = ["Warehouse", "Stock Value USD"]
        wh_value = wh_value.sort_values("Stock Value USD")
        fig2 = px.bar(wh_value, x="Stock Value USD", y="Warehouse",
                      orientation="h", color="Stock Value USD",
                      color_continuous_scale="Blues")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Prophet stockout forecasts
    if not stockout.empty:
        st.divider()
        st.subheader("🔮 90-Day Stockout Risk Forecast (Prophet)")

        risk_counts = stockout["stockout_risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map3 = {"LOW": "#00cc88", "MEDIUM": "#ffd700",
                      "HIGH": "#ff8c00", "CRITICAL": "#ff4b4b"}
        fig3 = px.bar(risk_counts, x="Risk", y="Count", color="Risk",
                      color_discrete_map=color_map3)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=280, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("SKUs Closest to Stockout")
        closest = stockout.sort_values("days_until_stockout").head(10)
        st.dataframe(closest[[
            "sku", "warehouse", "current_stock", "avg_daily_demand",
            "days_until_stockout", "forecasted_demand_90d", "stockout_risk"
        ]], use_container_width=True)


# ──────────────────────────────────────────
# PAGE 5 — EXTERNAL RISK SIGNALS
# ──────────────────────────────────────────
def page_external_risk(lpi_df, weather_df, currency_df):
    st.title("🌍 External Risk Signals")
    st.caption("Real-world logistics, weather, and currency risk data")
    st.divider()

    # ── LPI ──
    if not lpi_df.empty:
        st.subheader("🏭 Logistics Performance Index (World Bank)")
        st.caption("Official country logistics scores — higher = better supply chain infrastructure")

        lpi_display = lpi_df.copy()
        if "our_country_name" in lpi_display.columns:
            lpi_display = lpi_display.rename(columns={"our_country_name": "country"})

        col_a, col_b = st.columns(2)
        with col_a:
            if "lpi_score" in lpi_display.columns and "country" in lpi_display.columns:
                fig = px.bar(
                    lpi_display.sort_values("lpi_score", ascending=True),
                    x="lpi_score", y="country", orientation="h",
                    color="lpi_score", color_continuous_scale="RdYlGn",
                    title="LPI Score by Supplier Country"
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="white", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "logistics_risk_score" in lpi_display.columns:
                fig2 = px.bar(
                    lpi_display.sort_values("logistics_risk_score", ascending=False),
                    x="logistics_risk_score", y="country", orientation="h",
                    color="logistics_risk_score", color_continuous_scale="Reds",
                    title="Logistics Risk Score (inverse of LPI)"
                )
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="white", height=350)
                st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(lpi_display, use_container_width=True)

    st.divider()

    # ── WEATHER ──
    if not weather_df.empty:
        st.subheader("🌦️ Weather Risk — Supplier Port Cities (Last 30 Days)")

        color_map = {"LOW": "#00cc88", "MEDIUM": "#ffd700",
                     "HIGH": "#ff8c00", "CRITICAL": "#ff4b4b"}
        fig3 = px.bar(
            weather_df.sort_values("avg_disruption_score", ascending=False),
            x="country", y="avg_disruption_score",
            color="country_weather_risk",
            color_discrete_map=color_map,
            hover_data=["city", "severe_days", "warning_days", "avg_precipitation_mm"],
            title="Average Weather Disruption Score by Country"
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=350, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(weather_df[[
            "country", "city", "avg_disruption_score", "max_disruption_score",
            "severe_days", "warning_days", "avg_precipitation_mm",
            "avg_windspeed_kmh", "country_weather_risk"
        ]], use_container_width=True)

    st.divider()

    # ── CURRENCY ──
    if not currency_df.empty:
        st.subheader("💱 Currency Exchange Rate Risk (30-Day Volatility)")

        color_map2 = {"LOW": "#00cc88", "MEDIUM": "#ffd700",
                      "HIGH": "#ff8c00", "CRITICAL": "#ff4b4b"}
        fig4 = px.bar(
            currency_df.sort_values("currency_risk_score", ascending=False),
            x="country", y="currency_risk_score",
            color="currency_risk_tier",
            color_discrete_map=color_map2,
            hover_data=["currency_code", "pct_change_30d", "daily_volatility_pct"],
            title="Currency Risk Score by Supplier Country"
        )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=350)
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(currency_df[[
            "country", "currency_code", "current_rate_vs_usd",
            "pct_change_30d", "daily_volatility_pct",
            "currency_risk_score", "currency_risk_tier"
        ]], use_container_width=True)


# ──────────────────────────────────────────
# PAGE 6 — AI EXPLAINABILITY
# ──────────────────────────────────────────
def page_explainability(shap_df):
    st.title("🔬 AI Explainability (SHAP)")
    st.caption("Why is each supplier flagged as high risk? SHAP values explain every prediction.")
    st.divider()

    if shap_df.empty:
        st.warning("SHAP data not found. Run ml_pipeline/risk_explainer.py first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        tier_filter = st.multiselect("Filter by Risk Tier",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"])
    with col2:
        driver_filter = st.multiselect("Filter by Top Risk Driver",
            options=sorted(shap_df["top_risk_driver"].unique()), default=[])

    filtered = shap_df.copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if driver_filter:
        filtered = filtered[filtered["top_risk_driver"].isin(driver_filter)]

    # Global feature importance
    st.subheader("Global Feature Importance (Mean |SHAP|)")
    factor_cols = [
        "Reliability Score", "Lead Time (Days)", "Is Active",
        "Annual Spend (USD)", "Total Shipments", "Disrupted Shipments",
        "Avg Delay (Days)", "Max Delay (Days)", "Disruption Rate %",
        "Country", "Product Category", "Risk Level"
    ]
    available_cols = [c for c in factor_cols if c in shap_df.columns]
    if available_cols:
        importance = shap_df[available_cols].abs().mean().reset_index()
        importance.columns = ["Feature", "Mean |SHAP|"]
        importance = importance.sort_values("Mean |SHAP|", ascending=True)
        fig = px.bar(importance, x="Mean |SHAP|", y="Feature", orientation="h",
                     color="Mean |SHAP|", color_continuous_scale="Blues")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Top risk driver breakdown
    st.subheader("Top Risk Driver Distribution")
    driver_counts = shap_df["top_risk_driver"].value_counts().reset_index()
    driver_counts.columns = ["Driver", "Count"]
    fig2 = px.pie(driver_counts, values="Count", names="Driver", hole=0.4)
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       font_color="white", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Per-supplier explanations
    st.subheader(f"Supplier Risk Explanations ({len(filtered)} suppliers)")
    if not filtered.empty:
        for _, row in filtered.head(10).iterrows():
            with st.expander(
                f"{row['supplier_id']} — {row['supplier_name']} "
                f"| {row['risk_tier']} ({row['risk_probability']})"
            ):
                st.write(f"**Top Risk Driver:** {row['top_risk_driver']}")
                st.write(f"**Explanation:** {row['explanation']}")
                if available_cols:
                    factor_data = pd.DataFrame({
                        "Feature": available_cols,
                        "SHAP Value": [row[c] for c in available_cols]
                    }).sort_values("SHAP Value", key=abs, ascending=False)
                    fig3 = px.bar(factor_data, x="SHAP Value", y="Feature",
                                  orientation="h",
                                  color="SHAP Value",
                                  color_continuous_scale="RdBu_r")
                    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="white", height=300)
                    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main():
    (suppliers, shipments, inventory, risk_scores,
     stockout, anomalies, shap_df,
     lpi_df, weather_df, currency_df) = load_data()

    page = render_sidebar()

    if   page == "🏠 Overview":
        page_overview(suppliers, shipments, inventory, risk_scores)
    elif page == "⚠️ Supplier Risk":
        page_supplier_risk(risk_scores)
    elif page == "🚢 Shipment Monitor":
        page_shipment_monitor(shipments, anomalies)
    elif page == "📦 Inventory Health":
        page_inventory_health(inventory, stockout)
    elif page == "🌍 External Risk Signals":
        page_external_risk(lpi_df, weather_df, currency_df)
    elif page == "🔬 AI Explainability":
        page_explainability(shap_df)

if __name__ == "__main__":
    main()