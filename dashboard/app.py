# dashboard/app.py

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DB_PATH = "data/opticalflow.duckdb"

st.set_page_config(
    page_title   = "OpticalFlow",
    page_icon    = "👁️",
    layout       = "wide",
    initial_sidebar_state = "expanded"
)

# ──────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4f8bf9;
        margin-bottom: 10px;
    }
    .critical { border-left-color: #ff4b4b !important; }
    .warning  { border-left-color: #ffa500 !important; }
    .success  { border-left-color: #00cc88 !important; }
   .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 2rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #a0aec0 !important; font-size: 0.85rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)

    suppliers = con.execute("""
        SELECT * FROM transformed_staging.stg_suppliers
    """).df()

    shipments = con.execute("""
        SELECT * FROM transformed_staging.stg_shipments
    """).df()

    inventory = con.execute("""
        SELECT * FROM transformed_marts.mart_inventory_health
    """).df()

    risk_scores = con.execute("""
        SELECT * FROM predictions.supplier_risk_scores
        ORDER BY risk_probability DESC
    """).df()

    con.close()
    return suppliers, shipments, inventory, risk_scores


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/glasses.png",
        width=60
    )
    st.sidebar.title("👁️ OpticalFlow")
    st.sidebar.caption("AI-Driven Supply Chain Resilience")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "⚠️ Supplier Risk", "🚢 Shipment Monitor", "📦 Inventory Health"],
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.caption("🌍 Eyewear Supply Chain Intelligence")
    st.sidebar.caption("Built with Python · DuckDB · Streamlit")
    return page


# ──────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ──────────────────────────────────────────
def page_overview(suppliers, shipments, inventory, risk_scores):
    st.title("🏠 Supply Chain Overview")
    st.caption("Real-time snapshot of OpticalFlow supply chain health")
    st.divider()

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_suppliers    = len(suppliers)
    active_suppliers   = suppliers["is_active"].sum()
    critical_suppliers = len(risk_scores[risk_scores["risk_tier"] == "CRITICAL"])
    disrupted          = shipments["is_disrupted"].sum()
    disruption_rate    = round(disrupted / len(shipments) * 100, 1)
    stockouts          = len(inventory[inventory["stock_status"] == "STOCKOUT"])
    critical_stock     = len(inventory[inventory["stock_status"] == "CRITICAL"])

    col1.metric("Total Suppliers",    total_suppliers)
    col2.metric("Active Suppliers",   int(active_suppliers))
    col3.metric("🚨 Critical Risk",   critical_suppliers, delta=f"{critical_suppliers} need attention", delta_color="inverse")
    col4.metric("Disruption Rate",    f"{disruption_rate}%", delta="vs 15% benchmark", delta_color="inverse")
    col5.metric("Stock Alerts",       stockouts + critical_stock, delta=f"{stockouts} stockouts", delta_color="inverse")

    st.divider()

    col_a, col_b = st.columns(2)

    # Risk tier donut
    with col_a:
        st.subheader("Supplier Risk Distribution")
        risk_counts = risk_scores["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        color_map = {"CRITICAL": "#ff4b4b", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
        fig = px.pie(
            risk_counts, values="Count", names="Risk Tier",
            hole=0.5, color="Risk Tier", color_discrete_map=color_map
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=True, height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    # Shipment status bar
    with col_b:
        st.subheader("Shipment Status Breakdown")
        status_counts = shipments["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        status_colors = {
            "DELIVERED": "#00cc88", "ON_TIME": "#4f8bf9",
            "IN_TRANSIT": "#ffd700", "DELAYED": "#ff8c00", "CANCELLED": "#ff4b4b"
        }
        fig2 = px.bar(
            status_counts, x="Status", y="Count",
            color="Status", color_discrete_map=status_colors
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", showlegend=False, height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Supplier countries map
    st.subheader("🌍 Supplier Geographic Distribution")
    country_counts = suppliers["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]
    fig3 = px.choropleth(
        country_counts, locations="country",
        locationmode="country names", color="count",
        color_continuous_scale="Blues",
        title="Supplier Concentration by Country"
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=400, geo=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────
# PAGE 2 — SUPPLIER RISK
# ──────────────────────────────────────────
def page_supplier_risk(risk_scores):
    st.title("⚠️ Supplier Risk Intelligence")
    st.caption("AI-generated disruption risk scores for all suppliers")
    st.divider()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"]
        )
    with col2:
        country_filter = st.multiselect(
            "Filter by Country",
            options=sorted(risk_scores["country"].unique()),
            default=[]
        )

    filtered = risk_scores.copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if country_filter:
        filtered = filtered[filtered["country"].isin(country_filter)]

    # Risk score scatter
    st.subheader("Risk Probability vs Disruption Rate")
    fig = px.scatter(
        filtered,
        x="disruption_rate_pct",
        y="risk_probability",
        color="risk_tier",
        size="total_shipments",
        hover_name="supplier_name",
        hover_data=["country", "product_category", "avg_delay_days"],
        color_discrete_map={"CRITICAL": "#ff4b4b", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#00cc88"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk table
    st.subheader(f"Supplier Risk Scores ({len(filtered)} suppliers)")
    display_cols = [
        "supplier_id", "supplier_name", "country", "product_category",
        "risk_tier", "risk_probability", "disruption_rate_pct",
        "avg_delay_days", "total_shipments"
    ]

    def color_risk(val):
        colors = {"CRITICAL": "background-color: #ff4b4b; color: white",
                  "HIGH":     "background-color: #ff8c00; color: white",
                  "MEDIUM":   "background-color: #ffd700; color: black",
                  "LOW":      "background-color: #00cc88; color: black"}
        return colors.get(val, "")

    styled = filtered[display_cols].style.applymap(color_risk, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, height=400)


# ──────────────────────────────────────────
# PAGE 3 — SHIPMENT MONITOR
# ──────────────────────────────────────────
def page_shipment_monitor(shipments):
    st.title("🚢 Shipment Monitor")
    st.caption("Live tracking of shipment delays and carrier performance")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments",  len(shipments))
    col2.metric("Disrupted",        int(shipments["is_disrupted"].sum()))
    col3.metric("Avg Delay (days)", round(shipments["delay_days"].mean(), 1))
    col4.metric("Severe Delays",    len(shipments[shipments["delay_category"] == "SEVERE"]))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Delay Category Distribution")
        delay_counts = shipments["delay_category"].value_counts().reset_index()
        delay_counts.columns = ["Category", "Count"]
        color_map = {"NO_DELAY": "#00cc88", "MINOR": "#4f8bf9", "MODERATE": "#ffd700", "SEVERE": "#ff4b4b"}
        fig = px.bar(delay_counts, x="Category", y="Count", color="Category", color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Carrier Performance (Avg Delay Days)")
        carrier_perf = shipments.groupby("carrier")["delay_days"].mean().reset_index()
        carrier_perf.columns = ["Carrier", "Avg Delay Days"]
        carrier_perf = carrier_perf.sort_values("Avg Delay Days", ascending=True)
        fig2 = px.bar(carrier_perf, x="Avg Delay Days", y="Carrier", orientation="h", color="Avg Delay Days", color_continuous_scale="RdYlGn_r")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Origin country disruption heatmap
    st.subheader("Disruptions by Origin Country")
    origin_disruptions = shipments.groupby("origin_country").agg(
        total=("shipment_id", "count"),
        disrupted=("is_disrupted", "sum")
    ).reset_index()
    origin_disruptions["disruption_rate"] = (origin_disruptions["disrupted"] / origin_disruptions["total"] * 100).round(1)

    fig3 = px.choropleth(
        origin_disruptions, locations="origin_country",
        locationmode="country names", color="disruption_rate",
        color_continuous_scale="Reds", hover_data=["total", "disrupted"],
        title="Disruption Rate % by Origin Country"
    )
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=400, geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, use_container_width=True)

    # Recent disruptions table
    st.subheader("🚨 Recent Disrupted Shipments")
    disrupted_df = shipments[shipments["is_disrupted"]].sort_values("delay_days", ascending=False).head(20)
    st.dataframe(disrupted_df[[
        "shipment_id", "supplier_id", "sku", "origin_country",
        "destination_warehouse", "delay_days", "delay_category",
        "carrier", "freight_cost_usd", "status"
    ]], use_container_width=True)


# ──────────────────────────────────────────
# PAGE 4 — INVENTORY HEALTH
# ──────────────────────────────────────────
def page_inventory_health(inventory):
    st.title("📦 Inventory Health Monitor")
    st.caption("Stock levels, reorder alerts, and warehouse health")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs",       inventory["sku"].nunique())
    col2.metric("Warehouses",       inventory["warehouse"].nunique())
    col3.metric("🔴 Stockouts",     len(inventory[inventory["stock_status"] == "STOCKOUT"]))
    col4.metric("🟡 Below Reorder", len(inventory[inventory["stock_status"] == "CRITICAL"]))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Stock Status Distribution")
        status_counts = inventory["stock_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"HEALTHY": "#00cc88", "LOW": "#ffd700", "CRITICAL": "#ff8c00", "STOCKOUT": "#ff4b4b"}
        fig = px.pie(status_counts, values="Count", names="Status", hole=0.4, color="Status", color_discrete_map=color_map)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Stock Value by Warehouse (USD)")
        wh_value = inventory.groupby("warehouse")["stock_value_usd"].sum().reset_index()
        wh_value.columns = ["Warehouse", "Stock Value USD"]
        wh_value = wh_value.sort_values("Stock Value USD", ascending=True)
        fig2 = px.bar(wh_value, x="Stock Value USD", y="Warehouse", orientation="h", color="Stock Value USD", color_continuous_scale="Blues")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Critical inventory table
    st.subheader("🚨 Items Needing Immediate Attention")
    urgent = inventory[inventory["stock_status"].isin(["STOCKOUT", "CRITICAL"])].sort_values("quantity_on_hand")
    st.dataframe(urgent[[
        "inventory_id", "sku", "product_category", "warehouse",
        "supplier_name", "supplier_risk", "quantity_on_hand",
        "reorder_point", "stock_status", "days_since_restock"
    ]], use_container_width=True, height=350)


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
def main():
    suppliers, shipments, inventory, risk_scores = load_data()
    page = render_sidebar()

    if   page == "🏠 Overview":          page_overview(suppliers, shipments, inventory, risk_scores)
    elif page == "⚠️ Supplier Risk":     page_supplier_risk(risk_scores)
    elif page == "🚢 Shipment Monitor":  page_shipment_monitor(shipments)
    elif page == "📦 Inventory Health":  page_inventory_health(inventory)

if __name__ == "__main__":
    main()