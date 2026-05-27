# 👁️ OpticalFlow
### AI-Driven Supply Chain Resilience for the Eyewear Industry

![Python](https://img.shields.io/badge/Python-3.10-blue)
![DuckDB](https://img.shields.io/badge/DuckDB-latest-yellow)
![dbt](https://img.shields.io/badge/dbt-1.x-orange)
![Prefect](https://img.shields.io/badge/Prefect-2.x-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)

---

## The Problem

2.2 billion people globally live with vision impairment. A significant portion
lack access to corrective eyewear not because of funding gaps — but because of
**supply chain failures**. Clinics in low-income regions routinely run out of
specific lens prescriptions for months at a time due to unpredictable supplier
disruptions, shipping delays, and poor inventory visibility.

**OpticalFlow** is a data engineering and AI project that tackles this problem
by building an intelligent supply chain monitoring and prediction system for
the eyewear industry.

---

## What It Does

- **Monitors** supplier reliability, shipment delays, and inventory health in real time
- **Predicts** which suppliers are likely to cause disruptions using ML
- **Alerts** on critical stock levels and reorder points across warehouses
- **Orchestrates** the full data pipeline automatically on a daily schedule

---

## System ArchitectureSee [docs/architecture.md](docs/architecture.md) for the full diagram.

---

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| Language       | Python 3.10                       |
| Data Warehouse | DuckDB                            |
| Transforms     | dbt (dbt-duckdb)                  |
| ML Model       | Scikit-learn GradientBoosting     |
| Orchestration  | Prefect                           |
| Dashboard      | Streamlit + Plotly                |
| Data Gen       | Faker + NumPy                     |

---

## Project Structure---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/mbuguakevvz/OpticalFlow.git
cd OpticalFlow
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python orchestration/pipeline_flow.py
```

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Pipeline StagesEach stage is a Prefect task with automatic retries and logging.

---

## ML Model

The disruption risk model uses **Gradient Boosting Classification** trained on:
- Supplier reliability scores and lead times
- Historical shipment delay patterns
- Disruption rate per supplier
- Country and product category risk encoding

**Output:** A risk probability (0–1) and tier (LOW / MEDIUM / HIGH / CRITICAL)
for every supplier, saved to `predictions.supplier_risk_scores` in DuckDB.

---

## Humanitarian Impact

OpticalFlow is designed with humanitarian supply chains in mind:

- Models last-mile delivery risk for clinics in underserved regions
- Flags when warehouse stock will fall below critical levels before resupply
- Identifies high-risk suppliers so alternative sourcing can be planned early
- Built to scale to real NGO and public health supply chain data

---

## Author

**Kevin Mbugua**
Data Engineer | Kenya
GitHub: [@mbuguakevvz](https://github.com/mbuguakevvz)