# ml_pipeline/generate_demand_history.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import duckdb
import random

random.seed(42)
np.random.seed(42)

DB_PATH = "data/opticalflow.duckdb"

SKUS = [
    "SKU-LENS-001", "SKU-LENS-002", "SKU-LENS-003",
    "SKU-FRAME-001", "SKU-FRAME-002", "SKU-FRAME-003",
    "SKU-SUN-001", "SKU-SUN-002",
    "SKU-CASE-001", "SKU-COAT-001"
]

WAREHOUSES = [
    "Nairobi-KE", "Mombasa-KE", "Lagos-NG",
    "Accra-GH", "Cairo-EG", "Johannesburg-ZA"
]

def generate_demand_history(days=365):
    """
    Generates daily demand history per SKU per warehouse
    with realistic patterns:
    - Weekly seasonality (higher demand mid-week)
    - Monthly seasonality (higher demand start of month)
    - Random noise
    - Occasional demand spikes (humanitarian events)
    """
    print("Generating demand history...")
    records = []

    start_date = datetime.today() - timedelta(days=days)

    for sku in SKUS:
        for warehouse in WAREHOUSES:
            # Base demand varies by SKU and warehouse
            base_demand = random.randint(10, 120)

            for day_offset in range(days):
                date = start_date + timedelta(days=day_offset)

                # Weekly pattern — higher mid-week
                day_of_week   = date.weekday()
                weekly_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_week / 7)

                # Monthly pattern — higher at start of month
                day_of_month   = date.day
                monthly_factor = 1.0 + 0.2 * np.exp(-day_of_month / 10)

                # Random noise
                noise = np.random.normal(1.0, 0.15)

                # Occasional humanitarian demand spike (1% chance)
                spike = 3.0 if random.random() < 0.01 else 1.0

                demand = max(0, int(
                    base_demand * weekly_factor * monthly_factor * noise * spike
                ))

                records.append({
                    "date"      : date.strftime("%Y-%m-%d"),
                    "sku"       : sku,
                    "warehouse" : warehouse,
                    "demand"    : demand,
                })

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} daily demand records")
    return df


def save_to_duckdb(df):
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("DROP TABLE IF EXISTS raw.demand_history;")
    con.execute("CREATE TABLE raw.demand_history AS SELECT * FROM df;")
    count = con.execute("SELECT COUNT(*) FROM raw.demand_history").fetchone()[0]
    print(f"Saved {count:,} rows to raw.demand_history")
    con.close()


if __name__ == "__main__":
    df = generate_demand_history(days=365)
    save_to_duckdb(df)
    print("Done.")