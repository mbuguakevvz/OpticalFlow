# ingestion/generate_mock_data.py

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
SUPPLIER_COUNT     = 50
INVENTORY_ROWS     = 500
SHIPMENT_ROWS      = 1000

COUNTRIES = [
    "China", "Vietnam", "Italy", "Germany", "Kenya",
    "India", "USA", "France", "Japan", "South Korea"
]

PRODUCT_CATEGORIES = ["Prescription Lenses", "Frames", "Sunglasses", "Cases", "Lens Coatings"]

PRODUCT_SKUS = [
    "SKU-LENS-001", "SKU-LENS-002", "SKU-FRAME-001", "SKU-FRAME-002",
    "SKU-SUN-001",  "SKU-SUN-002",  "SKU-CASE-001", "SKU-COAT-001",
    "SKU-LENS-003", "SKU-FRAME-003"
]

WAREHOUSES = [
    "Nairobi-KE", "Mombasa-KE", "Lagos-NG", "Accra-GH",
    "Cairo-EG",   "Johannesburg-ZA", "London-UK", "Dubai-UAE"
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SHIPMENT_STATUSES = ["ON_TIME", "DELAYED", "IN_TRANSIT", "DELIVERED", "CANCELLED"]

# ──────────────────────────────────────────
# 1. SUPPLIERS
# ──────────────────────────────────────────
def generate_suppliers():
    records = []
    for i in range(1, SUPPLIER_COUNT + 1):
        country = random.choice(COUNTRIES)
        risk    = random.choices(
            RISK_LEVELS,
            weights=[40, 30, 20, 10]  # weighted toward lower risk
        )[0]
        records.append({
            "supplier_id"         : f"SUP-{i:03d}",
            "supplier_name"       : fake.company(),
            "country"             : country,
            "contact_email"       : fake.email(),
            "product_category"    : random.choice(PRODUCT_CATEGORIES),
            "lead_time_days"      : random.randint(7, 90),
            "reliability_score"   : round(random.uniform(0.5, 1.0), 2),
            "risk_level"          : risk,
            "active"              : random.choices([True, False], weights=[85, 15])[0],
            "onboarded_date"      : fake.date_between(start_date="-5y", end_date="-6m"),
            "annual_spend_usd"    : round(random.uniform(10000, 2000000), 2),
        })
    df = pd.DataFrame(records)
    path = f"{OUTPUT_DIR}/suppliers.csv"
    df.to_csv(path, index=False)
    print(f"[✓] suppliers.csv — {len(df)} rows → {path}")
    return df


# ──────────────────────────────────────────
# 2. INVENTORY
# ──────────────────────────────────────────
def generate_inventory(suppliers_df):
    records = []
    supplier_ids = suppliers_df["supplier_id"].tolist()
    for i in range(1, INVENTORY_ROWS + 1):
        qty_on_hand   = random.randint(0, 5000)
        reorder_point = random.randint(50, 500)
        records.append({
            "inventory_id"        : f"INV-{i:04d}",
            "sku"                 : random.choice(PRODUCT_SKUS),
            "product_category"    : random.choice(PRODUCT_CATEGORIES),
            "warehouse"           : random.choice(WAREHOUSES),
            "supplier_id"         : random.choice(supplier_ids),
            "quantity_on_hand"    : qty_on_hand,
            "reorder_point"       : reorder_point,
            "below_reorder"       : qty_on_hand < reorder_point,
            "unit_cost_usd"       : round(random.uniform(2.5, 350.0), 2),
            "last_restocked_date" : fake.date_between(start_date="-6m", end_date="today"),
            "expiry_date"         : fake.date_between(start_date="+6m", end_date="+3y"),
        })
    df = pd.DataFrame(records)
    path = f"{OUTPUT_DIR}/inventory.csv"
    df.to_csv(path, index=False)
    print(f"[✓] inventory.csv  — {len(df)} rows → {path}")
    return df


# ──────────────────────────────────────────
# 3. SHIPMENTS
# ──────────────────────────────────────────
def generate_shipments(suppliers_df):
    records = []
    supplier_ids = suppliers_df["supplier_id"].tolist()
    for i in range(1, SHIPMENT_ROWS + 1):
        supplier_id     = random.choice(supplier_ids)
        origin_country  = suppliers_df.loc[
            suppliers_df["supplier_id"] == supplier_id, "country"
        ].values[0]

        scheduled_date  = fake.date_between(start_date="-1y", end_date="+2m")
        delay_days      = random.choices(
            [0, random.randint(1,5), random.randint(6,20), random.randint(21,60)],
            weights=[50, 25, 15, 10]
        )[0]
        actual_date     = scheduled_date + timedelta(days=delay_days)
        status          = (
            "DELAYED"    if delay_days > 5  else
            "ON_TIME"    if delay_days == 0 else
            "IN_TRANSIT" if scheduled_date > datetime.today().date() else
            "DELIVERED"
        )

        records.append({
            "shipment_id"          : f"SHP-{i:04d}",
            "supplier_id"          : supplier_id,
            "sku"                  : random.choice(PRODUCT_SKUS),
            "origin_country"       : origin_country,
            "destination_warehouse": random.choice(WAREHOUSES),
            "quantity_shipped"     : random.randint(10, 2000),
            "scheduled_date"       : scheduled_date,
            "actual_arrival_date"  : actual_date,
            "delay_days"           : delay_days,
            "status"               : status,
            "freight_cost_usd"     : round(random.uniform(100, 15000), 2),
            "carrier"              : random.choice([
                "DHL", "FedEx", "Maersk", "MSC", "Kenya Airways Cargo",
                "Ethiopian Airlines Cargo", "UPS", "CEVA Logistics"
            ]),
            "disruption_flag"      : delay_days > 14,
        })
    df = pd.DataFrame(records)
    path = f"{OUTPUT_DIR}/shipments.csv"
    df.to_csv(path, index=False)
    print(f"[✓] shipments.csv  — {len(df)} rows → {path}")
    return df


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Generating OpticalFlow mock datasets...\n")
    suppliers = generate_suppliers()
    inventory = generate_inventory(suppliers)
    shipments = generate_shipments(suppliers)
    print("\n✅ All datasets generated successfully.\n")