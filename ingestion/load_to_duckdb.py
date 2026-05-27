# ingestion/load_to_duckdb.py

import duckdb
import pandas as pd
import os
from datetime import datetime

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DB_PATH      = "data/opticalflow.duckdb"
RAW_DATA_DIR = "data/raw"

TABLES = {
    "raw_suppliers" : "suppliers.csv",
    "raw_inventory" : "inventory.csv",
    "raw_shipments" : "shipments.csv",
}

# ──────────────────────────────────────────
# CONNECT
# ──────────────────────────────────────────
def get_connection():
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    print(f"[✓] Connected to DuckDB → {DB_PATH}\n")
    return con

# ──────────────────────────────────────────
# CREATE RAW SCHEMA
# ──────────────────────────────────────────
def create_schema(con):
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    print("[✓] Schema 'raw' ready")

# ──────────────────────────────────────────
# LOAD CSV → DUCKDB TABLE
# ──────────────────────────────────────────
def load_table(con, table_name, filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[✗] File not found: {filepath}")
        return

    df = pd.read_csv(filepath)

    # Add ingestion metadata columns
    df["_ingested_at"]  = datetime.utcnow().isoformat()
    df["_source_file"]  = filename

    # Drop and recreate table (full refresh for raw layer)
    con.execute(f"DROP TABLE IF EXISTS raw.{table_name};")
    con.execute(f"CREATE TABLE raw.{table_name} AS SELECT * FROM df;")

    row_count = con.execute(f"SELECT COUNT(*) FROM raw.{table_name};").fetchone()[0]
    print(f"[✓] raw.{table_name:<20} → {row_count} rows loaded")

# ──────────────────────────────────────────
# VERIFY LOAD
# ──────────────────────────────────────────
def verify_load(con):
    print("\n📊 Load Summary:")
    print("─" * 50)
    tables = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema = 'raw'
        ORDER BY table_name;
    """).fetchall()

    for schema, table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0]
        cols  = con.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table}'").fetchone()[0]
        print(f"  {schema}.{table:<25} {count:>6} rows  |  {cols} columns")

    print("─" * 50)

# ──────────────────────────────────────────
# QUICK DATA PREVIEW
# ──────────────────────────────────────────
def preview_tables(con):
    print("\n🔍 Data Previews (3 rows each):\n")
    for table_name in TABLES:
        print(f"── raw.{table_name} ──")
        df = con.execute(f"SELECT * FROM raw.{table_name} LIMIT 3;").df()
        print(df.to_string(index=False))
        print()

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 OpticalFlow — DuckDB Ingestion Pipeline\n")

    con = get_connection()
    create_schema(con)

    print()
    for table_name, filename in TABLES.items():
        load_table(con, table_name, filename)

    verify_load(con)
    preview_tables(con)

    con.close()
    print("✅ Ingestion complete. Database saved to:", DB_PATH)