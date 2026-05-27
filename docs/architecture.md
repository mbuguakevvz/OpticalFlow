# OpticalFlow — System Architecture

\`\`\`
+------------------+     +------------------+     +------------------+
|   DATA SOURCES   |     |   INGESTION      |     |   STORAGE        |
|------------------|     |------------------|     |------------------|
| Supplier Data    | --> | generate_mock_   | --> | DuckDB           |
| Inventory Data   |     | data.py          |     | raw.suppliers    |
| Shipment Records |     |                  |     | raw.inventory    |
|                  |     | load_to_duckdb   |     | raw.shipments    |
+------------------+     | .py              |     +------------------+
                         +------------------+              |
                                                           v
+------------------+     +------------------+     +------------------+
|   DASHBOARD      |     |   ML PIPELINE    |     |   TRANSFORMS     |
|------------------|     |------------------|     |------------------|
| Streamlit App    | <-- | disruption_      | <-- | dbt models       |
| Overview         |     | predictor.py     |     | stg_suppliers    |
| Supplier Risk    |     |                  |     | stg_inventory    |
| Shipment Monitor |     | GradientBoosting |     | stg_shipments    |
| Inventory Health |     | Classifier       |     | mart_supplier_   |
|                  |     |                  |     | risk             |
+------------------+     +------------------+     | mart_inventory_  |
                                                   | health           |
+------------------+                              +------------------+
|  ORCHESTRATION   |
|------------------|
| Prefect Flow     |
| Daily Schedule   |
| Health Checks    |
| Auto Retry       |
+------------------+
\`\`\`