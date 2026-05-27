-- models/staging/stg_inventory.sql
WITH source AS (
    SELECT * FROM raw.raw_inventory
)

SELECT
    inventory_id,
    TRIM(sku)                                   AS sku,
    TRIM(product_category)                      AS product_category,
    TRIM(warehouse)                             AS warehouse,
    supplier_id,
    CAST(quantity_on_hand AS INTEGER)           AS quantity_on_hand,
    CAST(reorder_point AS INTEGER)              AS reorder_point,
    CAST(below_reorder AS BOOLEAN)              AS is_below_reorder,
    CAST(unit_cost_usd AS DOUBLE)               AS unit_cost_usd,
    CAST(last_restocked_date AS DATE)           AS last_restocked_date,
    CAST(expiry_date AS DATE)                   AS expiry_date,
    ROUND(quantity_on_hand * unit_cost_usd, 2)  AS stock_value_usd,
    CAST(_ingested_at AS TIMESTAMP)             AS ingested_at
FROM source
WHERE inventory_id IS NOT NULL