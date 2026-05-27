-- models/staging/stg_shipments.sql
WITH source AS (
    SELECT * FROM raw.raw_shipments
)

SELECT
    shipment_id,
    supplier_id,
    TRIM(sku)                                   AS sku,
    TRIM(origin_country)                        AS origin_country,
    TRIM(destination_warehouse)                 AS destination_warehouse,
    CAST(quantity_shipped AS INTEGER)           AS quantity_shipped,
    CAST(scheduled_date AS DATE)                AS scheduled_date,
    CAST(actual_arrival_date AS DATE)           AS actual_arrival_date,
    CAST(delay_days AS INTEGER)                 AS delay_days,
    TRIM(status)                                AS status,
    CAST(freight_cost_usd AS DOUBLE)            AS freight_cost_usd,
    TRIM(carrier)                               AS carrier,
    CAST(disruption_flag AS BOOLEAN)            AS is_disrupted,
    CASE
        WHEN delay_days = 0             THEN 'NO_DELAY'
        WHEN delay_days BETWEEN 1 AND 5 THEN 'MINOR'
        WHEN delay_days BETWEEN 6 AND 14 THEN 'MODERATE'
        ELSE                                 'SEVERE'
    END                                         AS delay_category,
    CAST(_ingested_at AS TIMESTAMP)             AS ingested_at
FROM source
WHERE shipment_id IS NOT NULL