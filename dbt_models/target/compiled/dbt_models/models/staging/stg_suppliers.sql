-- models/staging/stg_suppliers.sql
WITH source AS (
    SELECT * FROM raw.raw_suppliers
)

SELECT
    supplier_id,
    TRIM(supplier_name)                         AS supplier_name,
    TRIM(country)                               AS country,
    LOWER(TRIM(contact_email))                  AS contact_email,
    TRIM(product_category)                      AS product_category,
    CAST(lead_time_days AS INTEGER)             AS lead_time_days,
    CAST(reliability_score AS DOUBLE)           AS reliability_score,
    TRIM(risk_level)                            AS risk_level,
    CAST(active AS BOOLEAN)                     AS is_active,
    CAST(onboarded_date AS DATE)                AS onboarded_date,
    CAST(annual_spend_usd AS DOUBLE)            AS annual_spend_usd,
    CAST(_ingested_at AS TIMESTAMP)             AS ingested_at
FROM source
WHERE supplier_id IS NOT NULL