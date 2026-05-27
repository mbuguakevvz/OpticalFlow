-- models/marts/mart_supplier_risk.sql
WITH suppliers AS (
    SELECT * FROM "opticalflow"."transformed_staging"."stg_suppliers"
),

shipments AS (
    SELECT * FROM "opticalflow"."transformed_staging"."stg_shipments"
),

shipment_stats AS (
    SELECT
        supplier_id,
        COUNT(*)                                        AS total_shipments,
        SUM(CASE WHEN is_disrupted THEN 1 ELSE 0 END)  AS disrupted_shipments,
        AVG(delay_days)                                 AS avg_delay_days,
        MAX(delay_days)                                 AS max_delay_days,
        SUM(freight_cost_usd)                           AS total_freight_cost
    FROM shipments
    GROUP BY supplier_id
)

SELECT
    s.supplier_id,
    s.supplier_name,
    s.country,
    s.product_category,
    s.risk_level,
    s.reliability_score,
    s.lead_time_days,
    s.is_active,
    s.annual_spend_usd,
    COALESCE(ss.total_shipments, 0)                     AS total_shipments,
    COALESCE(ss.disrupted_shipments, 0)                 AS disrupted_shipments,
    COALESCE(ss.avg_delay_days, 0)                      AS avg_delay_days,
    COALESCE(ss.max_delay_days, 0)                      AS max_delay_days,
    COALESCE(ss.total_freight_cost, 0)                  AS total_freight_cost,
    ROUND(
        COALESCE(ss.disrupted_shipments, 0) * 1.0 /
        NULLIF(ss.total_shipments, 0) * 100, 2
    )                                                   AS disruption_rate_pct,
    CASE
        WHEN s.risk_level = 'CRITICAL'                  THEN 4
        WHEN s.risk_level = 'HIGH'                      THEN 3
        WHEN s.risk_level = 'MEDIUM'                    THEN 2
        ELSE                                                 1
    END                                                 AS risk_score
FROM suppliers s
LEFT JOIN shipment_stats ss ON s.supplier_id = ss.supplier_id