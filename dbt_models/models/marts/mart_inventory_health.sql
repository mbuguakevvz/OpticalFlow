-- models/marts/mart_inventory_health.sql
WITH inventory AS (
    SELECT * FROM {{ ref('stg_inventory') }}
),

suppliers AS (
    SELECT supplier_id, supplier_name, country, risk_level
    FROM {{ ref('stg_suppliers') }}
)

SELECT
    i.inventory_id,
    i.sku,
    i.product_category,
    i.warehouse,
    i.supplier_id,
    s.supplier_name,
    s.country                                               AS supplier_country,
    s.risk_level                                            AS supplier_risk,
    i.quantity_on_hand,
    i.reorder_point,
    i.is_below_reorder,
    i.unit_cost_usd,
    i.stock_value_usd,
    i.last_restocked_date,
    i.expiry_date,
    CURRENT_DATE - i.last_restocked_date                    AS days_since_restock,
    i.expiry_date - CURRENT_DATE                            AS days_to_expiry,
    CASE
        WHEN i.quantity_on_hand = 0                         THEN 'STOCKOUT'
        WHEN i.is_below_reorder                             THEN 'CRITICAL'
        WHEN i.quantity_on_hand < i.reorder_point * 1.5    THEN 'LOW'
        ELSE                                                     'HEALTHY'
    END                                                     AS stock_status
FROM inventory i
LEFT JOIN suppliers s ON i.supplier_id = s.supplier_id