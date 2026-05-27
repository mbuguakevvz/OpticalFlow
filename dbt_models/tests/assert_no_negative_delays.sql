-- tests/assert_no_negative_delays.sql
-- This test PASSES if it returns 0 rows
-- It FAILS if any shipment has a negative delay

SELECT *
FROM {{ ref('stg_shipments') }}
WHERE delay_days < 0