-- Stage 4: features of the early-warning model, built in the database (D77).
--
-- Run by the `features` stage with search_path = <ews schema>, <olist schema>: objects are
-- created in the ews schema, Olist tables are read from the olist schema. Views keep the
-- tables they were built on, so later readers need no search_path.
--
-- THE RULE: a feature may only use what existed at the decision moment, the approval of the
-- order (D75). Every column below is listed with its moment in the temporal audit
-- (src/data_factory/ews_audit.py); a test fails if a column is missing there (D82).
--   * order facts written at purchase: items, prices, freight, payments, promised date;
--   * lookups that do not change: product, seller, customer, zip code, holiday calendar;
--   * history: only events strictly BEFORE the decision moment, with the moment each event
--     became known (a review is known when the customer answered it: answered_at, not the
--     day the survey was created; a delivery when delivered_customer_at happened).
-- Delivery dates, the final status, review scores of this order and the generated
-- delivery flags of olist.orders are never read for this order: they describe its future.
DROP PROCEDURE IF EXISTS refresh_features();
DROP TABLE IF EXISTS order_features CASCADE;
DROP VIEW IF EXISTS order_features_live, cheat_columns, order_labels, decision_orders CASCADE;

-- One row per order: when the decision is taken. 160 orders have no approval time: their
-- decision moment is the purchase, the earliest moment the order exists (D79).
CREATE VIEW decision_orders AS
SELECT order_id,
       customer_id,
       purchased_at,
       coalesce(approved_at, purchased_at) AS decision_at,
       approved_at IS NULL                 AS approval_missing,
       estimated_delivery_date
FROM orders;

-- The label and the moment it became known (label maturity, D76). Several reviews of one
-- order: negative if any of them is negative (score 1 or 2); known when the first negative
-- answer arrived, otherwise when the last answer arrived. Orders without a review have no row.
CREATE VIEW order_labels AS
SELECT order_id,
       min(score) <= 2 AS negative,
       CASE WHEN min(score) <= 2 THEN min(answered_at) FILTER (WHERE score <= 2)
            ELSE max(answered_at) END AS label_known_at
FROM order_reviews
GROUP BY order_id;

CREATE VIEW order_features_live AS
WITH item_rows AS (
    SELECT i.order_id, i.seller_id, i.product_id, i.price, i.freight_value, i.shipping_limit_at,
           p.weight_g, p.photos_qty, p.description_length,
           p.length_cm::integer * p.height_cm * p.width_cm AS volume_cm3,
           coalesce(c.category_name_english, p.category_name, 'unknown') AS category
    FROM order_items i
    JOIN products p USING (product_id)
    LEFT JOIN product_categories c ON c.category_name = p.category_name
),
items AS (
    SELECT order_id,
           count(*)                                   AS n_items,
           count(DISTINCT product_id)                 AS n_products,
           count(DISTINCT seller_id)                  AS n_sellers,
           sum(price)::double precision               AS total_price,
           sum(freight_value)::double precision       AS total_freight,
           max(shipping_limit_at)                     AS shipping_limit_at,
           sum(weight_g)::double precision            AS total_weight_g,
           max(volume_cm3)::double precision          AS max_volume_cm3,
           min(photos_qty)                            AS min_photos,
           min(description_length)                    AS min_description_length,
           mode() WITHIN GROUP (ORDER BY category)    AS main_category
    FROM item_rows
    GROUP BY order_id
),
main_seller AS (                      -- the seller with the largest share of the order
    SELECT DISTINCT ON (order_id) order_id, seller_id
    FROM (SELECT order_id, seller_id, sum(price) AS value FROM item_rows GROUP BY 1, 2) s
    ORDER BY order_id, value DESC, seller_id
),
pay AS (
    SELECT order_id,
           count(*)                                    AS n_payments,
           max(installments)                           AS max_installments,
           sum(payment_value)::double precision        AS total_paid,
           bool_or(payment_type = 'voucher')           AS uses_voucher,
           bool_or(payment_type = 'boleto')            AS uses_boleto
    FROM order_payments
    GROUP BY order_id
),
base AS (
    SELECT d.*, cu.customer_unique_id, cu.state AS customer_state, ms.seller_id, se.state AS seller_state,
           it.main_category,
           zc.latitude AS c_lat, zc.longitude AS c_lng, zs.latitude AS s_lat, zs.longitude AS s_lng
    FROM decision_orders d
    LEFT JOIN customers cu USING (customer_id)    -- a new order may come without its customer row
    LEFT JOIN items it USING (order_id)
    LEFT JOIN main_seller ms USING (order_id)
    LEFT JOIN sellers se USING (seller_id)
    LEFT JOIN zip_codes zc ON zc.zip_code_prefix = cu.zip_code_prefix
    LEFT JOIN zip_codes zs ON zs.zip_code_prefix = se.zip_code_prefix
),
-- History as one stream of events per seller / customer / category: the order being decided
-- (is_obs = 1, at its decision moment) and the facts of other orders at the moment they became
-- known. The window ends 1 microsecond before the row: only strictly earlier events count.
seller_stream AS (
    SELECT seller_id, decision_at AS ts, order_id, 1 AS is_obs, 0 AS rev, 0 AS neg, 0 AS dlv, 0 AS late
    FROM base WHERE seller_id IS NOT NULL
    UNION ALL
    SELECT DISTINCT i.seller_id, l.label_known_at, NULL::uuid, 0, 1, l.negative::int, 0, 0
    FROM order_items i JOIN order_labels l USING (order_id)
    UNION ALL
    SELECT DISTINCT i.seller_id, o.delivered_customer_at, NULL::uuid, 0, 0, 0, 1,
           (o.delivered_customer_at::date > o.estimated_delivery_date)::int
    FROM order_items i JOIN orders o USING (order_id)
    WHERE o.delivered_customer_at IS NOT NULL
),
seller_hist AS (
    SELECT order_id,
           sum(is_obs) OVER w AS seller_prior_orders,
           sum(rev)    OVER w AS seller_prior_reviews,
           sum(neg)    OVER w AS seller_prior_negatives,
           sum(dlv)    OVER w AS seller_prior_deliveries,
           sum(late)   OVER w AS seller_prior_late,
           min(ts) FILTER (WHERE is_obs = 1) OVER w AS seller_first_order_at,
           is_obs
    FROM seller_stream
    WINDOW w AS (PARTITION BY seller_id ORDER BY ts
                 RANGE BETWEEN UNBOUNDED PRECEDING AND '1 microsecond' PRECEDING)
),
customer_stream AS (
    SELECT customer_unique_id, decision_at AS ts, order_id, 1 AS is_obs, 0 AS neg
    FROM base
    UNION ALL
    SELECT cu.customer_unique_id, l.label_known_at, NULL::uuid, 0, l.negative::int
    FROM order_labels l JOIN orders o USING (order_id) JOIN customers cu USING (customer_id)
),
customer_hist AS (
    SELECT order_id,
           sum(is_obs) OVER w AS customer_prior_orders,
           sum(neg)    OVER w AS customer_prior_negatives,
           is_obs
    FROM customer_stream
    WINDOW w AS (PARTITION BY customer_unique_id ORDER BY ts
                 RANGE BETWEEN UNBOUNDED PRECEDING AND '1 microsecond' PRECEDING)
),
category_stream AS (
    SELECT main_category, decision_at AS ts, order_id, 1 AS is_obs, 0 AS rev, 0 AS neg
    FROM base WHERE main_category IS NOT NULL
    UNION ALL
    SELECT b.main_category, l.label_known_at, NULL::uuid, 0, 1, l.negative::int
    FROM order_labels l JOIN base b USING (order_id)
    WHERE b.main_category IS NOT NULL
),
category_hist AS (
    SELECT order_id,
           sum(rev) OVER w AS category_prior_reviews,
           sum(neg) OVER w AS category_prior_negatives,
           is_obs
    FROM category_stream
    WINDOW w AS (PARTITION BY main_category ORDER BY ts
                 RANGE BETWEEN UNBOUNDED PRECEDING AND '1 microsecond' PRECEDING)
),
calendar AS (                         -- working days and holidays inside the promised window
    SELECT b.order_id,
           count(*) FILTER (WHERE extract(isodow FROM day) < 6 AND h.holiday_date IS NULL) AS promised_working_days,
           count(h.holiday_date)                                                          AS holidays_in_window
    FROM base b
    CROSS JOIN LATERAL generate_series(b.decision_at::date + 1, b.estimated_delivery_date, interval '1 day') AS day
    LEFT JOIN (SELECT DISTINCT holiday_date FROM holidays) h ON h.holiday_date = day::date
    GROUP BY b.order_id
)
SELECT b.order_id,
       b.decision_at,
       -- order (written at purchase)
       b.approval_missing,
       extract(epoch FROM b.decision_at - b.purchased_at) / 3600.0      AS approval_hours,
       extract(hour FROM b.purchased_at)::int                           AS purchase_hour,
       extract(isodow FROM b.purchased_at)::int                         AS purchase_weekday,
       (b.estimated_delivery_date - b.decision_at::date)                AS promised_days,
       extract(epoch FROM it.shipping_limit_at - b.decision_at) / 86400.0 AS days_to_ship_limit,
       it.n_items, it.n_products, it.n_sellers,
       it.total_price, it.total_freight,
       it.total_freight / nullif(it.total_price + it.total_freight, 0)  AS freight_share,
       -- payment
       pay.n_payments, pay.max_installments, pay.uses_voucher, pay.uses_boleto,
       pay.total_paid - (it.total_price + it.total_freight)             AS paid_minus_order,
       -- product
       it.main_category, it.total_weight_g, it.max_volume_cm3, it.min_photos, it.min_description_length,
       -- customer and seller (geography)
       b.customer_state, b.seller_state,
       (b.customer_state = b.seller_state)                              AS same_state,
       2 * 6371 * asin(sqrt(power(sin(radians(b.c_lat - b.s_lat) / 2), 2)
           + cos(radians(b.s_lat)) * cos(radians(b.c_lat)) * power(sin(radians(b.c_lng - b.s_lng) / 2), 2)))
                                                                        AS distance_km,
       -- history, strictly before the decision moment
       sh.seller_prior_orders, sh.seller_prior_reviews,
       sh.seller_prior_negatives::double precision / nullif(sh.seller_prior_reviews, 0)  AS seller_prior_negative_rate,
       sh.seller_prior_late::double precision / nullif(sh.seller_prior_deliveries, 0)    AS seller_prior_late_rate,
       extract(epoch FROM b.decision_at - sh.seller_first_order_at) / 86400.0           AS seller_age_days,
       ch.customer_prior_orders, ch.customer_prior_negatives,
       kh.category_prior_negatives::double precision / nullif(kh.category_prior_reviews, 0) AS category_prior_negative_rate,
       -- external: holiday calendar (published in advance)
       cal.promised_working_days, cal.holidays_in_window,
       -- label, never a feature (D76): the model layer keeps it apart
       l.negative       AS label_negative,
       l.label_known_at AS label_known_at
FROM base b
LEFT JOIN items it USING (order_id)
LEFT JOIN pay USING (order_id)
LEFT JOIN seller_hist sh ON sh.order_id = b.order_id AND sh.is_obs = 1
LEFT JOIN customer_hist ch ON ch.order_id = b.order_id AND ch.is_obs = 1
LEFT JOIN category_hist kh ON kh.order_id = b.order_id AND kh.is_obs = 1
LEFT JOIN calendar cal ON cal.order_id = b.order_id
LEFT JOIN order_labels l ON l.order_id = b.order_id;

-- Snapshot (D78): the model reads this table, filled once per load by refresh_features().
CREATE TABLE order_features AS SELECT * FROM order_features_live WITH NO DATA;
ALTER TABLE order_features ADD PRIMARY KEY (order_id);

CREATE PROCEDURE refresh_features()
LANGUAGE sql
AS $$
    DELETE FROM order_features;
    INSERT INTO order_features SELECT * FROM order_features_live ORDER BY order_id;
$$;

-- AUDIT ONLY (D81): columns that look useful but are written after the decision. They are
-- read only to build the deliberate "cheat model" and measure how much leakage would flatter
-- the metric. The model layer refuses them (test_ews).
CREATE VIEW cheat_columns AS
SELECT o.order_id,
       o.status                                                          AS final_status,
       (o.delivered_customer_at::date - o.estimated_delivery_date)       AS days_late,
       extract(epoch FROM o.delivered_customer_at - o.purchased_at) / 86400.0 AS delivery_days,
       o.flag_status_delivery_mismatch,
       sn.seller_negative_rate_all_time
FROM orders o
LEFT JOIN (SELECT DISTINCT ON (i.order_id) i.order_id, s.rate AS seller_negative_rate_all_time
           FROM order_items i
           JOIN (SELECT i2.seller_id, avg(l.negative::int) AS rate
                 FROM order_items i2 JOIN order_labels l USING (order_id) GROUP BY 1) s USING (seller_id)
           ORDER BY i.order_id, i.order_item_id) sn USING (order_id);
