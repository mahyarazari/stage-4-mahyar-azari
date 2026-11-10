-- Prediction path (D84): new Olist rows, already checked by the contract and written to
-- temporary text tables pg_temp.new_<table>, are added to the Olist tables so that the same
-- feature view computes their features. Rows that already exist are left as they are.
-- The caller runs this inside a transaction and rolls it back: nothing stays.
INSERT INTO product_categories (category_name, category_name_english)
SELECT category_name, category_name_english FROM pg_temp.new_product_categories
ON CONFLICT DO NOTHING;

INSERT INTO sellers (seller_id, zip_code_prefix, city, state)
SELECT seller_id::uuid, zip_code_prefix, city, state FROM pg_temp.new_sellers
ON CONFLICT DO NOTHING;

INSERT INTO products (product_id, category_name, name_length, description_length, photos_qty, weight_g,
                      length_cm, height_cm, width_cm)
SELECT product_id::uuid, category_name, name_length::smallint, description_length::integer,
       photos_qty::smallint, weight_g::integer, length_cm::smallint, height_cm::smallint, width_cm::smallint
FROM pg_temp.new_products
ON CONFLICT DO NOTHING;

INSERT INTO customers (customer_id, customer_unique_id, zip_code_prefix, city, state)
SELECT customer_id::uuid, customer_unique_id::uuid, zip_code_prefix, city, state FROM pg_temp.new_customers
ON CONFLICT DO NOTHING;

INSERT INTO orders (order_id, customer_id, status, purchased_at, approved_at, delivered_carrier_at,
                    delivered_customer_at, estimated_delivery_date)
SELECT order_id::uuid, customer_id::uuid, status, purchased_at::timestamp, approved_at::timestamp,
       delivered_carrier_at::timestamp, delivered_customer_at::timestamp, estimated_delivery_date::date
FROM pg_temp.new_orders
ON CONFLICT DO NOTHING;

INSERT INTO order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_at, price, freight_value)
SELECT order_id::uuid, order_item_id::smallint, product_id::uuid, seller_id::uuid, shipping_limit_at::timestamp,
       price::numeric, freight_value::numeric
FROM pg_temp.new_order_items
ON CONFLICT DO NOTHING;

INSERT INTO order_payments (order_id, payment_sequential, payment_type, installments, payment_value)
SELECT order_id::uuid, payment_sequential::smallint, payment_type, installments::smallint, payment_value::numeric
FROM pg_temp.new_order_payments
ON CONFLICT DO NOTHING;
