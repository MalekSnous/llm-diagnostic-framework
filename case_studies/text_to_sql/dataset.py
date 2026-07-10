"""Text-to-SQL dataset: natural-language questions over a small e-commerce schema.

Each case is ``{"question", "gold_sql", "difficulty"}``. Scoring is by
**execution accuracy**: the model's SQL and the gold SQL are both executed on
the same deterministic in-memory SQLite database (``build_db``) and their
result sets compared (order-insensitive, floats rounded to 2 decimals). Unlike
text metrics, this cannot be inflated by verbosity or formatting tricks — the
query either computes the right answer or it doesn't.

Difficulty tiers (increasing):
- easy   : single-table SELECT with a WHERE filter or a simple aggregate.
- medium : GROUP BY / ORDER BY / LIMIT / HAVING on one table.
- hard   : joins across 2-4 tables, aggregation over joins, subqueries.
- expert : anti-joins ("never ordered"), relational division, nested
           aggregates, per-group maxima, ranking (2nd highest).

The seed data is hand-designed so that every gold query returns a non-empty
result and superlative questions ("most", "highest") have a unique answer —
no ties, so execution comparison is deterministic.
"""

import sqlite3

SCHEMA_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    city TEXT,                       -- may be NULL
    signup_date TEXT NOT NULL        -- ISO format YYYY-MM-DD
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,          -- 'electronics' | 'books' | 'home' | 'sports'
    price REAL NOT NULL,             -- current list price
    stock INTEGER NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,        -- ISO format YYYY-MM-DD
    status TEXT NOT NULL             -- 'delivered' | 'shipped' | 'pending' | 'cancelled'
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL         -- price actually paid (may be discounted)
);
"""

CUSTOMERS = [
    (1, "Alice Martin", "France", "Paris", "2024-01-15"),
    (2, "Bruno Keller", "Germany", "Berlin", "2024-02-03"),
    (3, "Carla Lopez", "Spain", "Madrid", "2024-02-20"),
    (4, "David Chen", "USA", "San Francisco", "2024-03-05"),
    (5, "Emma Dubois", "France", "Lyon", "2024-03-18"),
    (6, "Farid Haddad", "France", None, "2024-04-02"),
    (7, "Greta Schmidt", "Germany", "Munich", "2024-04-25"),
    (8, "Hugo Alvarez", "Spain", None, "2024-05-10"),
    (9, "Iris Wong", "USA", "New York", "2024-05-30"),
    (10, "Jonas Berg", "Germany", "Hamburg", "2024-06-12"),
    (11, "Karim Ben Ali", "France", "Marseille", "2024-06-28"),
    (12, "Lena Novak", "USA", "Chicago", "2024-07-04"),
]

PRODUCTS = [
    (1, "Laptop Pro 15", "electronics", 1299.00, 12),
    (2, "Wireless Mouse", "electronics", 29.90, 150),
    (3, "Mechanical Keyboard", "electronics", 89.50, 60),
    (4, "Noise-Cancelling Headphones", "electronics", 199.00, 35),
    (5, "SQL for Humans", "books", 39.00, 80),
    (6, "Deep Learning Basics", "books", 59.00, 45),
    (7, "Espresso Machine", "home", 249.00, 20),
    (8, "Air Purifier", "home", 149.00, 25),
    (9, "Yoga Mat", "sports", 24.50, 100),
    (10, "Running Shoes", "sports", 119.00, 40),
    (11, "Smart Watch", "electronics", 219.00, 0),
    (12, "Camping Tent", "sports", 179.00, 15),
]

ORDERS = [
    (1, 1, "2024-03-01", "delivered"),
    (2, 2, "2024-03-15", "delivered"),
    (3, 1, "2024-04-02", "delivered"),
    (4, 3, "2024-04-10", "cancelled"),
    (5, 4, "2024-04-22", "delivered"),
    (6, 5, "2024-05-01", "shipped"),
    (7, 2, "2024-05-05", "delivered"),
    (8, 1, "2024-05-20", "delivered"),
    (9, 6, "2024-06-01", "delivered"),
    (10, 7, "2024-06-08", "cancelled"),
    (11, 9, "2024-06-15", "delivered"),
    (12, 3, "2024-06-21", "delivered"),
    (13, 10, "2024-07-02", "shipped"),
    (14, 5, "2024-07-11", "delivered"),
    (15, 1, "2024-07-19", "pending"),
    (16, 11, "2024-08-01", "delivered"),
    (17, 7, "2024-08-09", "delivered"),
    (18, 9, "2024-08-15", "delivered"),
    (19, 4, "2024-09-01", "pending"),
    (20, 2, "2024-09-05", "delivered"),
]

ORDER_ITEMS = [
    (1, 2, 1, 29.90),
    (1, 5, 1, 39.00),
    (2, 1, 1, 1299.00),
    (3, 4, 1, 199.00),
    (3, 2, 2, 29.90),
    (4, 7, 1, 249.00),
    (5, 6, 2, 59.00),
    (5, 5, 1, 39.00),
    (6, 9, 2, 24.50),
    (6, 10, 1, 119.00),
    (7, 3, 1, 89.50),
    (7, 2, 1, 27.90),  # discounted
    (8, 1, 1, 1249.00),  # discounted
    (9, 8, 1, 149.00),
    (9, 9, 1, 24.50),
    (10, 11, 1, 219.00),
    (11, 4, 1, 199.00),
    (11, 6, 1, 59.00),
    (11, 2, 1, 29.90),
    (12, 10, 1, 119.00),
    (13, 7, 1, 249.00),
    (13, 2, 1, 29.90),
    (14, 5, 3, 39.00),
    (15, 3, 1, 89.50),
    (16, 9, 1, 24.50),
    (16, 5, 1, 39.00),
    (16, 2, 1, 29.90),
    (17, 6, 1, 59.00),
    (18, 1, 1, 1299.00),
    (18, 4, 1, 199.00),
    (19, 10, 2, 119.00),
    (20, 8, 1, 149.00),
    (20, 3, 1, 89.50),
]


def build_db() -> sqlite3.Connection:
    """Create and seed a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", ORDER_ITEMS)
    conn.commit()
    return conn


CASES = [
    # ----------------------------------------------------------------- easy
    {
        "question": "How many customers are there in total?",
        "gold_sql": "SELECT COUNT(*) FROM customers",
        "difficulty": "easy",
    },
    {
        "question": "List the names of all customers from France.",
        "gold_sql": "SELECT name FROM customers WHERE country = 'France'",
        "difficulty": "easy",
    },
    {
        "question": "How many products are in the 'electronics' category?",
        "gold_sql": "SELECT COUNT(*) FROM products WHERE category = 'electronics'",
        "difficulty": "easy",
    },
    {
        "question": "What is the price of the product named 'Espresso Machine'?",
        "gold_sql": "SELECT price FROM products WHERE name = 'Espresso Machine'",
        "difficulty": "easy",
    },
    {
        "question": "List the names of all products in the 'books' category.",
        "gold_sql": "SELECT name FROM products WHERE category = 'books'",
        "difficulty": "easy",
    },
    {
        "question": "How many orders have the status 'cancelled'?",
        "gold_sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
        "difficulty": "easy",
    },
    {
        "question": "List the names of customers who live in Germany.",
        "gold_sql": "SELECT name FROM customers WHERE country = 'Germany'",
        "difficulty": "easy",
    },
    {
        "question": "What is the stock quantity of the product 'Smart Watch'?",
        "gold_sql": "SELECT stock FROM products WHERE name = 'Smart Watch'",
        "difficulty": "easy",
    },
    {
        "question": "How many products cost more than 100?",
        "gold_sql": "SELECT COUNT(*) FROM products WHERE price > 100",
        "difficulty": "easy",
    },
    {
        "question": "List the names of products that are out of stock (stock equal to 0).",
        "gold_sql": "SELECT name FROM products WHERE stock = 0",
        "difficulty": "easy",
    },
    {
        "question": "How many customers signed up after June 1st, 2024?",
        "gold_sql": "SELECT COUNT(*) FROM customers WHERE signup_date > '2024-06-01'",
        "difficulty": "easy",
    },
    {
        "question": "List the names and prices of all products in the 'sports' category.",
        "gold_sql": "SELECT name, price FROM products WHERE category = 'sports'",
        "difficulty": "easy",
    },
    {
        "question": "How many orders were placed in total?",
        "gold_sql": "SELECT COUNT(*) FROM orders",
        "difficulty": "easy",
    },
    {
        "question": "What is the lowest product price?",
        "gold_sql": "SELECT MIN(price) FROM products",
        "difficulty": "easy",
    },
    {
        "question": "What is the average price of all products?",
        "gold_sql": "SELECT AVG(price) FROM products",
        "difficulty": "easy",
    },
    {
        "question": "List the distinct countries that customers come from.",
        "gold_sql": "SELECT DISTINCT country FROM customers",
        "difficulty": "easy",
    },
    {
        "question": "How many customers are from the USA?",
        "gold_sql": "SELECT COUNT(*) FROM customers WHERE country = 'USA'",
        "difficulty": "easy",
    },
    {
        "question": "List the names of products with a price under 50.",
        "gold_sql": "SELECT name FROM products WHERE price < 50",
        "difficulty": "easy",
    },
    {
        "question": "How many orders have the status 'delivered'?",
        "gold_sql": "SELECT COUNT(*) FROM orders WHERE status = 'delivered'",
        "difficulty": "easy",
    },
    {
        "question": "What is the total stock across all products?",
        "gold_sql": "SELECT SUM(stock) FROM products",
        "difficulty": "easy",
    },
    {
        "question": "What is the highest product price?",
        "gold_sql": "SELECT MAX(price) FROM products",
        "difficulty": "easy",
    },
    {
        "question": "List the ids and order dates of orders with status 'pending'.",
        "gold_sql": "SELECT id, order_date FROM orders WHERE status = 'pending'",
        "difficulty": "easy",
    },
    {
        "question": "How many distinct customers have placed at least one order?",
        "gold_sql": "SELECT COUNT(DISTINCT customer_id) FROM orders",
        "difficulty": "easy",
    },
    {
        "question": "List the names and stock of products with stock below 30.",
        "gold_sql": "SELECT name, stock FROM products WHERE stock < 30",
        "difficulty": "easy",
    },
    {
        "question": "What is the name of the customer with id 5?",
        "gold_sql": "SELECT name FROM customers WHERE id = 5",
        "difficulty": "easy",
    },
    # ----------------------------------------------------------------- medium
    {
        "question": "What is the name of the most expensive product?",
        "gold_sql": "SELECT name FROM products ORDER BY price DESC LIMIT 1",
        "difficulty": "medium",
    },
    {
        "question": "List the names and prices of the 3 cheapest products.",
        "gold_sql": "SELECT name, price FROM products ORDER BY price ASC LIMIT 3",
        "difficulty": "medium",
    },
    {
        "question": "How many products are there in each category? Return the category and the count.",
        "gold_sql": "SELECT category, COUNT(*) FROM products GROUP BY category",
        "difficulty": "medium",
    },
    {
        "question": "What is the average product price per category? Return the category and the average price.",
        "gold_sql": "SELECT category, AVG(price) FROM products GROUP BY category",
        "difficulty": "medium",
    },
    {
        "question": "List the names of customers who have no city recorded.",
        "gold_sql": "SELECT name FROM customers WHERE city IS NULL",
        "difficulty": "medium",
    },
    {
        "question": "How many orders did each customer place? Return the customer id and the number of orders.",
        "gold_sql": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        "difficulty": "medium",
    },
    {
        "question": "How many orders were placed in each month? Return the month as 'YYYY-MM' and the count.",
        "gold_sql": "SELECT substr(order_date, 1, 7), COUNT(*) FROM orders GROUP BY substr(order_date, 1, 7)",
        "difficulty": "medium",
    },
    {
        "question": "List the names of customers who signed up in March or April 2024.",
        "gold_sql": "SELECT name FROM customers WHERE signup_date >= '2024-03-01' AND signup_date < '2024-05-01'",
        "difficulty": "medium",
    },
    {
        "question": "What is the total quantity of items across all order items?",
        "gold_sql": "SELECT SUM(quantity) FROM order_items",
        "difficulty": "medium",
    },
    {
        "question": "Which product id appears in the most order items? Return only the product id.",
        "gold_sql": "SELECT product_id FROM order_items GROUP BY product_id ORDER BY COUNT(*) DESC LIMIT 1",
        "difficulty": "medium",
    },
    {
        "question": "List the countries that have more than 2 customers.",
        "gold_sql": "SELECT country FROM customers GROUP BY country HAVING COUNT(*) > 2",
        "difficulty": "medium",
    },
    {
        "question": "What is the total revenue across all order items (quantity times unit price)?",
        "gold_sql": "SELECT SUM(quantity * unit_price) FROM order_items",
        "difficulty": "medium",
    },
    {
        "question": "List the ids of orders that contain more than one order item.",
        "gold_sql": "SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(*) > 1",
        "difficulty": "medium",
    },
    {
        "question": "What are the names of the 2 most expensive products in the 'electronics' category?",
        "gold_sql": "SELECT name FROM products WHERE category = 'electronics' ORDER BY price DESC LIMIT 2",
        "difficulty": "medium",
    },
    {
        "question": "How many customers signed up in each month? Return the month as 'YYYY-MM' and the count.",
        "gold_sql": "SELECT substr(signup_date, 1, 7), COUNT(*) FROM customers GROUP BY substr(signup_date, 1, 7)",
        "difficulty": "medium",
    },
    {
        "question": "For each order status, how many orders have that status? Return the status and the count.",
        "gold_sql": "SELECT status, COUNT(*) FROM orders GROUP BY status",
        "difficulty": "medium",
    },
    {
        "question": "What is the maximum quantity in a single order item?",
        "gold_sql": "SELECT MAX(quantity) FROM order_items",
        "difficulty": "medium",
    },
    {
        "question": "List the first 5 product names in alphabetical order.",
        "gold_sql": "SELECT name FROM products ORDER BY name ASC LIMIT 5",
        "difficulty": "medium",
    },
    {
        "question": "What is the average stock of products in the 'electronics' category?",
        "gold_sql": "SELECT AVG(stock) FROM products WHERE category = 'electronics'",
        "difficulty": "medium",
    },
    {
        "question": "Which category has the most products? Return only the category.",
        "gold_sql": "SELECT category FROM products GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1",
        "difficulty": "medium",
    },
    {
        "question": "What is the earliest order date?",
        "gold_sql": "SELECT MIN(order_date) FROM orders",
        "difficulty": "medium",
    },
    {
        "question": "List the names of customers from France ordered by signup date, oldest first.",
        "gold_sql": "SELECT name FROM customers WHERE country = 'France' ORDER BY signup_date ASC",
        "difficulty": "medium",
    },
    {
        "question": "What is the total quantity of items in order 3?",
        "gold_sql": "SELECT SUM(quantity) FROM order_items WHERE order_id = 3",
        "difficulty": "medium",
    },
    {
        "question": "What is the price difference between the most expensive and the cheapest product?",
        "gold_sql": "SELECT MAX(price) - MIN(price) FROM products",
        "difficulty": "medium",
    },
    {
        "question": "How many orders were placed between 2024-05-01 and 2024-07-31 inclusive?",
        "gold_sql": "SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2024-05-01' AND '2024-07-31'",
        "difficulty": "medium",
    },
    # ----------------------------------------------------------------- hard
    {
        "question": "List the names of customers who placed at least 2 orders, with their number of orders.",
        "gold_sql": (
            "SELECT c.name, COUNT(*) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id HAVING COUNT(*) >= 2"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the total amount spent by each customer? Return the customer name and the total (quantity times unit price).",
        "gold_sql": (
            "SELECT c.name, SUM(oi.quantity * oi.unit_price) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id"
        ),
        "difficulty": "hard",
    },
    {
        "question": "Which product generated the highest total revenue? Return only the product name.",
        "gold_sql": (
            "SELECT p.name FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "GROUP BY p.id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the names of products that appear in at least 3 order items.",
        "gold_sql": (
            "SELECT p.name FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "GROUP BY p.id HAVING COUNT(*) >= 3"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the name of the customer who placed the most orders?",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id ORDER BY COUNT(*) DESC LIMIT 1"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the distinct names of customers who ordered the product 'Laptop Pro 15'.",
        "gold_sql": (
            "SELECT DISTINCT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "JOIN products p ON p.id = oi.product_id "
            "WHERE p.name = 'Laptop Pro 15'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "How many distinct orders contain at least one product from the 'books' category?",
        "gold_sql": (
            "SELECT COUNT(DISTINCT oi.order_id) FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id "
            "WHERE p.category = 'books'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the total revenue from cancelled orders?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
            "JOIN orders o ON o.id = oi.order_id "
            "WHERE o.status = 'cancelled'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "For each country, what is the total revenue from its customers? Return the country and the total.",
        "gold_sql": (
            "SELECT c.country, SUM(oi.quantity * oi.unit_price) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.country"
        ),
        "difficulty": "hard",
    },
    {
        "question": "Which customer spent the most money in total? Return only the customer name.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the names of products that have never been ordered.",
        "gold_sql": (
            "SELECT name FROM products "
            "WHERE id NOT IN (SELECT DISTINCT product_id FROM order_items)"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the average order value (the average over orders of each order's total quantity times unit price)?",
        "gold_sql": (
            "SELECT AVG(order_total) FROM ("
            "SELECT SUM(quantity * unit_price) AS order_total FROM order_items GROUP BY order_id)"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the distinct names of customers who have a pending order.",
        "gold_sql": (
            "SELECT DISTINCT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id WHERE o.status = 'pending'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "Which product category generated the most total revenue? Return only the category.",
        "gold_sql": (
            "SELECT p.category FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "GROUP BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the ids of orders whose total value (quantity times unit price) exceeds 200.",
        "gold_sql": (
            "SELECT order_id FROM order_items "
            "GROUP BY order_id HAVING SUM(quantity * unit_price) > 200"
        ),
        "difficulty": "hard",
    },
    {
        "question": "How many distinct products has the customer 'Alice Martin' ordered?",
        "gold_sql": (
            "SELECT COUNT(DISTINCT oi.product_id) FROM order_items oi "
            "JOIN orders o ON o.id = oi.order_id "
            "JOIN customers c ON c.id = o.customer_id "
            "WHERE c.name = 'Alice Martin'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the distinct names of German customers who placed at least one delivered order.",
        "gold_sql": (
            "SELECT DISTINCT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "WHERE c.country = 'Germany' AND o.status = 'delivered'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "For each customer who has ordered, what is their most recent order date? Return the customer name and the date.",
        "gold_sql": (
            "SELECT c.name, MAX(o.order_date) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id GROUP BY c.id"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the distinct names of products that were ordered with a quantity of 2 or more in a single order item.",
        "gold_sql": (
            "SELECT DISTINCT p.name FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id WHERE oi.quantity >= 2"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the total quantity sold for each product? Return the product name and the total quantity.",
        "gold_sql": (
            "SELECT p.name, SUM(oi.quantity) FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id GROUP BY p.id"
        ),
        "difficulty": "hard",
    },
    {
        "question": "Which order has the highest total value (quantity times unit price)? Return only the order id.",
        "gold_sql": (
            "SELECT order_id FROM order_items "
            "GROUP BY order_id ORDER BY SUM(quantity * unit_price) DESC LIMIT 1"
        ),
        "difficulty": "hard",
    },
    {
        "question": "For customers with at least one delivered order, how many delivered orders does each have? Return the customer name and the count.",
        "gold_sql": (
            "SELECT c.name, COUNT(*) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "WHERE o.status = 'delivered' GROUP BY c.id"
        ),
        "difficulty": "hard",
    },
    {
        "question": "What is the total revenue from orders placed in June 2024?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
            "JOIN orders o ON o.id = oi.order_id "
            "WHERE o.order_date >= '2024-06-01' AND o.order_date < '2024-07-01'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "List the distinct names of products ordered by customers from the USA.",
        "gold_sql": (
            "SELECT DISTINCT p.name FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "JOIN orders o ON o.id = oi.order_id "
            "JOIN customers c ON c.id = o.customer_id "
            "WHERE c.country = 'USA'"
        ),
        "difficulty": "hard",
    },
    {
        "question": "How many orders include products from more than one category?",
        "gold_sql": (
            "SELECT COUNT(*) FROM ("
            "SELECT oi.order_id FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id "
            "GROUP BY oi.order_id HAVING COUNT(DISTINCT p.category) > 1)"
        ),
        "difficulty": "hard",
    },
    # ----------------------------------------------------------------- expert
    {
        "question": "List the names of customers who never placed an order.",
        "gold_sql": (
            "SELECT name FROM customers "
            "WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What is the name of the second most expensive product?",
        "gold_sql": "SELECT name FROM products ORDER BY price DESC LIMIT 1 OFFSET 1",
        "difficulty": "expert",
    },
    {
        "question": "For each category, what is the name of its most expensive product? Return the category and the product name.",
        "gold_sql": (
            "SELECT category, name FROM products p "
            "WHERE price = (SELECT MAX(price) FROM products p2 WHERE p2.category = p.category)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of customers whose total spend is higher than the average total spend per ordering customer.",
        "gold_sql": (
            "WITH spend AS ("
            "SELECT c.id AS cid, c.name AS name, SUM(oi.quantity * oi.unit_price) AS total "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id GROUP BY c.id) "
            "SELECT name FROM spend WHERE total > (SELECT AVG(total) FROM spend)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What percentage of all orders are cancelled? Return a single number between 0 and 100.",
        "gold_sql": (
            "SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) "
            "FROM orders"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Which month had the highest total revenue? Return only the month as 'YYYY-MM'.",
        "gold_sql": (
            "SELECT substr(o.order_date, 1, 7) AS month FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY month ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of customers who ordered products from more than one category.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "JOIN products p ON p.id = oi.product_id "
            "GROUP BY c.id HAVING COUNT(DISTINCT p.category) > 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Which customer has the highest average order value (average of their orders' totals)? Return only the customer name.",
        "gold_sql": (
            "WITH totals AS ("
            "SELECT o.customer_id AS cid, SUM(oi.quantity * oi.unit_price) AS t "
            "FROM orders o JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id) "
            "SELECT c.name FROM customers c JOIN totals ON totals.cid = c.id "
            "GROUP BY c.id ORDER BY AVG(totals.t) DESC LIMIT 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of products that were never ordered by any French customer.",
        "gold_sql": (
            "SELECT name FROM products WHERE id NOT IN ("
            "SELECT oi.product_id FROM order_items oi "
            "JOIN orders o ON o.id = oi.order_id "
            "JOIN customers c ON c.id = o.customer_id "
            "WHERE c.country = 'France')"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Which customers placed the maximum number of orders? Return the name of every customer tied for the most orders.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id HAVING COUNT(*) = ("
            "SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM orders GROUP BY customer_id))"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What is the total revenue excluding cancelled orders?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
            "JOIN orders o ON o.id = oi.order_id WHERE o.status != 'cancelled'"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What is the average order value of delivered orders only?",
        "gold_sql": (
            "SELECT AVG(t) FROM ("
            "SELECT SUM(oi.quantity * oi.unit_price) AS t FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.id "
            "WHERE o.status = 'delivered' GROUP BY o.id)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Which country's ordering customers have the highest average total spend per customer? Return only the country.",
        "gold_sql": (
            "WITH spend AS ("
            "SELECT c.country AS country, c.id AS cid, SUM(oi.quantity * oi.unit_price) AS total "
            "FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id GROUP BY c.id) "
            "SELECT country FROM spend GROUP BY country ORDER BY AVG(total) DESC LIMIT 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of customers who ordered both 'electronics' products and 'books' products.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "JOIN products p ON p.id = oi.product_id "
            "GROUP BY c.id "
            "HAVING SUM(CASE WHEN p.category = 'electronics' THEN 1 ELSE 0 END) > 0 "
            "AND SUM(CASE WHEN p.category = 'books' THEN 1 ELSE 0 END) > 0"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Which product was ordered by the largest number of distinct customers? Return only the product name.",
        "gold_sql": (
            "SELECT p.name FROM products p "
            "JOIN order_items oi ON oi.product_id = p.id "
            "JOIN orders o ON o.id = oi.order_id "
            "GROUP BY p.id ORDER BY COUNT(DISTINCT o.customer_id) DESC LIMIT 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "For each order status, what is the total value of orders with that status? Return the status and the total (quantity times unit price).",
        "gold_sql": (
            "SELECT o.status, SUM(oi.quantity * oi.unit_price) FROM orders o "
            "JOIN order_items oi ON oi.order_id = o.id GROUP BY o.status"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the ids of orders in which every item is an 'electronics' product.",
        "gold_sql": (
            "SELECT oi.order_id FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id "
            "GROUP BY oi.order_id "
            "HAVING COUNT(*) = SUM(CASE WHEN p.category = 'electronics' THEN 1 ELSE 0 END)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "Among customers who placed at least one order, who signed up most recently? Return only the customer name.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "WHERE c.id IN (SELECT DISTINCT customer_id FROM orders) "
            "ORDER BY c.signup_date DESC LIMIT 1"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of the top 2 customers by total spend (quantity times unit price).",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 2"
        ),
        "difficulty": "expert",
    },
    {
        "question": "How many customers placed orders in more than one distinct month?",
        "gold_sql": (
            "SELECT COUNT(*) FROM ("
            "SELECT customer_id FROM orders "
            "GROUP BY customer_id HAVING COUNT(DISTINCT substr(order_date, 1, 7)) > 1)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of products priced higher than every product in the 'books' category.",
        "gold_sql": (
            "SELECT name FROM products "
            "WHERE price > (SELECT MAX(price) FROM products WHERE category = 'books')"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What is the total discount given across all order items, where the discount is the product's list price minus the unit price actually paid, times the quantity?",
        "gold_sql": (
            "SELECT SUM((p.price - oi.unit_price) * oi.quantity) FROM order_items oi "
            "JOIN products p ON p.id = oi.product_id "
            "WHERE oi.unit_price < p.price"
        ),
        "difficulty": "expert",
    },
    {
        "question": "List the names of customers who have at least one order and whose orders were all delivered.",
        "gold_sql": (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id "
            "HAVING COUNT(*) = SUM(CASE WHEN o.status = 'delivered' THEN 1 ELSE 0 END)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "For each month with orders, how many distinct customers placed an order? Return the month as 'YYYY-MM' and the count.",
        "gold_sql": (
            "SELECT substr(order_date, 1, 7), COUNT(DISTINCT customer_id) "
            "FROM orders GROUP BY substr(order_date, 1, 7)"
        ),
        "difficulty": "expert",
    },
    {
        "question": "What is the second highest order total (quantity times unit price summed per order)?",
        "gold_sql": (
            "SELECT SUM(quantity * unit_price) AS t FROM order_items "
            "GROUP BY order_id ORDER BY t DESC LIMIT 1 OFFSET 1"
        ),
        "difficulty": "expert",
    },
]
