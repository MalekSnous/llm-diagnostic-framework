# Case Study: Text-to-SQL with Execution Accuracy

**Question:** how well do models turn natural language into *correct* SQL — and
does few-shot prompting move the needle?

The other two studies score text (fuzzy F1, substring QA). This one closes the
loop with a metric that cannot be gamed: the model's query is **executed** on a
seeded SQLite database and its result set compared to the gold query's result
set. Verbosity, markdown fences, column aliases, row order — none of it matters.
The query either computes the right answer or it doesn't. This is the logical
endpoint of the framework's "beware metric artifacts" lesson.

## The task

~100 natural-language questions over a small e-commerce schema (`customers`,
`products`, `orders`, `order_items` — see [`dataset.py`](dataset.py)), in four
difficulty tiers:

| Tier | What it stresses |
|------|-----------------|
| easy | single-table SELECT + WHERE, simple aggregates |
| medium | GROUP BY / ORDER BY / LIMIT / HAVING |
| hard | multi-table joins, aggregation over joins, subqueries |
| expert | anti-joins ("never ordered"), relational division, nested aggregates, ranking |

The seed data is hand-designed so every gold query returns a non-empty result
and every superlative ("most", "highest") has a unique answer — no ties, so
execution comparison is deterministic.

## What it does

1. **Baseline** — zero-shot: schema + question, "return only the SQL".
2. **Prompt engineering** — few-shot with task-specific examples that teach the
   output contract (bare SQL, only the requested columns) and classic patterns
   (anti-join, aggregate-over-join).
3. Score both by **execution accuracy** (order-insensitive row comparison,
   floats rounded to 2 decimals) plus a **valid-SQL rate**, with a
   per-difficulty breakdown, then save a JSON + HTML report under
   `results/case_studies/`.

RAG is deliberately not tested: the whole schema fits in the prompt, so there
is nothing to retrieve.

## Run it

```bash
make run-sql-study                 # defaults to gpt-4o-mini
make run-sql-study model=gpt-4o    # or any supported model

# Cross-model comparison after several runs
python scripts/compare_models.py --study text_to_sql
```

The dataset and scorer are fully offline (stdlib `sqlite3`); only the model
calls need an API key.
