# 📊 Results & Methodology

## Reproducibility note (read first)

An earlier version of this document reported specific accuracy/cost figures
(e.g. "Phi-2 100%", "GPT-4o $4.06"). Those numbers came from two defects that
have since been fixed:

1. **Recall-only substring scoring** for entity extraction. It counted an expected
   entity as "found" if its text appeared anywhere in the raw output, with **no
   precision penalty**. A verbose/echoing model (Phi-2) therefore scored ~100%
   simply by producing lots of text — an artifact, not real quality.
2. **Token-pricing bug** (~1000×). Per-1M provider prices were stored in a table
   whose math is per-1K, inflating every OpenAI cost by ~1000× (hence the
   impossible "$4.06 for 8 cases").

Both are fixed and unit-tested (`tests/test_evaluator.py`, `tests/test_llm_client.py`).
The fabricated figures were **removed rather than shown misleadingly**. Re-run the
studies to produce numbers you can trust.

---

## Research question

**Does optimizing an LLM with prompt engineering or RAG always improve performance,
and is it worth the cost?**

Answer, from running the harness: **no — it's task-dependent**. The framework
exists precisely to measure this per task instead of assuming.

## Methodology

| Aspect | Choice |
|--------|--------|
| Task A | Medical entity extraction (conditions, meds, procedures) |
| Task B | Document QA over a private corpus (`data/product_docs.txt`) |
| Strategies | Baseline (zero-shot) · Prompt engineering (few-shot) · RAG |
| Entity metric | **Precision / Recall / F1** with containment matching (`Evaluator.fuzzy_entity_metrics`) — verbosity-robust |
| QA metric | Gold-answer containment |
| Cost | Real token cost (OpenAI per-1K, Anthropic per-1M); local = $0 |

> Sample size in the bundled studies is small (8 cases). Treat outputs as a
> **demonstration of the methodology**, not a benchmark leaderboard — increase
> `num_cases` and add repetitions for statistical claims.

## What to look for when you run it

- **Beat the baseline, don't assume it.** On a model already strong at the task,
  few-shot/RAG often *reduce* F1 by adding noise (medical study).
- **RAG is conditional.** It wins when the answer is outside the model's
  knowledge (document-QA study) and hurts when it just dilutes the prompt.
- **Cost per quality point** is the decision metric, not raw F1 or raw price.
- **Sanity-check the metric.** If a weak local model "wins", suspect the metric
  before believing the result — that is exactly the bug this project caught.

## Reproduce

```bash
# Hosted model (needs OPENAI_API_KEY)
python case_studies/medical_entity_extraction/run_study.py --model gpt-4o-mini

# Local model (needs the [local] extra; no API key)
python case_studies/medical_entity_extraction/run_study.py --model microsoft/phi-2

# RAG document QA (needs the [rag] extra)
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini
```

Each run writes a JSON + interactive HTML report to `results/case_studies/`.
