# Case Study: Document QA with RAG

**Question:** does Retrieval-Augmented Generation actually help, and at what cost?

This study is the deliberate counterpoint to the [medical entity extraction
study](../medical_entity_extraction/), where generic RAG *degraded* an already
strong model. Here the questions are about **AcmeCloud**, a fictional internal
product whose facts (plan limits, token lifetimes, regions, SLAs, billing) appear
only in [`data/product_docs.txt`](../../data/product_docs.txt) and in **no model's
training data**.

A closed-book model therefore has to guess; a RAG-augmented model can retrieve the
exact answer. This is the canonical situation where retrieval pays for itself, and
the study quantifies the accuracy gain against the extra token cost.

## What it does

1. **Baseline** — ask each question with no context (closed-book).
2. **RAG** — index the product docs in a vector store (`RAGSystem`), retrieve the
   top-k relevant chunks, and answer with that context.
3. Score both with a case-insensitive substring match against accepted answers,
   then print a Rich comparison table and save a JSON + HTML report under
   `results/case_studies/`.

## Run it

```bash
# Requires the RAG extra: pip install -e ".[rag]"
make run-rag-study                 # defaults to gpt-4o-mini
make run-rag-study model=gpt-4o    # or any supported model

# Directly, skipping the report:
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini --no-report
```

API-based models need `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) in your environment.

## Expected result

Baseline accuracy is low (the model cannot know AcmeCloud-specific facts), while
RAG accuracy is high. The takeaway, paired with the medical study:

> **RAG helps when the answer lives outside the model's knowledge — not as a
> blanket "best practice." Always measure baseline first.**
