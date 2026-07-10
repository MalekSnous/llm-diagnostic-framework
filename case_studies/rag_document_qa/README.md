# Case Study: Document QA with a full RAG pipeline

**Question:** does Retrieval-Augmented Generation actually help, how much of the
gain comes from each stage (retrieval vs reranking), and at what cost?

This study is the deliberate counterpoint to the [medical entity extraction
study](../medical_entity_extraction/), where generic RAG *degraded* an already
strong model. Here the questions are about **AcmeCloud**, a fictional internal
platform whose facts (plan limits, token lifetimes, SLAs, error codes, release
history) exist only in [`data/acmecloud_kb/`](../../data/acmecloud_kb/) — 12
markdown documents in **no model's training data**. A closed-book model has to
guess; a RAG-augmented model can retrieve the answer.

## The pipeline

The study runs on the modular pipeline in
[`llm_diagnostic/rag/`](../../llm_diagnostic/rag/):

```
ingestion → chunking → embedding → vector store → retrieval → reranking → answer
 (12 docs)  (paragraph-  (dense or   (in-memory     (top-12)    (cross-     (grounded
             aware, with  TF-IDF)     cosine)                    encoder)     prompt with
             overlap)                                                         citations +
                                                                              abstention)
```

Every stage is swappable: dense embeddings (`all-MiniLM-L6-v2`) when the `rag`
extra is installed, a TF-IDF embedder (scikit-learn, core install) otherwise, so
the whole study — and its tests — can run offline.

## The dataset

100 questions in `dataset.py`, 25 per difficulty tier:

| Tier | What it probes |
|---|---|
| easy | single fact, phrased close to the source wording |
| medium | single fact, paraphrased (lexical overlap drops) |
| hard | synthesis: plan comparisons, light arithmetic, multi-fact answers |
| expert | cross-document reasoning **+ 8 unanswerable questions** (the system must abstain, not hallucinate) |

Each answerable case carries its **gold source documents**, so the study reports
retrieval recall@k separately from answer accuracy — when RAG fails you can see
*which stage* failed. A test enforces that every non-derived accepted answer
literally occurs in its gold documents, so the dataset can't drift from the
corpus.

## The three arms

1. **Baseline (closed-book)** — the model answers from parametric memory alone.
2. **RAG** — top-4 chunks by embedding similarity, grounded prompt with source
   citations and an explicit abstention contract.
3. **RAG + rerank** — top-12 candidates, cross-encoder reranking
   (`ms-marco-MiniLM-L-6-v2`) down to 4.

Scoring is a boundary-aware match against accepted answers: purely numeric
answers use digit boundaries ("50" is not satisfied by "500" or "3.50"), so
verbosity can't inflate the score.

## Run it

```bash
make run-rag-study                                  # defaults to gpt-4o-mini
make run-rag-study model=gpt-4o                     # or any supported model

# Options:
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini \
    --embedder tfidf        # force the offline embedder (auto|dense|tfidf)
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini \
    --limit 20 --no-report  # quick smoke run, tier-balanced sample
```

Dense embeddings and the reranking arm need the extra: `pip install -e ".[rag]"`.
Without it the study still runs (TF-IDF, no rerank arm). API models need their
key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GROQ_API_KEY`) in `.env`.

Results land in `results/case_studies/` (JSON + HTML); the cross-model
comparison is built by `make benchmark` (the study is part of the default
matrix).

## Expected result

Closed-book accuracy is low — and on unanswerable questions the baseline tends
to hallucinate plausible limits instead of admitting ignorance. RAG recovers
most answerable questions and makes abstention possible ("the docs don't say").
The takeaway, paired with the medical study:

> **RAG helps when the answer lives outside the model's knowledge — not as a
> blanket "best practice." Measure the baseline, then measure retrieval and
> generation separately.**
