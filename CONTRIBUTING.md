# Contributing

Thanks for your interest in improving the LLM Diagnostic Framework!

## Development setup

```bash
git clone https://github.com/MalekSnous/llm-diagnostic-framework.git
cd llm-diagnostic-framework

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + dev tools (no heavy ML stack)
pre-commit install             # run linters/formatters on each commit
```

The core install is intentionally lightweight. Heavy backends are opt-in extras,
imported lazily so the framework imports without them:

| Extra | Enables |
|-------|---------|
| `.[local]` | Local Hugging Face models (`HuggingFaceClient`) |
| `.[rag]` | RAG strategy (`RAGSystem`: sentence-transformers + chromadb) |
| `.[finetune]` | LoRA fine-tuning (`FineTuningStrategy`) |
| `.[dev]` | Tests + linters (ruff, black, isort, mypy, pytest) |
| `.[all]` | Everything |

## Checks (must pass in CI)

```bash
make lint     # ruff + black --check + isort --check + mypy
make test     # pytest

# auto-fix formatting:
make format
```

The test suite is **fully offline**: it uses a deterministic `MockLLMClient`
(see `tests/conftest.py`) and never calls a real LLM API or downloads a model.
No API keys are required to run or contribute. Please keep it that way — mock or
`pytest.importorskip` any heavy/networked dependency.

## Adding a new case study

1. Copy the structure of `case_studies/rag_document_qa/`:
   `create_test_cases()` → `run_baseline()` → one or more strategies →
   `generate_summary()` → save via `llm_diagnostic.utils.case_study_reporter`.
2. Put domain data under `data/`.
3. Add a `README.md` explaining the question, data, how to run, and expected result.
4. Add an offline test for the scoring/data logic (see `tests/test_case_study_rag.py`).
5. Add a `make run-<name>-study` target.

## Adding a new diagnostic test or strategy

Subclass `BaseFailureTest` (in `llm_diagnostic/failure_tests/base_test.py`) or
`BaseImprovementStrategy` (`llm_diagnostic/improvements/base_strategy.py`),
implement the abstract methods, export it from the relevant `__init__.py`, and
add a parametrized test.

## Pull requests

- Keep PRs focused; describe the motivation.
- Ensure `make lint` and `make test` pass.
- Update docs/README when behaviour or commands change.
