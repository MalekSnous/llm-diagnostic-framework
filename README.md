# LLM Diagnostic Framework

> **A Systematic Approach to LLM Performance Analysis**: Diagnose failures, test improvements, and make data-driven decisions about LLM optimization strategies.

[![CI](https://github.com/MalekSnous/llm-diagnostic-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/MalekSnous/llm-diagnostic-framework/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🔗 Live demo & interactive reports → [maleksnous.github.io/llm-diagnostic-framework](https://maleksnous.github.io/llm-diagnostic-framework/)**

---

## 🎯 Why This Framework?

**The Problem**: Everyone tries prompt engineering, RAG, and fine-tuning on their LLM applications—but which actually helps? When does prompt engineering *hurt* performance? Is RAG worth the cost?

**The Solution**: This framework provides **empirical testing and cost-benefit analysis** to answer these questions systematically, not by guessing.

### What it measures

For a given task, the framework runs a **baseline** and each optimization
strategy (prompt engineering, RAG, fine-tuning) under one harness, and reports
**quality vs cost** so the trade-off is explicit:

- **Quality** — task-appropriate metrics. For entity extraction it uses
  **precision / recall / F1** (via `Evaluator.fuzzy_entity_metrics`), not a
  recall-only substring scan — so a verbose model can't win just by dumping text.
  For text-to-SQL it goes further: the generated query is **executed** and its
  result set compared to the gold query's (**execution accuracy**) — a metric
  that can't be gamed by formatting at all.
- **Cost** — real token cost per provider (OpenAI per-1K, Anthropic per-1M),
  local models at $0.

> **Reproducibility note.** Earlier headline numbers in this README came from a
> draft scoring method (recall-only substring match) and a token-pricing bug
> (per-1M vs per-1K, ~1000× cost inflation). Both are now fixed and covered by
> tests. The figures in [Latest Results](#-latest-results-medical-entity-extraction)
> come from a real run with that corrected scoring and pricing — re-run the case
> studies to reproduce them for your own setup.

**Qualitative lessons that hold up:**
1. 📊 **Measure the baseline first** — optimization is not free lunch.
2. ❌ **Generic strategies can *hurt*** a model that's already strong at the task.
3. ✅ **RAG wins when the answer is outside the model's knowledge** (see the
   document-QA study) and can hurt when it just adds noise (medical study).
4. 💰 **Match model size to task** — and judge on cost *per quality point*, not raw accuracy.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/MalekSnous/llm-diagnostic-framework.git
cd llm-diagnostic-framework
pip install -e .            # core + hosted API clients (OpenAI/Anthropic)
```

The core install is lightweight. Heavy ML backends are **opt-in extras** (imported
lazily, so the framework works without them):

```bash
pip install -e ".[rag]"        # RAG strategy (sentence-transformers + chromadb)
pip install -e ".[local]"      # local Hugging Face models (torch + transformers)
pip install -e ".[finetune]"   # LoRA fine-tuning
pip install -e ".[dev]"        # tests + linters
pip install -e ".[all]"        # everything
```

### Run Your First Diagnostic

```python
from llm_diagnostic import ContextLimitsTest, get_llm_client

# 1. Test a model
client = get_llm_client("gpt-4o-mini")
test = ContextLimitsTest()
test.generate_test_cases(num_cases=15)
results = test.run_test(client)

# 2. Get actionable insights
insights = test.get_diagnostic_insights()
print(insights["recommendations"])
# → ["Context window becomes unreliable beyond 4000 tokens",
#    "Consider RAG with chunking for longer documents"]

# 3. Measure improvement impact
from llm_diagnostic import RAGSystem
strategy = RAGSystem()
improved = strategy.apply(test.test_cases, results, client, knowledge_base=docs)

# Compare: Accuracy +20%, Cost +15% → Worth it!
```

### Generate Professional Reports

```bash
# Run diagnostics
python scripts/run_diagnostics.py \
  --model gpt-4o-mini \
  --test all \
  --num-cases 50

# Generate HTML report
python scripts/generate_report.py \
  --input results/gpt4o-mini/diagnostics/ \
  --output report.html \
  --model gpt-4o-mini
```

---

## 📊 Case Studies

Three studies, three **metric families** — because the right metric is
task-dependent, and so is the right optimization:

### 1. Medical Entity Extraction — when optimization can *hurt*

Extract medical entities (conditions, medications, procedures) from clinical text:
- **Input**: "Patient with hypertension and diabetes. Prescribed metformin 500mg."
- **Expected**: `["hypertension", "diabetes", "metformin"]`
- **Metric**: precision / recall / **F1** with containment matching.

Compares **baseline vs prompt engineering vs RAG** across hosted and local models.
The expected pattern: on a model already strong at the task, generic strategies
can add noise and *reduce* F1 — so the baseline must be beaten, not assumed.

```bash
python case_studies/medical_entity_extraction/run_study.py --model gpt-4o-mini
python case_studies/medical_entity_extraction/run_study.py --model microsoft/phi-2   # local, [local] extra
```

### 2. Document QA with RAG — when retrieval *wins*

100 questions about a fictional internal platform whose facts exist only in a
private multi-document knowledge base (`data/acmecloud_kb/`) — no model can know
them from training. Three arms through a **full modular RAG pipeline**
(`llm_diagnostic/rag/`: ingestion → chunking → embedding → vector search →
reranking → answer): closed-book baseline, RAG, and RAG + cross-encoder
reranking. Four difficulty tiers (verbatim lookups → cross-document reasoning),
including **unanswerable questions** where the right behaviour is to abstain.
Reports answer accuracy (digit-boundary matching, so "50" can't be satisfied by
"500") *and* retrieval recall@k against gold source documents.

```bash
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini            # TF-IDF fallback works offline
pip install -e ".[rag]"                                                         # enables dense embeddings + reranking
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini            # dense + rerank arms
```

### 3. Text-to-SQL — a metric you can't game

Natural-language questions over an e-commerce SQLite schema. The model's SQL is
**executed** against the seeded database and its result set compared to the gold
query's — **execution accuracy**. Verbosity, fences, aliases, row order: none of
it can inflate the score. ~100 questions in four difficulty tiers (single-table
filters → anti-joins and nested aggregates).

```bash
python case_studies/text_to_sql/run_study.py --model gpt-4o-mini
```

RAG is deliberately skipped here (the schema fits in the prompt); the comparison
is zero-shot vs few-shot SQL examples.

Each study writes a JSON + interactive HTML report to `results/case_studies/`.

### Run the whole benchmark in one command

```bash
make benchmark models="gpt-4o-mini groq/llama-3.1-8b-instant"
# → runs every study for every model, rebuilds the cross-model comparison
#   pages, and publishes all reports to docs/ (GitHub Pages deploys on push)
```

Omit `models=...` to use the default 5-model list (`BENCH_MODELS` in the
Makefile). Failures for individual models (missing key, etc.) are non-fatal —
whatever ran still gets compared and published.

---

## 📊 Latest Results (Medical Entity Extraction)

Five models on the **same 97-case clinical dataset** (easy → expert difficulty),
scored by entity-extraction **F1** (precision/recall, verbosity-robust) with
**real token cost**. Baseline = zero-shot, +Prompt Eng = few-shot prompting.

🔗 **Interactive version (with per-difficulty breakdown):** [maleksnous.github.io/llm-diagnostic-framework → Model comparison](https://maleksnous.github.io/llm-diagnostic-framework/comparison_medical_entity_extraction.html)

| Model | Baseline F1 | +Prompt Eng | Δ (pts) | Cost (97 cases) |
|-------|------------:|------------:|--------:|----------------:|
| **gpt-4o** | 68.6% | **71.3%** | +2.8 | $0.043 |
| **gpt-4o-mini** | 57.3% | **70.4%** | +13.1 | **$0.0032** |
| claude-sonnet-4-6 | 11.4%* | 64.3% | +52.9 | $0.158 |
| groq/llama-3.1-8b-instant | 58.2% | 61.7% | +3.5 | **$0.0012** |
| groq/llama-3.3-70b-versatile | 57.8% | 59.2% | +1.4 | $0.013 |

### Interpretation

1. **gpt-4o-mini is the value winner.** With prompt engineering it reaches **70.4% F1**
   — within ~1 point of gpt-4o (71.3%) — at roughly **13× lower cost**. For this task,
   the premium model isn't worth it.
2. **A low baseline ≠ a weak model.** Sonnet's 11.4% baseline (*) is largely a
   **formatting artifact**: it answered in prose, which the entity parser penalizes.
   Few-shot prompting standardized the output format and F1 jumped to 64.3%. Read the
   baseline→prompt delta as much about **output format** as raw capability.
3. **Bigger ≠ better.** Llama-3.3-70B (59.2%) does **not** beat Llama-3.1-8B (61.7%) here,
   despite ~9× the parameters and ~11× the cost — match model size to the task.
4. **Prompt engineering's main job here is standardizing output**, so its biggest gains
   land on the models whose baseline format was poorest (Sonnet, gpt-4o-mini).
5. **Cost spans ~135×** across models (Sonnet vs Llama-3.1-8B) — always judge
   **cost per quality point**, not raw F1.

> Snapshot of one run (June 2026, RAG excluded). Small sample (97 cases) and a
> containment-based metric sensitive to output format — treat as a methodology
> demonstration, not a leaderboard. Re-run to reproduce.

## 📊 Latest Results (RAG Document QA)

Same 100-question private-knowledge dataset, same harness: closed-book baseline
vs the full RAG pipeline (dense retrieval + cross-encoder reranking).

🔗 **Interactive version:** [maleksnous.github.io/llm-diagnostic-framework → RAG comparison](https://maleksnous.github.io/llm-diagnostic-framework/comparison_rag_document_qa.html)

| Model | Baseline (closed-book) | +RAG (best arm) | Δ (pts) | Recall@4 | Cost (100 q) |
|-------|-----------------------:|----------------:|--------:|---------:|-------------:|
| claude-sonnet-4-6 | 8.0% | **100.0%** | +92.0 | 100% | $0.284 |
| **gpt-4o-mini** | 9.0% | **94.0%** | +85.0 | 100% | **$0.012** |
| groq/llama-3.1-8b-instant | 8.0% | 91.0% | +83.0 | 100% | **$0.0038** |
| gpt-4o | 8.0% | 91.0% | +83.0 | 100% | $0.192 |

1. **The mirror image of the medical study**: here every model's closed-book
   baseline collapses to ~8-10% (the facts are private; the remaining points come
   from correctly abstaining on unanswerable questions), and retrieval recovers
   87-100%.
2. **Retrieval was never the bottleneck** (recall@4 = 100% on this corpus): the
   87→100% spread between models is entirely about *reading* the retrieved
   context — and reranking still adds +2 to +4 points on three of the four
   models by putting the right chunk first.
3. **An 8B model with RAG (91%) beats every model without it** — retrieval, not
   parameter count, is what buys accuracy when knowledge is the constraint.

> Snapshot of one run (July 2026). llama-3.3-70b hit a daily API quota mid-run
> and is absent. Re-run to reproduce: `make benchmark studies="rag_document_qa"`.

---

## 🛠️ Framework Components

### 1. Diagnostic Tests

Systematic tests for LLM failure modes:

| Test | What It Measures | When to Use |
|------|------------------|-------------|
| **Context Window Limits** | Information retention in long documents | Document Q&A, summarization |
| **Reasoning Depth** | Multi-hop logical inference | Complex reasoning tasks |
| **Knowledge Boundaries** | Training cutoff & rare facts | Recent events, specialized domains |
| **Structure Validation** | JSON/code generation accuracy | API responses, data extraction |
| **Hallucination Patterns** | Factuality vs fluency tradeoff | High-stakes decisions |

### 2. Improvement Strategies

Tested and benchmarked:

| Strategy | Implementation | Expected Gain | Cost Impact | Best For |
|----------|----------------|---------------|-------------|----------|
| **Prompt Engineering** | Few-shot, Chain-of-Thought | 0-30% | None | Small models, ambiguous tasks |
| **RAG System** | Vector search + context | 10-60% | +10-30% | Knowledge gaps, recent info |
| **Fine-tuning** | LoRA on domain data | 15-40% | $0.50-5 | Domain-specific, consistent style |

### 3. Evaluation & Reporting

- **Automated HTML reports** with interactive visualizations
- **Cost tracking** for API calls (OpenAI, Anthropic)
- **Comparative analysis** across models and strategies
- **Professional output** ready for portfolio/stakeholders

---

## 📖 Usage Examples

### CLI Interface

```bash
# Run full diagnostic suite
python scripts/run_diagnostics.py \
  --model gpt-4o-mini \
  --test all \
  --num-cases 50 \
  --output-dir results/diagnostics

# Test improvement strategies
python scripts/run_improvements.py \
  --strategy all \
  --baseline results/diagnostics/context_window_limits_*.json \
  --kb data/knowledge_base.txt \
  --model gpt-4o-mini

# Generate professional report
python scripts/generate_report.py \
  --input results/diagnostics/ \
  --improvements results/improvements/ \
  --output full_report.html \
  --model gpt-4o-mini
```

### Python API

```python
from llm_diagnostic import (
    get_llm_client,
    ReasoningDepthTest,
    PromptEngineeringStrategy,
    RAGSystem
)

# Initialize
client = get_llm_client("gpt-4o-mini")
test = ReasoningDepthTest()

# Run diagnostic
test.generate_test_cases(num_cases=20)
baseline = test.run_test(client)

# Test improvements
strategy = PromptEngineeringStrategy()
config = strategy.configure(technique="chain-of-thought")
improved = strategy.apply(test.test_cases, baseline, config, client)

# Compare performance
print(f"Baseline: {baseline.accuracy:.1%}")
print(f"Improved: {improved.accuracy:.1%}")
print(f"Cost: ${improved.total_cost:.4f}")
print(f"ROI: {improved.roi:.2f}x")
```

---

## 📂 Project Structure

```
llm-diagnostic-framework/
├── llm_diagnostic/              # Core package
│   ├── core/                    # LLM clients, evaluators
│   │   ├── llm_client.py       # OpenAI, HuggingFace clients
│   │   └── evaluator.py        # Metrics calculation
│   ├── failure_tests/           # Diagnostic tests
│   │   ├── context_limits.py
│   │   ├── reasoning_depth.py
│   │   ├── knowledge_boundaries.py
│   │   ├── structure_validation.py
│   │   └── hallucination_patterns.py
│   ├── improvements/            # Optimization strategies
│   │   ├── prompt_engineering.py
│   │   ├── rag_system.py
│   │   └── base_strategy.py
│   ├── rag/                     # Modular RAG pipeline
│   │   ├── ingestion.py        # corpus loading
│   │   ├── chunking.py         # paragraph-aware chunks + overlap
│   │   ├── embeddings.py       # dense (MiniLM) or TF-IDF backends
│   │   ├── vector_store.py     # in-memory cosine store
│   │   ├── reranker.py         # cross-encoder second stage
│   │   └── pipeline.py         # retrieve → rerank → grounded answer
│   ├── utils/                   # Helpers
│   │   └── case_study_reporter.py  # HTML report generation
│   └── cli.py                   # Console entry points (llm-diagnose/-improve/-report)
├── scripts/                     # CLI tools
│   ├── run_diagnostics.py
│   ├── run_improvements.py
│   └── generate_report.py
├── case_studies/                # Complete examples
│   ├── medical_entity_extraction/   # RAG/prompting can *hurt* a strong model
│   ├── rag_document_qa/             # RAG *wins* on private-document QA
│   └── text_to_sql/                 # execution accuracy — a metric you can't game
├── data/                        # Domain data (knowledge bases, doc corpora)
├── results/                     # Output reports (JSON + HTML)
├── tests/                       # Offline pytest suite (mocked LLM client)
└── .github/workflows/ci.yml     # Lint + type-check + tests on 3.9–3.11
```

---

## 🎓 Key Learnings (methodology, not marketing)

### 1. **Match model size to task**
A bigger model isn't automatically better for a narrow task; judge on **cost per
quality point**, not raw accuracy or raw price.

### 2. **Optimization can hurt**
Generic few-shot examples or RAG context can *dilute* a model that's already
strong at the task. **Baseline testing is mandatory** — not all strategies help.

### 3. **RAG is conditional, not a default**
RAG wins when the answer is **outside the model's knowledge** (the document-QA
study) and can hurt when it just adds noise (the medical study).

### 4. **Beware metric artifacts** *(a real bug this project caught)*
A recall-only substring metric made a weak local model look like it scored 100%
— it was simply being verbose, so every expected token appeared somewhere in its
output. Switching to **precision/recall/F1** (`Evaluator.fuzzy_entity_metrics`)
removed the illusion. **Always sanity-check a metric against raw model output.**

### 5. **Get the cost units right**
Token pricing is quoted per-1M by providers but often computed per-1K in code — a
1000× error waiting to happen. The pricing table and `_calculate_cost` are now
consistent and unit-tested.

---

## 🧪 Running Tests

The test suite is **fully offline** — it uses a deterministic mock LLM client
(`tests/conftest.py`), so it needs no API keys, makes no network calls, and
downloads no models.

```bash
# Install dev dependencies + git hooks
pip install -e ".[dev]"
pre-commit install

# Run the suite (with coverage, per pyproject config)
make test            # or: pytest

# Lint + type-check (matches CI)
make lint            # ruff + black --check + isort --check + mypy

# Auto-format
make format          # ruff --fix + black + isort
```

### Continuous Integration

`.github/workflows/ci.yml` runs lint, format checks, mypy, and the test suite on
Python 3.9 / 3.10 / 3.11 for every push and pull request. Because tests are mocked,
**CI requires no API keys and incurs no cost.**

---

## 🤝 Contributing

Contributions welcome! Priority areas:

- **New diagnostic tests** (multimodal, code execution, adversarial)
- **Additional models** (Anthropic Claude, Gemini, local LLMs)
- **Case studies** in different domains (legal, financial, code generation)
- **Visualization improvements** (interactive dashboards)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with **PyTorch**, **Transformers** (Hugging Face), **OpenAI API**
- Embedding models from **Sentence Transformers**
- Vector database by **ChromaDB**
- Inspired by systematic ML research methodologies

---

## 📧 Author

**Malek Senoussi, PhD**  
Machine Learning Engineer | LLM Optimization Specialist

- 🌐 Portfolio: [maleksnous.github.io](https://maleksnous.github.io)
- 💼 LinkedIn: [maleksen](https://linkedin.com/in/maleksen)
- 📧 Email: malek.senoussi@gmail.com
- 🐙 GitHub: [@MalekSnous](https://github.com/MalekSnous)

---

## 🔗 Resources

- 📚 [Full Documentation](docs/)
- 📊 [Case Study Reports](results/case_studies/)
- 🎓 [Tutorial Notebooks](notebooks/)
- 📖 [API Reference](docs/api_reference.md)

---

## 🌟 Star History

If this framework helped you make better LLM decisions, **please star the repository!** ⭐

---

## 💡 Next Steps

1. **Clone the repo** and run the medical case study
2. **Test your own model** with the diagnostic suite
3. **Compare strategies** for your specific use case
4. **Share your results** - open an issue or PR!

**Remember**: Trust data, not hype. Measure everything. 📊
