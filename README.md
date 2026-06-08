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
- **Cost** — real token cost per provider (OpenAI per-1K, Anthropic per-1M),
  local models at $0.

> **Reproducibility note.** Earlier headline numbers in this README came from a
> draft scoring method (recall-only substring match) and a token-pricing bug
> (per-1M vs per-1K, ~1000× cost inflation). Both are now fixed and covered by
> tests, so old figures were removed rather than shown misleadingly. Re-run the
> case studies (below) to generate trustworthy numbers for your own setup.

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

Two studies designed to tell **opposite** stories — which is the whole point:
optimization is task-dependent, so you must measure.

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

Questions about a private document (`data/product_docs.txt`) the model was never
trained on. Baseline (closed-book) must guess; RAG retrieves the answer.

```bash
pip install -e ".[rag]"
python case_studies/rag_document_qa/run_study.py --model gpt-4o-mini
```

Each study writes a JSON + interactive HTML report to `results/case_studies/`.

> Concrete numbers are intentionally **not** hard-coded here — run the studies with
> your own models/keys to get figures you can trust. See the reproducibility note above.

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
│   ├── utils/                   # Helpers
│   │   └── case_study_reporter.py  # HTML report generation
│   └── cli.py                   # Console entry points (llm-diagnose/-improve/-report)
├── scripts/                     # CLI tools
│   ├── run_diagnostics.py
│   ├── run_improvements.py
│   └── generate_report.py
├── case_studies/                # Complete examples
│   ├── medical_entity_extraction/   # RAG/prompting can *hurt* a strong model
│   └── rag_document_qa/             # RAG *wins* on private-document QA
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
