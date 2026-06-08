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

### Key Insight from Real Testing

Our medical entity extraction case study revealed **counterintuitive results**:

| Model | Baseline | +Prompt Eng | +RAG | Cost |
|-------|----------|-------------|------|------|
| **GPT-4o-mini** | 75.6% | **71.0% (-4.6%)** | **66.9% (-8.8%)** | $0.41 |
| **GPT-4o** | 71.5% | **68.5% (-2.9%)** | **64.8% (-6.7%)** | $4.06 |
| **Phi-2 (local)** | 71.7% | **100% (+28.3%)** | **90% (+18.3%)** | $0.00 |

**Findings:**
1. ✅ **Smaller models benefit more** from optimization (Phi-2: +28%)
2. ❌ **Advanced models can degrade** with generic strategies (GPT-4o-mini: -8.8%)
3. 💰 **GPT-4o costs 10x more** but performs worse than GPT-4o-mini
4. 📊 **Empirical testing is essential** - theory doesn't predict real performance

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

## 📊 Real Results: Medical Entity Extraction Case Study

### The Challenge

Extract medical entities (conditions, medications, procedures) from clinical text:
- **Input**: "Patient with hypertension and diabetes. Prescribed metformin 500mg."
- **Expected**: `["hypertension", "diabetes", "metformin"]`

### Models Tested

We compared **4 models** across **3 optimization strategies**:

#### 1. **GPT-4o-mini** (Recommended for Production)

| Strategy | Accuracy | Change | Cost (8 cases) | Cost/Case |
|----------|----------|--------|----------------|-----------|
| **Baseline (Zero-shot)** | **75.6%** | - | **$0.41** | **$0.05** |
| Prompt Engineering | 71.0% | **-4.6%** ❌ | $0.24 | $0.03 |
| RAG System | 66.9% | **-8.8%** ❌ | $0.34 | $0.04 |

**Key Finding**: GPT-4o-mini's zero-shot performance was optimal. Generic optimization strategies **degraded** accuracy by confusing the model with unnecessary context.

#### 2. **GPT-4o** (Premium, Not Worth It)

| Strategy | Accuracy | Change | Cost (8 cases) | Cost/Case |
|----------|----------|--------|----------------|-----------|
| Baseline | 71.5% | - | **$4.06** | **$0.51** |
| Prompt Engineering | 68.5% | -2.9% ❌ | $3.79 | $0.47 |
| RAG System | 64.8% | -6.7% ❌ | $4.14 | $0.52 |

**Key Finding**: GPT-4o cost **10x more** than GPT-4o-mini while performing **worse** (71.5% vs 75.6%). For simple entity extraction, bigger ≠ better.

#### 3. **Phi-2** (Local, Best for Optimization)

| Strategy | Accuracy | Change | Cost |
|----------|----------|--------|------|
| Baseline | 71.7% | - | $0.00 |
| Prompt Engineering | **100%** | **+28.3%** ✅ | $0.00 |
| RAG System | **90%** | **+18.3%** ✅ | $0.00 |

**Key Finding**: Smaller models (2.7B params) benefit **dramatically** from optimization strategies (+28%). Perfect for local deployment with fine-tuning budget.

### Cost-Benefit Analysis Summary

| Model | Best Strategy | Final Accuracy | Total Cost | ROI |
|-------|---------------|----------------|------------|-----|
| **GPT-4o-mini** | Zero-shot | 75.6% | $0.41 | ✅ Best value |
| GPT-4o | Zero-shot | 71.5% | $4.06 | ❌ Worst value |
| **Phi-2** | Prompt Eng | 100% | $0.00 | ✅ Best accuracy |

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

## 🎓 Key Learnings

### 1. **Model Size ≠ Performance**

GPT-4o-mini (8B params) **outperformed** GPT-4o (1.7T params) on entity extraction:
- GPT-4o-mini: 75.6% @ $0.05/case
- GPT-4o: 71.5% @ $0.51/case

**Lesson**: Match model size to task complexity. Overkill wastes money.

### 2. **Optimization Can Hurt**

Both GPT-4o-mini and GPT-4o **degraded** with generic few-shot examples:
- Generic examples confused models that already had strong medical knowledge
- RAG context diluted focus on entity extraction

**Lesson**: Baseline testing is mandatory. Not all strategies help all tasks.

### 3. **Smaller Models Need More Help**

Phi-2 (2.7B params) gained **+28%** from prompt engineering:
- Baseline: 71.7%
- With optimization: 100%

**Lesson**: Budget-constrained projects should use smaller models + optimization.

### 4. **Cost-Effectiveness Varies Wildly**

| Model | Accuracy | Cost/1K cases | Cost per Point |
|-------|----------|---------------|----------------|
| GPT-4o | 71.5% | $510 | $7.13 |
| GPT-4o-mini | 75.6% | $50 | $0.66 |
| Phi-2 (optimized) | 100% | $0 | $0.00 |

**Lesson**: Always calculate cost per accuracy point, not just total cost.

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
