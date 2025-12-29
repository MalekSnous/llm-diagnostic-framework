# 🚀 Quick Start Guide

Get started with LLM Diagnostic Framework in 5 minutes.

---

## 1. Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/MalekSnous/llm-diagnostic-framework.git
cd llm-diagnostic-framework

# Run setup script
bash scripts/setup_environment.sh

# Activate environment
source venv/bin/activate
```

---

## 2. Configure API Keys (1 minute)

Edit `.env` file:

```bash
# OpenAI (for GPT-4)
OPENAI_API_KEY=sk-your-key-here

# Anthropic (for Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Hugging Face (for open-source models)
HUGGINGFACE_TOKEN=hf_your-token-here
```

---

## 3. Run Your First Diagnostic (2 minutes)

### Option A: CLI (Easiest)

```bash
# Run all diagnostic tests
python scripts/run_diagnostics.py \
  --model gpt-4o-mini \
  --test all \
  --num-cases 10

# Results saved to: results/diagnostics/
```

### Option B: Python Script

```python
from llm_diagnostic import ContextLimitsTest, get_llm_client

# Initialize
test = ContextLimitsTest()
test.generate_test_cases(num_cases=10)

client = get_llm_client("gpt-4-turbo-preview")

# Run test
results = test.run_test(client)

# Get insights
insights = test.get_diagnostic_insights()
print(insights["recommendations"])

# Save results
test.save_results("context_test_results.json")
```

### Option C: Makefile (Shortest)

```bash
make diagnose model="gpt-4-turbo-preview"
```

---

## 4. View Results

### Generate HTML Report

```bash
python scripts/generate_report.py \
  --input results/diagnostics/ \
  --output report.html \
  --model gpt-4-turbo-preview

# Open report.html in browser
open report.html
```

### Or Read JSON

```bash
cat results/diagnostics/context_limits_*.json | jq '.diagnostic_insights.recommendations'
```

---

## 5. Apply Improvements

Once you've identified issues, apply improvements:

```bash
# Prompt engineering
python scripts/run_improvements.py \
  --strategy prompt \
  --baseline results/diagnostics/context_limits_*.json \
  --technique few-shot

# RAG system
python scripts/run_improvements.py \
  --strategy rag \
  --baseline results/diagnostics/knowledge_boundaries_*.json \
  --kb data/knowledge_base.txt
```

---

## Common Use Cases

### Use Case 1: Model Comparison

```bash
# Test GPT-4
python scripts/run_diagnostics.py --model gpt-4-turbo-preview --test all

# Test Claude
python scripts/run_diagnostics.py --model claude-3-sonnet-20240229 --test all

# Compare results
python scripts/generate_report.py --input results/ --output comparison.html
```

### Use Case 2: Task-Specific Diagnosis

```python
from llm_diagnostic import (
    StructureValidationTest,
    HallucinationPatternsTest,
    get_llm_client
)

# Your task: Generate valid JSON
client = get_llm_client("gpt-4-turbo-preview")

# Test structure generation
structure_test = StructureValidationTest()
structure_test.generate_test_cases(num_cases=20)
structure_results = structure_test.run_test(client)

# Test hallucination risk
hallucination_test = HallucinationPatternsTest()
hallucination_test.generate_test_cases(num_cases=15)
hallucination_results = hallucination_test.run_test(client)

# Get combined insights
structure_insights = structure_test.get_diagnostic_insights()
hallucination_insights = hallucination_test.get_diagnostic_insights()

print("Structure Issues:", structure_insights["recommendations"])
print("Hallucination Risk:", hallucination_insights["overall_hallucination_risk"])
```

### Use Case 3: Production Pipeline

```python
from llm_diagnostic import (
    get_llm_client,
    ContextLimitsTest,
    RAGSystem
)

# 1. Diagnose
test = ContextLimitsTest()
test.generate_test_cases()
client = get_llm_client("gpt-4-turbo-preview")
baseline_results = test.run_test(client)

insights = test.get_diagnostic_insights()

if insights["bottleneck_identified"]:
    print(f"Issue detected: {insights['recommendations'][0]}")
    
    # 2. Apply fix
    strategy = RAGSystem()
    config = strategy.configure(top_k=3)
    
    improved_results = strategy.apply(
        test_cases=test.test_cases,
        baseline_results=baseline_results,
        config=config,
        llm_client=client,
        knowledge_base=your_documents
    )
    
    # 3. Measure improvement
    improvement = strategy.results[0]
    print(f"Accuracy improved: {improvement.improvement_delta['accuracy'] * 100:.1f}%")
    print(f"Cost: ${improvement.total_cost:.4f}")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall
pip install -e .
```

### Issue: "API key not found"

```bash
# Check .env file
cat .env | grep API_KEY

# Make sure keys are set
export OPENAI_API_KEY=sk-your-key
```

### Issue: "Out of memory" (for local models)

```python
# Use smaller models or quantization
from llm_diagnostic import get_llm_client

# This will use 4-bit quantization automatically
client = get_llm_client("mistralai/Mistral-7B-Instruct-v0.2")
```

---

## Next Steps

1. **Read the full docs**: [README.md](README.md)
2. **Explore case studies**: [case_studies/](case_studies/)
3. **Run all tests**: `make test`
4. **Deploy**: See [deployment/](deployment/) for Docker/K8s configs

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/MalekSnous/llm-diagnostic-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MalekSnous/llm-diagnostic-framework/discussions)
- **Email**: malek.senoussi@gmail.com

---

**Happy diagnosing! 🔍**