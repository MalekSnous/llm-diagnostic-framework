# 🏠 Running Locally - Complete Guide

This guide walks you through running the LLM Diagnostic Framework on your local machine.

---

## Prerequisites

### System Requirements

- **Python**: 3.9+ (3.10 recommended)
- **RAM**: 8GB minimum (16GB+ recommended for local models)
- **Disk**: 5GB free space (more if using local models)
- **GPU**: Optional but recommended for fine-tuning
- **OS**: Linux, macOS, or Windows (WSL recommended)

### Accounts Needed

At least one of:
- OpenAI API key (for GPT-4, GPT-3.5)
- Anthropic API key (for Claude)
- Hugging Face token (for open-source models)

---

## Step 1: Clone and Setup

### Option A: Automatic Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/MalekSnous/llm-diagnostic-framework.git
cd llm-diagnostic-framework

# Run setup script (creates venv, installs deps, etc.)
bash scripts/setup_environment.sh

# Activate environment
source venv/bin/activate  # Linux/macOS
# Or: .\venv\Scripts\activate  # Windows
```

### Option B: Manual Setup

```bash
# Clone
git clone https://github.com/MalekSnous/llm-diagnostic-framework.git
cd llm-diagnostic-framework

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install package
pip install -e .

# Create directories
mkdir -p results/diagnostics results/improvements data/.cache
```

---

## Step 2: Configure API Keys

### Edit .env file

```bash
# Copy template
cp .env.example .env

# Edit with your keys
nano .env  # or vim, code, etc.
```

### Add your API keys:

```bash
# OpenAI (required for GPT-4 tests)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic (optional, for Claude tests)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Hugging Face (optional, for local models)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Default settings
DEFAULT_LLM_MODEL=gpt-4o-mini
MAX_TOKENS=500
TEMPERATURE=0.0
LOG_LEVEL=INFO
```

### Verify setup:

```bash
# Test that environment variables are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY')[:10] + '...')"
```

---

## Step 3: Run Your First Test

### Test 1: Quick Diagnostic

```bash
# Simple context test (5 test cases, fast)
python scripts/run_diagnostics.py \
  --model gpt-4o-mini \
  --test context \
  --num-cases 5

# Check results
ls results/diagnostics/
cat results/diagnostics/context_window_limits_*.json | jq '.aggregated_results.success_rate'
```

**Expected output:**
```
Running Context Window Limits...
Generating 5 test cases...
Running Context Window Limits: 100%|████████| 5/5 [00:15<00:00, 3.02s/it]

Context Window Limits Results
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric         ┃ Value   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Success Rate   │ 80.0%   │
│ Total Cost     │ $0.0234 │
└────────────────┴─────────┘

Results saved to results/diagnostics/context_window_limits_20241215_143022.json
```

### Test 2: Run All Diagnostics

```bash
# Full diagnostic suite (takes ~5-10 minutes)
python scripts/run_diagnostics.py \
  --model gpt-4o-mini \
  --test all \
  --num-cases 10
```

### Test 3: Using Makefile

```bash
# View available commands
make help

# Run diagnosis
make diagnose model="gpt-4o-mini"

# Generate report
make report case="context"
```

---

## Step 4: Interpret Results

### View JSON Results

```bash
# Pretty print results
cat results/diagnostics/context_window_limits_*.json | jq '.'

# Extract recommendations
cat results/diagnostics/*.json | jq '.diagnostic_insights.recommendations[]'

# Compare success rates
cat results/diagnostics/*.json | jq '{test: .test_name, success_rate: .aggregated_results.success_rate}'
```

### Generate HTML Report

```bash
# Create visual report


python3 scripts/generate_report.py  \
  --input results/gpt4o-mini/diagnostics/   \
  --improvements results/gpt4o-mini/improvements/    \
  --output results/gpt4o-test_report.html    \
  --model gpt-4o-mini


# Open in browser
open results/gpt4o-test_report.html  # macOS
# or: xdg-open diagnostic_report.html  # Linux
# or: start diagnostic_report.html  # Windows
```

---

## Step 5: Apply Improvements

### Prompt Engineering

```bash
# Apply few-shot prompting


python3 scripts/run_improvements.py \
  --strategy prompt \
  --baseline results/gpt4o-mini/diagnostics/reasoning_depth_*.json \
  --model gpt-4o-mini \
  --output-dir results/gpt4o-mini/improvements
```

### RAG System

```bash
# First, create a knowledge base
echo "Your knowledge base content here" > data/knowledge_base.txt
echo "Add documents, one per line or paragraph" >> data/knowledge_base.txt

# Apply RAG

python3 scripts/run_improvements.py \
  --strategy rag \
  --baseline results/gpt4o-mini/diagnostics/context_window_limits_*.json \
  --kb data/knowledge_base.txt \
  --model gpt-4o-mini \
  --output-dir results/gpt4o-mini/improvements

```

### Compare Strategies

```bash
# Run multiple strategies
python3 scripts/run_improvements.py \
  --strategy all \
  --baseline results/gpt4o-mini/diagnostics/context_window_limits_*.json \
  --kb data/knowledge_base.txt \
  --model gpt-4o-mini \
  --output-dir results/gpt4o-mini/improvements

# View comparison
cat results/improvements/*.json | jq '{strategy: .strategy_name, improvement: .results[0].improvement_delta}'
```

---

## Step 6: Run Case Studies

### Medical Entity Extraction Example

```bash
# Run complete case study
python case_studies/medical_entity_extraction/run_study.py

# Expected output: baseline → improvements → comparison
```

### Custom Case Study

```python
# Create your own: custom_case_study.py
from llm_diagnostic import get_llm_client, ContextLimitsTest, RAGSystem

# Your test cases
test_cases = [...]  # Your domain-specific tests

# Run baseline
client = get_llm_client("gpt-4o-mini")
test = ContextLimitsTest()
# ... (follow pattern from medical example)
```

---

## Step 7: Using Local Models (Optional)

### Download and Run Mistral-7B

```bash
# Hugging Face token required
export HUGGINGFACE_TOKEN=hf_your_token_here

# Run with local model
python scripts/run_diagnostics.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --test context \
  --num-cases 5

# Note: First run downloads model (~14GB)
# Requires GPU for reasonable speed (or patience on CPU)
```

---

## Troubleshooting

### Issue: "ImportError: No module named llm_diagnostic"

```bash
# Solution 1: Reinstall package
pip install -e .

# Solution 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "OpenAI API Error: Invalid API key"

```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Verify key format (should start with 'sk-proj-' or 'sk-')
echo $OPENAI_API_KEY

# Test manually
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list().data[0].id)"
```

### Issue: "Out of Memory" (for local models)

```bash
# Solution 1: Use CPU (slower but works)
export CUDA_VISIBLE_DEVICES=""

# Solution 2: Use smaller model
python scripts/run_diagnostics.py --model gpt-3.5-turbo

# Solution 3: Reduce batch size
# Edit fine_tuning config to use smaller batch_size=1
```

### Issue: "Permission denied" on scripts

```bash
# Make scripts executable
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

### Issue: Docker errors

```bash
# Build Docker image
make docker-build

# Run container
make docker-run

# Check logs
docker logs llm-diagnostic

# Enter container for debugging
docker exec -it llm-diagnostic bash
```

---

## Performance Tips

### Speed Up Testing

```bash
# Use fewer test cases during development
python scripts/run_diagnostics.py --test context --num-cases 3

# Use faster model for iteration
python scripts/run_diagnostics.py --model gpt-3.5-turbo --test all

# Run only specific tests
python scripts/run_diagnostics.py --test reasoning --num-cases 5
```

### Reduce Costs

```bash
# Use gpt-3.5-turbo ($0.0005/1K vs $0.01/1K for GPT-4)
export DEFAULT_LLM_MODEL=gpt-3.5-turbo

# Cache results
# Results are saved automatically, reuse them:
python scripts/run_improvements.py --baseline results/diagnostics/cached_result.json

# Use local models (free after download)
python scripts/run_diagnostics.py --model mistralai/Mistral-7B-Instruct-v0.2
```

---

## Development Workflow

### Typical iteration:

```bash
# 1. Make changes to code
vim llm_diagnostic/core/llm_client.py

# 2. Run tests
make test

# 3. Try it out
python scripts/run_diagnostics.py --test context --num-cases 3

# 4. Format code
make format

# 5. Commit
git add .
git commit -m "Add feature X"
```

### Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use IPython
from IPython import embed; embed()

# Run with verbose logging
export LOG_LEVEL=DEBUG
python scripts/run_diagnostics.py --test context
```

---

## Next Steps

1. **Customize for your domain**: Add your own test cases
2. **Extend the framework**: Create new failure tests or improvement strategies
3. **Deploy**: See `deployment/` for production deployment guides
4. **Contribute**: Submit PRs for new features!

---

## Getting Help

- **Documentation**: See [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
- **Issues**: [GitHub Issues](https://github.com/MalekSnous/llm-diagnostic-framework/issues)
- **Email**: malek.senoussi@gmail.com

---

**Happy local testing! 🏠**