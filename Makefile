.PHONY: help install install-dev test test-cov lint format clean docker-build docker-run deploy-modal run-medical-study run-rag-study run-sql-study publish publish-only compare-medical benchmark

# Use the project venv's interpreter when it exists, so `make` works without
# activating the venv first. Override explicitly with: make PYTHON=python3.11 ...
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

help:
	@echo "LLM Diagnostic Framework - Available Commands"
	@echo "=============================================="
	@echo "Setup:"
	@echo "  make install         - Install package and dependencies"
	@echo "  make install-dev     - Install with dev dependencies"
	@echo "  make setup-env       - Create .env file from template"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo "  make lint            - Run linters"
	@echo "  make format          - Format code with black and isort"
	@echo "  make clean           - Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make diagnose        - Run diagnostic on a task"
	@echo "  make improve         - Run improvement strategy"
	@echo "  make report          - Generate HTML report"
	@echo ""
	@echo "Case studies & publishing:"
	@echo "  make run-medical-study model=gpt-4o-mini"
	@echo "  make run-rag-study model=gpt-4o-mini"
	@echo "  make run-sql-study model=gpt-4o-mini"
	@echo "  make benchmark models=\"gpt-4o-mini groq/llama-3.1-8b-instant\""
	@echo "                                  - Run ALL studies for each model, build the"
	@echo "                                    cross-model comparisons, publish to docs/"
	@echo "                                    (filter with studies=\"text_to_sql\")"
	@echo "  make publish model=gpt-4o-mini  - Run studies + publish reports to docs/"
	@echo "  make publish-only               - Publish existing reports (no re-run)"
	@echo "  make compare-medical            - Compare gpt-4o/gpt-4o-mini/phi-2/sonnet (medical, no RAG)"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container locally"
	@echo "  make deploy-modal    - Deploy to Modal"
	@echo "  make deploy-web      - Deploy web app"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,web]"

install-all:
	pip install -e ".[all]"

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please add your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=llm_diagnostic --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	ruff check .
	black --check .
	isort --check-only .
	mypy llm_diagnostic/

format:
	ruff check . --fix
	black .
	isort .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Diagnostic commands
diagnose:
	@echo "Running diagnostic..."
	$(PYTHON) scripts/run_diagnostics.py --task "$(task)" --model "$(model)"

improve:
	@echo "Running improvement strategy: $(strategy)"
	$(PYTHON) scripts/run_improvements.py --strategy "$(strategy)" --case-study "$(case)"

report:
	@echo "Generating report..."
	$(PYTHON) scripts/generate_report.py --case-study "$(case)"

# Docker
docker-build:
	docker build -t llm-diagnostic:latest -f deployment/docker/Dockerfile .

docker-run:
	docker-compose -f deployment/docker/docker-compose.yml up

docker-stop:
	docker-compose -f deployment/docker/docker-compose.yml down

# Web app
web-backend:
	cd web/backend && uvicorn main:app --reload --port 8000

web-frontend:
	cd web/frontend && streamlit run app.py

# Deployment
deploy-modal:
	modal deploy deployment/modal/modal_app.py

deploy-web:
	@echo "Building and deploying web app..."
	$(MAKE) docker-build
	@echo "Push to your container registry and deploy"

# Notebooks
notebook:
	jupyter notebook notebooks/

# Download models (for fine-tuning)
download-models:
	bash scripts/download_models.sh

# Run case study (override the model with: make run-medical-study model=gpt-4o)
MODEL ?= gpt-4o-mini

run-medical-study:
	$(PYTHON) case_studies/medical_entity_extraction/run_study.py --model $(or $(model),$(MODEL))

run-rag-study:
	$(PYTHON) case_studies/rag_document_qa/run_study.py --model $(or $(model),$(MODEL))

run-sql-study:
	$(PYTHON) case_studies/text_to_sql/run_study.py --model $(or $(model),$(MODEL))

# One command: run all studies for a model, then copy the freshest HTML reports
# into docs/ and regenerate the landing-page cards. Study failures (e.g. missing
# API key or RAG extra) are non-fatal so whatever ran still gets published.
# Usage:  make publish model=gpt-4o-mini
publish:
	-$(PYTHON) case_studies/medical_entity_extraction/run_study.py --model $(or $(model),$(MODEL))
	-$(PYTHON) case_studies/rag_document_qa/run_study.py --model $(or $(model),$(MODEL))
	-$(PYTHON) case_studies/text_to_sql/run_study.py --model $(or $(model),$(MODEL))
	$(PYTHON) scripts/publish_reports.py
	@echo ""
	@echo "Reports published to docs/. Review, then commit & push to deploy Pages:"
	@echo "    git add docs/ && git commit -m 'Update published reports' && git push"

# Only (re)publish reports that already exist in results/, without re-running.
publish-only:
	$(PYTHON) scripts/publish_reports.py

# Compare several models on the medical study in one command (RAG skipped — compare
# it later). Study failures (missing key, no local model) are non-fatal.
# Override the list:  make compare-medical models="gpt-4o gpt-4o-mini claude-sonnet-4-6"
MEDICAL_MODELS ?= gpt-4o gpt-4o-mini microsoft/phi-2 claude-sonnet-4-6
compare-medical:
	@for m in $(or $(models),$(MEDICAL_MODELS)); do \
	  echo "=== $$m ==="; \
	  $(PYTHON) case_studies/medical_entity_extraction/run_study.py --model $$m --skip-rag || true; \
	done
	$(PYTHON) scripts/compare_models.py --study medical_entity_extraction
	@echo ""
	@echo "Comparison written to docs/comparison_medical_entity_extraction.html"

# Full benchmark: every model × every benchmarkable study (medical without RAG,
# text-to-SQL, RAG document QA), then rebuild the cross-model comparison for each study that ran
# and publish everything to docs/. Study failures (missing key, no local model)
# are non-fatal so the rest of the matrix still runs. Override either axis:
#   make benchmark models="gpt-4o-mini groq/llama-3.1-8b-instant"
#   make benchmark models="gpt-4o claude-sonnet-4-6" studies="text_to_sql"
BENCH_MODELS ?= gpt-4o gpt-4o-mini claude-sonnet-4-6 groq/llama-3.1-8b-instant groq/llama-3.3-70b-versatile
BENCH_STUDIES ?= medical_entity_extraction text_to_sql rag_document_qa
benchmark:
	@for m in $(or $(models),$(BENCH_MODELS)); do \
	  for s in $(or $(studies),$(BENCH_STUDIES)); do \
	    echo "=== $$m — $$s ==="; \
	    case $$s in \
	      medical_entity_extraction) \
	        $(PYTHON) case_studies/$$s/run_study.py --model $$m --skip-rag || true ;; \
	      *) \
	        $(PYTHON) case_studies/$$s/run_study.py --model $$m || true ;; \
	    esac; \
	  done; \
	done
	@for s in $(or $(studies),$(BENCH_STUDIES)); do \
	  $(PYTHON) scripts/compare_models.py --study $$s; \
	done
	$(PYTHON) scripts/publish_reports.py
	@echo ""
	@echo "Benchmarks published to docs/. Review, then commit & push to deploy Pages:"
	@echo "    git add docs/ results/case_studies/ && git commit -m 'Update benchmarks' && git push"

# GitHub Actions locally (requires act)
test-ci:
	act -j test