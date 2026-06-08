.PHONY: help install install-dev test test-cov lint format clean docker-build docker-run deploy-modal run-medical-study run-rag-study publish publish-only

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
	@echo "  make publish model=gpt-4o-mini  - Run studies + publish reports to docs/"
	@echo "  make publish-only               - Publish existing reports (no re-run)"
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
	python scripts/run_diagnostics.py --task "$(task)" --model "$(model)"

improve:
	@echo "Running improvement strategy: $(strategy)"
	python scripts/run_improvements.py --strategy "$(strategy)" --case-study "$(case)"

report:
	@echo "Generating report..."
	python scripts/generate_report.py --case-study "$(case)"

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
	python case_studies/medical_entity_extraction/run_study.py --model $(or $(model),$(MODEL))

run-rag-study:
	python case_studies/rag_document_qa/run_study.py --model $(or $(model),$(MODEL))

# One command: run both studies for a model, then copy the freshest HTML reports
# into docs/ and regenerate the landing-page cards. Study failures (e.g. missing
# API key or RAG extra) are non-fatal so whatever ran still gets published.
# Usage:  make publish model=gpt-4o-mini
publish:
	-python case_studies/medical_entity_extraction/run_study.py --model $(or $(model),$(MODEL))
	-python case_studies/rag_document_qa/run_study.py --model $(or $(model),$(MODEL))
	python scripts/publish_reports.py
	@echo ""
	@echo "Reports published to docs/. Review, then commit & push to deploy Pages:"
	@echo "    git add docs/ && git commit -m 'Update published reports' && git push"

# Only (re)publish reports that already exist in results/, without re-running.
publish-only:
	python scripts/publish_reports.py

# GitHub Actions locally (requires act)
test-ci:
	act -j test