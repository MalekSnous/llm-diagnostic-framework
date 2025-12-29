FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY llm_diagnostic/ llm_diagnostic/
COPY scripts/ scripts/
COPY case_studies/ case_studies/

# Install package
RUN pip install -e .

# Create directories for results
RUN mkdir -p /app/results/diagnostics /app/results/improvements

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RESULTS_DIR=/app/results

# Expose port for web app (if needed)
EXPOSE 8000

# Default command
CMD ["python", "-m", "llm_diagnostic"]