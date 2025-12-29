#!/bin/bash
# Setup script for LLM Diagnostic Framework

set -e  # Exit on error

echo "🚀 Setting up LLM Diagnostic Framework..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version OK: $python_version"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "✅ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
echo ""

# Install package in editable mode
echo "📦 Installing llm-diagnostic package..."
pip install -e . -q
echo "✅ Package installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p results/diagnostics
mkdir -p results/improvements
mkdir -p data/test_datasets
mkdir -p .cache
echo "✅ Directories created"
echo ""

# Setup .env file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please add your API keys to .env file"
else
    echo "✅ .env file already exists"
fi
echo ""

# Check if API keys are set
echo "🔑 Checking API keys..."
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    echo "⚠️  OPENAI_API_KEY not set in .env"
fi
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "sk-ant-..." ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set in .env"
fi
echo ""

# Run tests
echo "🧪 Running tests..."
if pytest tests/ -q --tb=no; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed (this is OK if API keys not set)"
fi
echo ""

echo "════════════════════════════════════════"
echo "✅ Setup complete!"
echo "════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Add your API keys to .env file"
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run example: make diagnose model=gpt-4-turbo-preview"
echo "  4. View help: make help"
echo ""
echo "Happy diagnosing! 🔍"