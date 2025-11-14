#!/bin/bash
# Setup script for Ominari development environment

echo "🚀 Setting up Ominari development environment..."

# Check if virtual environment exists
if [ ! -f ".venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync --all-extras

# Install pre-commit
echo "Installing pre-commit..."
pip install pre-commit

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create secrets baseline if it doesn't exist
if [ ! -f ".secrets.baseline" ]; then
    echo "Creating secrets baseline..."
    detect-secrets scan > .secrets.baseline || true
fi

# Run initial checks
echo "Running initial checks..."
pre-commit run --all-files || true

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. The pre-commit hook will run automatically before each commit"
echo "2. To run checks manually: pre-commit run --all-files"
echo "3. To skip hooks temporarily: git commit --no-verify"
echo "4. Dashboard is at: http://localhost:8888"
echo ""
echo "🔄 Git workflow:"
echo "- feature/* branches → develop (staging deployment)"
echo "- develop → main (production deployment)"
echo "- All commits to main trigger automatic production deployment"