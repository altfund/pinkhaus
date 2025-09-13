#!/bin/bash
# Quick deployment script for new machines

set -e

echo "🚀 Ominari Trading System - New Machine Setup"
echo "=============================================="

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git first."  
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
uv sync

# Setup environment
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp .env.production .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env file with your configuration"
else
    echo "✅ .env file already exists"
fi

# Start PostgreSQL
echo "🗄️ Starting PostgreSQL container..."
if ! docker ps | grep -q "ominari-postgres-hybrid"; then
    docker run -d --name ominari-postgres-hybrid \
        -p 5435:5432 \
        -e POSTGRES_USER=ominari_user \
        -e POSTGRES_PASSWORD=ominari_2025_secure \
        -e POSTGRES_DB=ominari_production \
        -v ominari_postgres_data:/var/lib/postgresql/data \
        postgres:15-alpine
    
    echo "⏳ Waiting for PostgreSQL to start..."
    sleep 10
    echo "✅ PostgreSQL container started"
else
    echo "✅ PostgreSQL container already running"
fi

# Initialize databases
echo "🔧 Initializing database schema..."
uv run python setup_postgresql_hybrid.py

# Test hybrid access
echo "🧪 Testing hybrid database access..."
uv run python hybrid_database_access.py

echo ""
echo "✅ Deployment Complete!"
echo "======================="
echo ""
echo "📝 Next Steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Start the trading system:"
echo "   uv run python ominari_unified.py"
echo ""
echo "3. For paper trading only:"
echo "   uv run python paper_trading_engine.py"
echo ""
echo "4. To test blockchain sync:"
echo "   uv run python blockchain_sync_daemon.py"
echo ""
echo "📊 Health Checks:"
echo "- PostgreSQL: docker ps | grep postgres-hybrid"
echo "- System health: uv run python system_health_check.py"
echo "- Database stats: uv run python hybrid_database_access.py"
echo ""
echo "⚠️  IMPORTANT: Paper trading is enabled by default."
echo "   Only disable it when you're ready for live trading!"
echo ""