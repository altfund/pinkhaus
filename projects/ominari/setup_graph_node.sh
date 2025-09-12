#!/bin/bash
# Setup script for Graph Node and Thales subgraphs

set -e

echo "🚀 Setting up Graph Node infrastructure..."

# Check if docker and docker-compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p data/graph-node
mkdir -p subgraphs

# Check if .env file exists and has RPC URLs
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating with placeholder values..."
    cat > .env.graph << EOF
# Graph Node Configuration
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_KEY
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY
EOF
    echo "Please update .env.graph with your actual RPC URLs"
fi

# Start Graph Node infrastructure
echo "📦 Starting Graph Node infrastructure..."
docker-compose -f docker-compose.graph-node.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose -f docker-compose.graph-node.yml ps

# Check Graph Node health
echo "🏥 Checking Graph Node health..."
curl -s http://localhost:8030/ || echo "Graph Node not yet ready"

echo "✅ Graph Node infrastructure is starting up!"
echo ""
echo "Next steps:"
echo "1. Clone the Thales subgraph repository:"
echo "   git clone https://github.com/thales-markets/thales-subgraph.git subgraphs/thales-subgraph"
echo ""
echo "2. Deploy the subgraphs using deploy_subgraphs.sh"
echo ""
echo "📊 Monitoring:"
echo "- GraphQL endpoint: http://localhost:8000/"
echo "- Metrics: http://localhost:8040/metrics"
echo "- IPFS: http://localhost:8080/"
echo "- PostgreSQL: localhost:5432"