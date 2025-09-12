#!/bin/bash
# Deploy Thales/Overtime subgraphs to local Graph Node

set -e

GRAPH_NODE_URL="http://localhost:8020/"
IPFS_URL="http://localhost:5001/"

echo "🚀 Deploying Thales/Overtime subgraphs to local Graph Node..."

# Check if subgraph directory exists
if [ ! -d "subgraphs/thales-subgraph" ]; then
    echo "📥 Cloning Thales subgraph repository..."
    mkdir -p subgraphs
    git clone https://github.com/thales-markets/thales-subgraph.git subgraphs/thales-subgraph
fi

cd subgraphs/thales-subgraph

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build and deploy Overtime Optimism subgraph
echo "🏗️ Building Overtime Optimism subgraph..."
npm run codegen:overtime-optimism || npm run codegen
npm run build:overtime-optimism || npm run build

echo "🚀 Deploying Overtime Optimism subgraph..."
npx graph create --node $GRAPH_NODE_URL overtime/optimism || true
npx graph deploy --node $GRAPH_NODE_URL --ipfs $IPFS_URL overtime/optimism

# Build and deploy Overtime Arbitrum subgraph
echo "🏗️ Building Overtime Arbitrum subgraph..."
# Modify subgraph.yaml for Arbitrum if needed
if [ -f "subgraph.overtime-arbitrum.yaml" ]; then
    cp subgraph.overtime-arbitrum.yaml subgraph.yaml
fi

npm run codegen
npm run build

echo "🚀 Deploying Overtime Arbitrum subgraph..."
npx graph create --node $GRAPH_NODE_URL overtime/arbitrum || true
npx graph deploy --node $GRAPH_NODE_URL --ipfs $IPFS_URL overtime/arbitrum

cd ../..

echo "✅ Subgraphs deployed successfully!"
echo ""
echo "GraphQL endpoints:"
echo "- Optimism: http://localhost:8000/subgraphs/name/overtime/optimism"
echo "- Arbitrum: http://localhost:8000/subgraphs/name/overtime/arbitrum"
echo ""
echo "Test with:"
echo 'curl -X POST http://localhost:8000/subgraphs/name/overtime/optimism -H "Content-Type: application/json" -d '"'"'{"query":"{ markets(first: 5) { id } }"}'"'"