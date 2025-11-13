#!/bin/bash

echo "🚀 Starting local blockchain environment for Ominari DApp"
echo "================================================"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env.local exists, if not create from example
if [ ! -f ".env.local" ]; then
    echo "📝 .env.local not found, creating from example..."
    cp .env.example .env.local 2>/dev/null || echo "⚠️  No .env.example found, using defaults"
fi

# Start Hardhat node in the background
echo ""
echo "⛓️  Starting Hardhat local blockchain..."
echo "   Network: localhost"
echo "   URL: http://localhost:8545"
echo "   Chain ID: 31337"
echo ""

# Run hardhat node with initial balance
npx hardhat node --hostname 0.0.0.0 &
HARDHAT_PID=$!

# Wait for node to start
echo "⏳ Waiting for blockchain to start..."
sleep 5

# Check if node is running
if ! curl -s http://localhost:8545 > /dev/null; then
    echo "❌ Failed to start Hardhat node"
    exit 1
fi

echo "✅ Hardhat node is running (PID: $HARDHAT_PID)"
echo ""

# Deploy contracts
echo "📜 Deploying smart contracts..."
npx hardhat run scripts/deploy-local.js --network localhost

echo ""
echo "✨ Local blockchain environment is ready!"
echo ""
echo "📍 Important addresses:"
echo "   - Check deployments/localhost.json for contract addresses"
echo ""
echo "🔧 Available commands:"
echo "   - npm run test:local     (Run local tests)"
echo "   - npm run console:local  (Interactive console)"
echo "   - npm run frontend:dev   (Start frontend)"
echo ""
echo "⚠️  Keep this terminal open to maintain the blockchain"
echo "   Press Ctrl+C to stop"
echo ""

# Keep script running
wait $HARDHAT_PID