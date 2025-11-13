#!/bin/bash

echo "📜 Compiling Ominari Smart Contracts"
echo "===================================="

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required to compile contracts"
    echo "   Install with: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    exit 1
fi

# Check if Hardhat is installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Clean previous artifacts
echo "🧹 Cleaning previous artifacts..."
rm -rf artifacts cache

# Compile contracts
echo "🔨 Compiling contracts..."
npx hardhat compile

# Check compilation result
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    
    # Generate TypeScript types (if needed)
    if [ -f "node_modules/.bin/typechain" ]; then
        echo "📝 Generating TypeScript types..."
        npm run typechain
    fi
    
    # Copy ABIs to accessible location
    echo "📋 Extracting ABIs..."
    mkdir -p contracts/abis
    
    # Extract ABIs from artifacts
    for contract in OminariTradingEngine KellyOptimizer ChunkManager; do
        if [ -f "artifacts/contracts/core/$contract.sol/$contract.json" ]; then
            jq '.abi' "artifacts/contracts/core/$contract.sol/$contract.json" > "contracts/abis/$contract.json"
            echo "   ✓ $contract.json"
        fi
    done
    
    echo ""
    echo "✨ Contracts ready for deployment!"
    echo "   Run: npm run deploy:local"
else
    echo "❌ Compilation failed!"
    echo "   Check for errors above"
    exit 1
fi