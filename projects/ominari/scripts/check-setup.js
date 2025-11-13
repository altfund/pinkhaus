#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log("🔍 Checking Ominari DApp Local Setup");
console.log("====================================\n");

let errors = [];
let warnings = [];

// 1. Check Node.js version
try {
  const nodeVersion = process.version;
  console.log(`✓ Node.js version: ${nodeVersion}`);
  
  const majorVersion = parseInt(nodeVersion.split('.')[0].substring(1));
  if (majorVersion < 16) {
    errors.push("Node.js version must be 16 or higher");
  }
} catch (e) {
  errors.push("Could not check Node.js version");
}

// 2. Check if npm packages are installed
const nodeModulesExists = fs.existsSync(path.join(__dirname, '../node_modules'));
if (nodeModulesExists) {
  console.log("✓ Node modules installed");
} else {
  warnings.push("Node modules not installed. Run: npm install");
}

// 3. Check for required files
const requiredFiles = [
  'hardhat.config.js',
  'package.json',
  '.env.local',
  'contracts/core/OminariTradingEngine.sol',
  'contracts/core/KellyOptimizer.sol',
  'contracts/core/ChunkManager.sol'
];

console.log("\n📁 Checking required files:");
requiredFiles.forEach(file => {
  const filePath = path.join(__dirname, '..', file);
  if (fs.existsSync(filePath)) {
    console.log(`  ✓ ${file}`);
  } else {
    if (file === '.env.local') {
      warnings.push(`.env.local not found. Copy from .env.example or let setup create it`);
    } else {
      errors.push(`Missing required file: ${file}`);
    }
  }
});

// 4. Check if contracts can compile
if (nodeModulesExists) {
  console.log("\n📜 Checking contract compilation:");
  try {
    execSync('npx hardhat compile', { stdio: 'pipe' });
    console.log("  ✓ Contracts compile successfully");
  } catch (e) {
    errors.push("Contract compilation failed. Check for syntax errors.");
  }
}

// 5. Check if local blockchain is running
console.log("\n🔗 Checking blockchain connection:");
try {
  const response = execSync('curl -s -X POST -H "Content-Type: application/json" --data \'{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}\' http://localhost:8545', { stdio: 'pipe' });
  const result = JSON.parse(response.toString());
  if (result.result) {
    console.log("  ✓ Local blockchain is running");
    console.log(`  Current block: ${parseInt(result.result, 16)}`);
  }
} catch (e) {
  warnings.push("Local blockchain not running. Run: npm run blockchain:start");
}

// 6. Check for existing deployment
const deploymentPath = path.join(__dirname, '../deployments/localhost.json');
if (fs.existsSync(deploymentPath)) {
  console.log("\n📋 Found existing deployment:");
  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf-8'));
  console.log(`  ✓ Trading Engine: ${deployment.contracts.tradingEngine}`);
  console.log(`  ✓ Kelly Optimizer: ${deployment.contracts.kellyOptimizer}`);
  console.log(`  ✓ Chunk Manager: ${deployment.contracts.chunkManager}`);
} else {
  warnings.push("No local deployment found. Deploy with: npm run deploy:local");
}

// Summary
console.log("\n📊 Summary:");
console.log("===========");

if (errors.length === 0 && warnings.length === 0) {
  console.log("✅ Everything looks good! You're ready to test locally.");
  console.log("\nNext steps:");
  console.log("1. Start blockchain: npm run blockchain:start");
  console.log("2. Run tests: npm run test:local");
  console.log("3. Try trading: npm run trading:test");
} else {
  if (errors.length > 0) {
    console.log("\n❌ Errors found:");
    errors.forEach(e => console.log(`  - ${e}`));
  }
  
  if (warnings.length > 0) {
    console.log("\n⚠️  Warnings:");
    warnings.forEach(w => console.log(`  - ${w}`));
  }
  
  console.log("\n📖 See LOCAL_TESTING_GUIDE.md for detailed setup instructions");
}

process.exit(errors.length > 0 ? 1 : 0);