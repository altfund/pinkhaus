# Ominari DApp Deployment Configuration

## Smart Contract Deployment

### Networks Configuration
```javascript
// hardhat.config.js
module.exports = {
  solidity: "0.8.19",
  networks: {
    // Local Development
    hardhat: {
      chainId: 31337
    },
    
    // Testnets
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 11155111
    },
    polygonMumbai: {
      url: process.env.POLYGON_MUMBAI_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 80001
    },
    arbitrumGoerli: {
      url: process.env.ARBITRUM_GOERLI_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 421613
    },
    
    // Mainnets
    ethereum: {
      url: process.env.ETHEREUM_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 1
    },
    polygon: {
      url: process.env.POLYGON_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 137
    },
    arbitrum: {
      url: process.env.ARBITRUM_RPC_URL,
      accounts: [process.env.DEPLOYER_PRIVATE_KEY],
      chainId: 42161
    }
  }
};
```

### Deployment Script
```javascript
// scripts/deploy.js
const hre = require("hardhat");

async function main() {
  const network = hre.network.name;
  console.log(`Deploying to ${network}...`);
  
  // Deploy Kelly Optimizer
  const KellyOptimizer = await hre.ethers.getContractFactory("KellyOptimizer");
  const kellyOptimizer = await KellyOptimizer.deploy();
  await kellyOptimizer.deployed();
  console.log("KellyOptimizer deployed to:", kellyOptimizer.address);
  
  // Deploy Chunk Manager
  const ChunkManager = await hre.ethers.getContractFactory("ChunkManager");
  const chunkManager = await ChunkManager.deploy();
  await chunkManager.deployed();
  console.log("ChunkManager deployed to:", chunkManager.address);
  
  // Deploy Trading Engine
  const OminariTradingEngine = await hre.ethers.getContractFactory("OminariTradingEngine");
  const tradingEngine = await OminariTradingEngine.deploy(
    kellyOptimizer.address,
    chunkManager.address
  );
  await tradingEngine.deployed();
  console.log("OminariTradingEngine deployed to:", tradingEngine.address);
  
  // Save deployment addresses
  const fs = require("fs");
  const deployments = {
    network,
    timestamp: new Date().toISOString(),
    contracts: {
      kellyOptimizer: kellyOptimizer.address,
      chunkManager: chunkManager.address,
      tradingEngine: tradingEngine.address
    }
  };
  
  fs.writeFileSync(
    `deployments/${network}.json`,
    JSON.stringify(deployments, null, 2)
  );
}

main().catch(console.error);
```

## IPFS Deployment

### Frontend Build Configuration
```yaml
# .fleek.json
{
  "build": {
    "command": "npm run build",
    "output": "dist"
  },
  "publish": {
    "provider": "ipfs"
  }
}
```

### IPFS Upload Script
```bash
#!/bin/bash
# scripts/deploy-ipfs.sh

# Build frontend
npm run build

# Upload to IPFS via Pinata
IPFS_HASH=$(npx pinata-cli pin dist/)
echo "Frontend deployed to IPFS: $IPFS_HASH"

# Update ENS record
npx hardhat run scripts/update-ens.js --network ethereum
```

## TheGraph Configuration

### Subgraph Manifest
```yaml
# subgraph.yaml
specVersion: 0.2.0
description: Ominari Trading Protocol
repository: https://github.com/ominari/subgraph
schema:
  file: ./schema.graphql
dataSources:
  - kind: ethereum/contract
    name: OminariTradingEngine
    network: polygon
    source:
      address: "0x..." # Contract address
      abi: OminariTradingEngine
      startBlock: 12345678
    mapping:
      kind: ethereum/events
      apiVersion: 0.0.7
      language: wasm/assemblyscript
      entities:
        - TradingSession
        - Position
        - MarketData
      abis:
        - name: OminariTradingEngine
          file: ./abis/OminariTradingEngine.json
      eventHandlers:
        - event: SessionCreated(indexed uint256,indexed address,uint256)
          handler: handleSessionCreated
        - event: PositionPlaced(indexed uint256,indexed uint256,bytes32,uint256,uint8)
          handler: handlePositionPlaced
        - event: PositionSettled(indexed uint256,bool,uint256)
          handler: handlePositionSettled
      file: ./src/mappings.ts
```

## Docker Configuration (Minimal)

### Oracle Service Only
```dockerfile
# Dockerfile.oracle
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY oracle/ ./oracle/
CMD ["node", "oracle/index.js"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  oracle:
    build:
      context: .
      dockerfile: Dockerfile.oracle
    environment:
      - RPC_URL=${RPC_URL}
      - ORACLE_PRIVATE_KEY=${ORACLE_PRIVATE_KEY}
      - CONTRACT_ADDRESS=${CONTRACT_ADDRESS}
    restart: unless-stopped
```

## Environment Variables

### Production .env
```bash
# Network RPCs
ETHEREUM_RPC_URL=https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY

# Deployment Keys
DEPLOYER_PRIVATE_KEY=0x...
ORACLE_PRIVATE_KEY=0x...

# Contract Addresses (populated after deployment)
TRADING_ENGINE_ADDRESS=0x...
KELLY_OPTIMIZER_ADDRESS=0x...
CHUNK_MANAGER_ADDRESS=0x...

# IPFS
PINATA_API_KEY=...
PINATA_SECRET_KEY=...

# TheGraph
GRAPH_ACCESS_TOKEN=...

# ENS
ENS_NAME=ominari.eth
```

## GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy DApp

on:
  push:
    branches: [main]

jobs:
  deploy-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Install dependencies
        run: npm ci
      
      - name: Deploy to Polygon
        env:
          POLYGON_RPC_URL: ${{ secrets.POLYGON_RPC_URL }}
          DEPLOYER_PRIVATE_KEY: ${{ secrets.DEPLOYER_PRIVATE_KEY }}
        run: npx hardhat run scripts/deploy.js --network polygon
      
      - name: Verify contracts
        run: npx hardhat verify --network polygon
  
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build frontend
        run: |
          cd frontend
          npm ci
          npm run build
      
      - name: Deploy to IPFS
        env:
          PINATA_API_KEY: ${{ secrets.PINATA_API_KEY }}
          PINATA_SECRET_KEY: ${{ secrets.PINATA_SECRET_KEY }}
        run: |
          npx pinata-cli pin frontend/dist/
      
      - name: Update ENS
        env:
          ETHEREUM_RPC_URL: ${{ secrets.ETHEREUM_RPC_URL }}
          DEPLOYER_PRIVATE_KEY: ${{ secrets.DEPLOYER_PRIVATE_KEY }}
        run: npx hardhat run scripts/update-ens.js --network ethereum
```

## Monitoring & Analytics

### Dune Analytics Queries
```sql
-- Daily trading volume
SELECT 
  date_trunc('day', evt_block_time) as day,
  COUNT(*) as trades,
  SUM(stake) as volume
FROM ominari_tradingengine_evt_positionplaced
GROUP BY 1
ORDER BY 1 DESC;

-- User statistics
SELECT 
  trader,
  COUNT(DISTINCT session_id) as sessions,
  SUM(initial_bankroll) as total_deposited,
  AVG(final_bankroll - initial_bankroll) as avg_profit
FROM ominari_sessions
GROUP BY trader
ORDER BY sessions DESC;
```

### Health Check Endpoints
```javascript
// oracle/health.js
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    lastBlock: oracle.lastProcessedBlock,
    pendingSettlements: oracle.pendingQueue.length,
    uptime: process.uptime()
  });
});
```

## Cost Optimization

### Multi-chain Strategy
1. **Ethereum Mainnet**: Core trading engine only
2. **Polygon**: High-frequency operations
3. **Arbitrum**: Backup and overflow
4. **IPFS**: All static content
5. **TheGraph**: Decentralized indexing

### Gas Optimization
```solidity
// Use packed structs
struct PackedPosition {
    uint128 stake;
    uint64 odds;
    uint32 timestamp;
    uint8 outcome;
    bool isSettled;
    bool isWon;
}
```

## Security Measures

### Multi-signature Wallet
```javascript
// Deploy with Gnosis Safe
const safe = await Safe.create({
  ethAdapter,
  safeAddress: SAFE_ADDRESS,
  contractNetworks
});
```

### Upgradeable Contracts
```solidity
// Use OpenZeppelin upgradeable pattern
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
```

## Deployment Checklist

- [ ] Smart contracts audited
- [ ] Testnet deployment successful
- [ ] IPFS frontend tested
- [ ] TheGraph indexing verified
- [ ] Multi-sig wallet configured
- [ ] ENS domain configured
- [ ] Monitoring dashboards set up
- [ ] Documentation published
- [ ] Community announcement prepared
- [ ] Mainnet deployment executed