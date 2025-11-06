# Ominari DApp Architecture & Altfund2 Integration Plan

## Overview
This document outlines the transformation of Ominari into a decentralized application (DApp) with minimal Web2 infrastructure and compatibility with Altfund2's Django architecture.

## Current State Analysis

### Ominari System Components
- **Web Dashboard**: Flask/SocketIO running on port 8888
- **PostgreSQL Database**: Centralized data storage (flox postgres on port 5999)
- **Trading Engine**: Python-based Kelly optimization & portfolio management
- **Blockchain Integration**: Read-only via Overtime Markets protocol
- **Paper Trading**: Session management with PostgreSQL backend

### Altfund2 Architecture (Django-based)
- **Django 4.2**: Web framework with multi-app structure
- **Celery**: Task queue for async operations (can be leveraged for blockchain ops)
- **PostgreSQL**: Primary database (similar to Ominari)
- **API Layer**: Django REST framework
- **Docker/AWS ECS**: Current deployment infrastructure
- **Payment Integration**: Stripe via djstripe

## DApp Transformation Strategy

### Phase 1: Smart Contract Development
1. **Core Trading Contracts**
   ```solidity
   - OminariTradingEngine.sol: Main trading logic
   - KellyOptimizer.sol: On-chain Kelly criterion calculations
   - DynamicChunkManager.sol: Time-based market grouping
   - CapitalStateManager.sol: Multi-state capital tracking
   ```

2. **Data Oracle Integration**
   - Use existing Overtime Markets oracles
   - Chainlink price feeds for additional data
   - Custom oracle for off-chain computation results

3. **Session & Position Management**
   - On-chain paper trading sessions
   - Position NFTs for trade representation
   - Settlement automation via smart contracts

### Phase 2: Decentralized Storage Layer
1. **IPFS/Filecoin Integration**
   - Historical market data archival
   - Trading reports and analytics
   - Dashboard static assets

2. **Ceramic Network**
   - User preferences and settings
   - Trading strategies configuration
   - Session metadata

3. **TheGraph Protocol**
   - Index blockchain events
   - Query historical positions
   - Real-time market data aggregation

### Phase 3: Web3 Frontend Migration
1. **Progressive Web3 Integration**
   ```javascript
   - Web3Modal for wallet connection
   - Ethers.js for contract interaction
   - WalletConnect for mobile support
   ```

2. **Hybrid Architecture**
   - Keep Flask/SocketIO for real-time updates
   - Add Web3 provider abstraction layer
   - Gradual migration of features

3. **State Management**
   - On-chain: Critical trading state
   - IPFS: Large datasets and history
   - Local: UI state and caching

### Phase 4: Altfund2 Integration

1. **Django App Creation**
   ```python
   # altfund2/ominari_trading/
   - models.py: Bridge between Django ORM and blockchain
   - tasks.py: Celery tasks for blockchain operations
   - views.py: API endpoints for DApp interaction
   - blockchain.py: Web3 integration layer
   ```

2. **Shared Infrastructure**
   - Use Altfund2's Celery for async blockchain ops
   - Leverage existing user authentication
   - Integrate with payment system for premium features

3. **API Compatibility Layer**
   ```python
   # RESTful endpoints for DApp
   /api/v1/ominari/positions/
   /api/v1/ominari/markets/
   /api/v1/ominari/optimize/
   /api/v1/ominari/sessions/
   ```

### Phase 5: Decentralized Deployment

1. **Smart Contract Deployment**
   - Ethereum Mainnet for production
   - Polygon for cost-effective operations
   - Arbitrum for high-frequency updates

2. **Frontend Hosting**
   - IPFS pinning via Pinata/Infura
   - ENS domain for decentralized naming
   - Fleek for automatic IPFS deployment

3. **Minimal Infrastructure**
   ```yaml
   # Required Web2 components (minimal)
   - RPC endpoint aggregator (Alchemy/Infura fallback)
   - WebSocket proxy for real-time (can be P2P later)
   - Initial deployment orchestration only
   ```

## Implementation Roadmap

### Week 1-2: Smart Contract Development
- [ ] Core trading logic contracts
- [ ] Test suite on testnet
- [ ] Security audit preparation

### Week 3-4: Storage Layer
- [ ] IPFS integration for data
- [ ] TheGraph subgraph creation
- [ ] Ceramic schemas design

### Week 5-6: Frontend Web3 Migration
- [ ] Wallet integration
- [ ] Contract interaction layer
- [ ] Hybrid operation mode

### Week 7-8: Altfund2 Integration
- [ ] Django app creation
- [ ] API compatibility layer
- [ ] Shared authentication

### Week 9-10: Testing & Deployment
- [ ] Testnet deployment
- [ ] Integration testing
- [ ] Mainnet preparation

## Technical Architecture

### Smart Contract Architecture
```
┌─────────────────────────────────────┐
│     OminariTradingEngine.sol        │
├─────────────────────────────────────┤
│ - createSession()                   │
│ - placeBet()                        │
│ - optimizePortfolio()               │
│ - settlePositions()                 │
└─────────────────────────────────────┘
           │
           ├── KellyOptimizer.sol
           ├── ChunkManager.sol
           └── OracleInterface.sol
```

### Data Flow Architecture
```
User → DApp Frontend → Smart Contracts → Blockchain
         ↓                    ↓              ↓
    Local Storage         TheGraph       Events
         ↓                    ↓              ↓
      IPFS/Ceramic    Indexed Data    Oracle Updates
```

### Integration with Altfund2
```
Altfund2 Django
    │
    ├── ominari_trading/ (new Django app)
    │     ├── blockchain.py (Web3 integration)
    │     ├── api/views.py (DApp REST API)
    │     └── tasks.py (Celery blockchain tasks)
    │
    └── existing apps (accounts, payments, etc.)
```

## Security Considerations

1. **Smart Contract Security**
   - Multi-sig admin functions
   - Upgradeable proxy pattern
   - Comprehensive test coverage
   - Professional audit required

2. **Oracle Security**
   - Multiple data source validation
   - Dispute resolution mechanism
   - Fallback data providers

3. **User Security**
   - Non-custodial design
   - Session timeout mechanisms
   - Secure key management guidance

## Cost Analysis

### One-time Costs
- Smart contract deployment: ~$500-2000 (mainnet)
- Audit: $10,000-30,000
- IPFS pinning setup: $100/month

### Ongoing Costs (Minimal)
- IPFS storage: ~$50-100/month
- RPC endpoints: $50-200/month (or use public)
- TheGraph indexing: $100-500/month

### User Costs
- Gas fees per transaction
- No monthly hosting fees
- Optional premium features via Altfund2

## Migration Strategy

1. **Parallel Operation**
   - Run DApp alongside current system
   - Gradual feature migration
   - User opt-in to DApp mode

2. **Data Migration**
   - Export historical data to IPFS
   - Create blockchain snapshots
   - Maintain data integrity

3. **User Migration**
   - Wallet creation assistance
   - Gas fee education
   - Incentive program for early adopters

## Next Steps

1. **Immediate Actions**
   - Set up development environment for smart contracts
   - Create initial contract interfaces
   - Design TheGraph schemas

2. **Technical Decisions**
   - Choose primary blockchain (Ethereum vs Polygon)
   - Select IPFS pinning service
   - Decide on oracle provider

3. **Integration Planning**
   - Meet with Altfund2 team
   - Design API specifications
   - Plan authentication bridge

## Conclusion

This DApp transformation will make Ominari:
- **Decentralized**: Minimal reliance on centralized infrastructure
- **Cost-effective**: Reduced operational costs
- **Censorship-resistant**: No single point of failure
- **Integrated**: Seamless compatibility with Altfund2
- **Future-proof**: Built on Web3 standards

The architecture ensures that Ominari becomes a true DApp while maintaining compatibility with Altfund2's existing Django infrastructure, creating a bridge between Web2 and Web3 paradigms.