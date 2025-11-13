# Ominari DApp - Local Testing Guide

This guide will help you test the Ominari DApp locally before deploying to testnet or mainnet.

## Prerequisites

- Node.js v16+ installed
- Git
- A code editor (VS Code recommended)
- MetaMask browser extension

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Local Blockchain

In the first terminal window:
```bash
npm run blockchain:start
```

This will:
- Start a local Hardhat blockchain on `http://localhost:8545`
- Deploy all smart contracts
- Create test markets
- Save deployment addresses to `deployments/localhost.json`

Keep this terminal running!

### 3. Run Integration Tests

In a second terminal:
```bash
npm run test:local
```

This runs comprehensive tests including:
- Session creation
- Kelly optimization
- Bet placement
- Position settlement
- Security checks

### 4. Interactive Trading Test

Run the interactive trading simulation:
```bash
npm run trading:test
```

This will:
- Create sessions for test users (Alice & Bob)
- Show available markets
- Run Kelly optimization
- Place bets
- Simulate match results
- Show profits/losses

## Connect MetaMask

1. Open MetaMask
2. Add network:
   - Network Name: `Hardhat Local`
   - RPC URL: `http://localhost:8545`
   - Chain ID: `31337`
   - Currency Symbol: `ETH`

3. Import test accounts (private keys from Hardhat):
   ```
   Account #0: 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
   Account #1: 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
   Account #2: 0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a
   Account #3: 0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6
   ```

## Contract Addresses

After deployment, find your contract addresses in:
```
deployments/localhost.json
```

Example:
```json
{
  "contracts": {
    "tradingEngine": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
    "kellyOptimizer": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
    "chunkManager": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
  }
}
```

## Using Hardhat Console

For interactive contract testing:
```bash
npm run console:local
```

Example commands:
```javascript
// Get deployed contract
const TradingEngine = await ethers.getContractFactory("OminariTradingEngine");
const tradingEngine = TradingEngine.attach("0x5FbDB2315678afecb367f032d93F642f64180aa3");

// Create session
const tx = await tradingEngine.createSession(ethers.utils.parseEther("1.0"));
await tx.wait();

// Get session info
const session = await tradingEngine.getSession(1);
console.log(session);
```

## Test Scenarios

### Scenario 1: Basic Trading Flow
1. Create a session with 10 ETH
2. View available markets
3. Place a bet on home team
4. Wait for oracle to resolve market
5. Check winnings

### Scenario 2: Kelly Optimization
1. Create session with 100 ETH
2. Get 5 upcoming markets
3. Run Kelly optimization
4. Place recommended bets
5. Track portfolio performance

### Scenario 3: Multi-user Trading
1. Create sessions for 3 users
2. Each user places different bets
3. Resolve markets with different outcomes
4. Compare final bankrolls

## Frontend Testing (Coming Soon)

Start the frontend development server:
```bash
npm run frontend:dev
```

Navigate to `http://localhost:3000` and connect MetaMask.

## Django Integration Testing

1. Copy the Django app to altfund2:
```bash
cp -r altfund2_integration/ominari_trading /path/to/altfund2/
```

2. Run migrations:
```bash
cd /path/to/altfund2
python manage.py migrate ominari_trading
```

3. Start Django server:
```bash
python manage.py runserver
```

4. Access API at `http://localhost:8000/api/v1/ominari/`

## Troubleshooting

### "Nonce too high" error
Reset MetaMask account (Settings → Advanced → Reset Account)

### Contract deployment fails
Make sure the blockchain is running and accessible at `http://localhost:8545`

### Tests timeout
Increase timeout in test files:
```javascript
this.timeout(30000); // 30 seconds
```

### Port 8545 already in use
Kill existing process:
```bash
lsof -ti:8545 | xargs kill -9
```

## Next Steps

After successful local testing:
1. Deploy to Sepolia testnet
2. Test with testnet ETH
3. Conduct security audit
4. Deploy to mainnet

## Resources

- [Hardhat Documentation](https://hardhat.org/docs)
- [Ethers.js Documentation](https://docs.ethers.io/v5/)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/4.x/)
- [MetaMask Documentation](https://docs.metamask.io/)

## Support

For issues or questions:
- Check existing issues on GitHub
- Join our Discord community
- Email: support@ominari.io