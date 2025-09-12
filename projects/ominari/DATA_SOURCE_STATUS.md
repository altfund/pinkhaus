# Ominari Data Source & Trading Status Report

## Current Data Sources

### 1. REST API (Overtime) - ✅ ACTIVE
- **Status**: Working and pulling data every 5 minutes
- **Endpoint**: https://api.overtime.io/overtime-v2/networks/10/markets
- **Data Retrieved**: 101 active soccer markets
- **Script**: `free_data_pull.py` (v1 is working, v2 needs fixing)

### 2. GraphQL (The Graph) - ❌ DISABLED
- **Status**: Disabled via `FEATURE_GRAPHQL_STREAMING=False`
- **Reason**: The Graph endpoints were deprecated/moved
- **Original Endpoints**: 
  - https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism
  - https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum
- **Error**: "This endpoint has been removed"

### 3. Blockchain (Optimism) - ⚠️ RUNNING BUT NO DATA
- **Status**: Active but finding 0 trades/markets
- **Network**: Optimism (Chain ID: 10)
- **Contract**: SportsAMMV2 at 0x170a5714112daEfF20E798B6e92e25B86Ea603C1
- **Blocks Scanned**: ~140631000+ (current blocks)
- **Issue**: Either wrong contract address or no on-chain activity

## Paper Trading Status

### Configuration
- **Enabled**: Yes (default in config)
- **Initial Capital**: $10,000
- **Current Status**: Demo data only

### Paper Trading Files
- `paper_portfolio.json`: Contains demo portfolio with 2 pending bets
- `paper_trades.csv`: Contains 15 historical demo trades
- `paper_trading_engine.py`: Full engine implementation exists

### Paper Trading Flow
1. **Signal Generation**: Working (but only ImpliedRawSignal active)
2. **Position Sizing**: Uses Kelly criterion
3. **Trade Execution**: Code exists in `integrations.py` (_execute_paper_trades)
4. **Recording**: Should save to paper_trades database

### Issues
- No real paper trades being executed (only demo data)
- Signals returning no edge (implied probability = market probability)
- External signal providers (gRPC) not running

## Real Blockchain Trading Capture

### Current Status
- **Blockchain Reader**: Running every 5 minutes
- **Trades Found**: 0
- **Markets Found**: 0

### Possible Issues
1. Wrong contract address
2. No actual trades happening on-chain
3. Need to monitor different events
4. Should check Overtime's actual contract addresses

## Recommendations

1. **Fix Data Sources**:
   - Update GraphQL endpoints if new ones available
   - Verify correct blockchain contract addresses
   - Get v2 data pull script working

2. **Enable Real Signals**:
   - Start external signal providers (gRPC servers)
   - Or implement better internal signals with edge

3. **Paper Trading**:
   - Need signals with positive edge to trigger trades
   - Currently only using ImpliedRawSignal (0% edge)

4. **Blockchain Monitoring**:
   - Verify correct Overtime contract addresses
   - Check if trades are happening on different contracts
   - Consider monitoring SportPositionalMarket contracts