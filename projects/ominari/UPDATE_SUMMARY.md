# Ominari Configuration Updates

## Contract Address Update ✅
- **Old Address**: 0x170a5714112daEfF20E798B6e92e25B86Ea603C1 (incorrect)
- **New Address**: 0xFb4e4811C7A811E098A556bD79B64c20b479E431 (correct SportsAMMV2)
- **Source**: https://v2.contracts.overtime.io/
- **Updated in**: config.py and .env

## GraphQL Status ⚠️
- **Current Status**: The Graph hosted service is deprecated
- **Old Endpoints**: 
  - https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism
  - https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum
- **New Status**: Need to find decentralized network endpoints or use alternative

## Current Data Flow
1. **REST API**: ✅ Working (https://api.overtime.io/)
2. **Blockchain**: ⚠️ Correct contract but low activity (0 trades in last 1000 blocks)
3. **GraphQL**: ❌ Disabled (endpoints deprecated)

## Recommendations
1. **For Blockchain Data**:
   - The contract address is now correct
   - May need to scan larger block ranges or specific times when trades occur
   - Consider monitoring market creation events instead of just trades

2. **For GraphQL**:
   - Could re-enable if we find working endpoints
   - The subgraphs may have moved to decentralized network
   - Alternative: Rely on REST API + blockchain events

3. **For Paper Trading**:
   - Need to enable signal providers with actual edge
   - Currently only ImpliedRawSignal (0% edge) is active
   - External gRPC signals need to be started

## Next Steps
1. Restart the system to use the new contract address
2. Monitor blockchain for any market/trade activity
3. Enable better signal providers for paper trading to work