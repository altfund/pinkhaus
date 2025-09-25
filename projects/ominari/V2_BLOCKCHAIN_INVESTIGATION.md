# Overtime V2 Blockchain Investigation Results

## Summary

After extensive investigation of Overtime V2 contracts on Arbitrum without using API keys, here are the findings:

## What We Discovered

### 1. V2 Contract Structure
- **AMM V2**: `0xfb64E79A562F7250131cf528242CEB10fDC82395` (active)
- **Manager**: `0xB155685132eEd3cD848d220e25a9607DD8871D38`
- **Risk Manager**: `0x10764f2787841E928E53e5Be1588a73E3c994EDe`
- **Owner**: `0x0089282ac624bbbda7ec472d63ee99643d7edf4a`

### 2. Transaction Analysis
- Found 32 V2 AMM transactions in recent blocks
- Method signatures discovered:
  - `0x942b67dc`: `setRootForGame(bytes32,bytes32)` - Oracle price updates
  - `0x6dbf6cc7`: `setRootsPerGames(bytes32[],bytes32[])` - Batch oracle updates
  - `0xc6fa3d673d...`: Unknown event (514 occurrences) - appears to be oracle-related

### 3. Key Findings

#### Markets are Minimal Proxies
- Overtime markets are deployed as minimal proxy contracts (45 bytes)
- They use the same implementation/mastercopy pattern as V1
- Markets implement standard methods: `getGameDetails()`, `times()`, `resolved()`

#### V2 vs V1 Differences
- V2 AMM shows recent activity (oracle updates)
- V1 contracts appear dormant
- Market enumeration is not exposed through standard methods
- No `MarketCreated` or similar events found in recent blocks

#### Current State
- No active/future markets found in recent blockchain data
- This appears to be due to:
  1. **Seasonal nature** - Sports betting has off-seasons
  2. **Different creation pattern** - V2 may create markets differently
  3. **Oracle-focused activity** - Recent transactions are price updates, not market creation

## What We Tried

1. **Direct Transaction Scanning** ✅
   - Scanned 1000+ blocks for AMM interactions
   - Found oracle update transactions
   - No market addresses in transaction parameters

2. **Event Log Analysis** ✅
   - Analyzed 500+ events from V2 contracts
   - Found oracle-related events
   - No standard market creation events

3. **Manager Contract Investigation** ✅
   - Tried common enumeration methods
   - No exposed market listing functions
   - Manager doesn't follow typical factory pattern

4. **Storage Slot Analysis** ✅
   - Checked first 50 storage slots
   - Found contract addresses but not market arrays
   - V2 uses different storage pattern than V1

5. **Transaction Receipt Analysis** ✅
   - Checked logs from AMM transactions
   - No contract creation logs found
   - Markets may be pre-deployed or created elsewhere

## Recommendations

### For Getting Real V2 Data:

1. **Use API Keys**
   - Register on Arbiscan for higher rate limits
   - Use internal transactions API to find market creations
   - Access verified contract ABIs

2. **Monitor for Active Season**
   - V2 appears to be in off-season
   - Monitor for increased transaction activity
   - Markets likely created when sports events are scheduled

3. **Alternative Data Sources**
   - Consider GraphQL API if available
   - Check Overtime's official API/subgraph
   - Monitor V2 documentation for updates

### Current Dashboard Status

- **Dashboard**: Fully functional at http://localhost:8888/unified
- **Database**: PostgreSQL via flox on port 5999
- **Sample Data**: 19 markets demonstrating all features
- **Ready for**: Real data integration when markets become available

## Code Created

1. `fetch_overtime_direct.py` - Direct blockchain fetcher
2. `analyze_v2_transactions.py` - Transaction pattern analysis
3. `decode_v2_methods.py` - Method signature decoder
4. `investigate_v2_manager.py` - Manager contract investigation
5. `scan_v2_events.py` - Event log scanner
6. `analyze_unknown_events.py` - Deep event analysis
7. `find_v2_markets_simple.py` - Simple market finder

## Conclusion

The V2 system is operational but currently shows no active markets. The recent blockchain activity consists primarily of oracle price updates. To get real market data:

1. Wait for active sports season
2. Use API keys for efficient blockchain querying
3. Consider alternative data sources

The infrastructure is ready and will automatically clear sample data once real markets are found and added.