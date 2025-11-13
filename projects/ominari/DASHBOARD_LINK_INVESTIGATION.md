# Dashboard Link Investigation Results

## Key Findings

### Market IDs Are Not Blockchain Addresses

The IDs in our database like `v2_0x3230323530393230...` are **NOT blockchain addresses**. They are hex-encoded internal IDs.

**Example decoding:**
- `v2_0x3230323530393230393431313834373000000000000000000000000000000000`
- Hex part decodes to: `2025092094118470`
- Format appears to be: YYYYMMDD + unique identifier

### What This Means

1. **No Valid Blockchain Links**: These aren't Ethereum addresses, so we can't link to blockchain explorers
2. **Overtime Markets Links**: We're generating links like `https://overtimemarkets.xyz/markets/v2_0x...` but these may not work since they're internal IDs
3. **v2_ Prefix**: Likely indicates version 2 of the internal system, not v2 smart contracts

## Current Dashboard Status

- ✅ Clean soccer data (no American Football misclassifications)
- ✅ Proper nation/league filtering
- ✅ Real odds display
- ⚠️  Overtime Markets links generated but may not work
- ❌ No blockchain explorer links (IDs aren't blockchain addresses)

## Recommendation

Without real blockchain addresses or knowledge of Overtime Markets' URL structure, it's better to:
1. Not show broken links
2. Wait until we have proper market URLs or blockchain addresses
3. Focus on the working features: clean data and odds display