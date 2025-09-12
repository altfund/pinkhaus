# P&L Data Repair Summary

## The Problem

The user correctly identified that something was wrong with the P&L calculations. Upon investigation, I found that the data was severely corrupted:

1. **Final values were 0 for winning positions** - This should have been `stake × odds`
2. **Execution stakes were inconsistent** - Sometimes less than stake (impossible with fees)
3. **P&L values didn't match any logical calculation**

## Root Cause

The data corruption appears to have happened after positions were closed. The calculation logic in `paper_trading_sessions.py` is actually correct:
- `final_value = stake × odds` for wins
- `pnl = final_value - execution_stake`

But somehow the stored data had:
- `final_value = 0` for wins
- Arbitrary P&L values that didn't match the formula

## The Solution

Created `repair_pnl_data.py` which:

1. **Fixed execution stakes** - Ensured `execution_stake = stake + fees`
2. **Fixed final values**:
   - Wins: `final_value = stake × odds`
   - Losses: `final_value = 0`
3. **Recalculated P&L correctly**:
   - Wins: `P&L = (stake × odds) - execution_stake`
   - Losses: `P&L = -execution_stake`

## Results

### Before Repair:
- Total P&L: -$4,931.79 (incorrect)
- Portfolio Value: $5,068.21
- Many positions had illogical P&L values

### After Repair:
- Total P&L: -$4,873.43 (correct)
- Portfolio Value: $5,126.57
- All P&L values now follow the correct formula

### What Changed:
- Fixed 115 out of 144 P&L values
- Fixed 117 execution stakes
- Fixed 32 final values (all the wins)

## Correct P&L Formula

**For Wins:**
```
P&L = (stake × odds) - execution_stake
```
Where:
- `stake` = the base amount bet
- `odds` = the decimal odds
- `execution_stake` = stake + fees

**For Losses:**
```
P&L = -execution_stake
```

## Example

Bet $100 at 2.0 odds with 3% fee:
- Stake: $100
- Fees: $3
- Execution stake: $103

If win:
- Gross payout: $100 × 2.0 = $200
- P&L: $200 - $103 = $97 profit

If lose:
- P&L: -$103 (lose stake + fees)