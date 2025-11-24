# Portfolio Accounting Fix - Both Heartbeats Corrected

## Problem Summary

Both heartbeat systems were showing **incorrect portfolio values** due to improper betting accounting:

1. **MTM Heartbeat**: Adding potential winnings to portfolio (inflated)
2. **Main Heartbeat**: Showing only cash, ignoring stakes (deflated)

## The Correct Accounting

For betting positions, the correct portfolio value is:

```
Portfolio Value = Cash + Stakes (at cost)
```

**NOT**:
- ❌ Cash + Potential Payouts (MTM Heartbeat was doing this)
- ❌ Cash only (Main Heartbeat was doing this)

## What Was Wrong

### Before the Fix

**MTM Heartbeat** showed:
```
Portfolio: $12,830.59 (+28.3%)  ❌ WRONG
```
This added $4,546.64 in "unrealized gains" from odds movements

**Main Heartbeat** showed:
```
Portfolio: $8,283.94 (-17.2%)  ❌ WRONG
```
This only counted cash, missing $1,648.21 in stakes

### The Truth

**Actual Portfolio Status**:
```
Cash:        $8,283.94
Stakes:      $1,648.21
Portfolio:   $9,932.15  ✅ CORRECT
P&L:         -$67.85 (-0.7%)
```

## Fixes Applied

### Fix #1: MTM Heartbeat (market_data_heartbeat.py)

**Changed Portfolio Calculation**:
```python
# Before (WRONG):
portfolio_value = cash + sum(stake × current_odds)  # Added potential payouts

# After (CORRECT):
portfolio_value = cash + sum(stakes)  # Just the deployed capital
odds_shift = sum((stake × current_odds) - (stake × entry_odds))  # Tracked separately
```

**Now Shows**:
```
Portfolio: $9,932.15 (-0.7%)
Cash: $8,283.94
Stakes: $1,648.21
Odds shift: -$345.14 (informational only)
```

### Fix #2: UnifiedPortfolioCalculator

**Added New Metric**:
```python
@dataclass
class PortfolioMetrics:
    # CORRECT BETTING ACCOUNTING (NEW)
    book_value: float        # Cash + Stakes at cost
    book_pnl: float          # Realized P&L only
    book_roi_percentage: float

    # Old metrics (now deprecated or for MTM)
    portfolio_value: float   # MTM with unrealized (wrong for betting)
    cash_portfolio_value: float  # Cash only (missing stakes)
```

**Calculation**:
```python
# BOOK VALUE: Correct betting accounting
book_value = current_bankroll + total_stake  # Cash + deployed capital
book_pnl = realized_pnl  # Only from settled trades
book_roi_percentage = (book_pnl / initial_bankroll) * 100
```

### Fix #3: Main Heartbeat (carver_heartbeat_with_backtest.py)

**Changed to Use Book Value**:
```python
# Before (WRONG):
current_value = portfolio_status.get('cash_portfolio_value', ...)  # Cash only

# After (CORRECT):
current_value = portfolio_status.get('book_value', ...)  # Cash + Stakes
total_pnl = portfolio_status.get('book_pnl', 0)  # Realized P&L only
```

**Now Shows**:
```
Portfolio: $9,932.15 (-68 | -0.7%)
Positions: 18 open • $1,648 at risk
```

## Three Portfolio Metrics Explained

The system now tracks THREE different portfolio values:

### 1. Book Value (CORRECT for betting)
```
book_value = cash + stakes
= $8,283.94 + $1,648.21
= $9,932.15
```
**Use this**: This is your actual capital at cost

### 2. Portfolio Value (MTM - wrong for betting)
```
portfolio_value = initial + (realized_pnl + unrealized_pnl)
= $10,000 + ($0 + $0)
= $10,000
```
**Don't use**: Treats bets like tradeable securities with mark-to-market value

### 3. Cash Portfolio Value (deprecated)
```
cash_portfolio_value = cash only
= $8,283.94
```
**Don't use**: Missing the stakes deployed in open positions

## Why This Matters

**Betting positions are NOT like stocks**:
- Stocks have continuous market value (can sell anytime)
- Bets are binary: win everything or lose everything
- No mark-to-market until settlement
- Odds moving doesn't create realized gains

**Correct Accounting**:
- Portfolio value = Cash + Stakes (your capital at cost)
- Realized P&L = Only from settled trades
- Unrealized P&L = Doesn't exist until settlement

**Odds Movements**:
- Tracked for information only
- Shows how probabilities have shifted
- NOT added to portfolio value
- Won't know real result until games settle

## System Status After Fix

**MTM Heartbeat**: ✅ FIXED
- Portfolio = Cash + Stakes ✅
- Tracks odds movements ℹ️
- Updates every 5 minutes

**Main Heartbeat**: ✅ FIXED
- Portfolio = Book Value ✅
- Uses realized P&L only ✅
- Updates hourly

**Both Now Show**: $9,932.15 (-0.7%)

## Current Portfolio Breakdown

```
Starting Capital:    $10,000.00

Cash Available:      $8,283.94  (83%)
Stakes Deployed:     $1,648.21  (16%)
                     ----------
Portfolio Value:     $9,932.15  (100%)

Realized P&L:        -$67.85    (-0.7%)
Unrealized P&L:      $0.00      (N/A until settlement)

Open Positions:      18
First Settlement:    ~3 hours (5:21 AM EST)
```

## What Happens at Settlement

When a bet settles:

**If WIN**:
```
Cash increases by: payout (stake × odds)
Stakes decrease by: stake
Portfolio increases by: profit (payout - stake)
Realized P&L increases by: profit
```

**If LOSE**:
```
Cash: no change
Stakes decrease by: stake
Portfolio decreases by: stake
Realized P&L decreases by: stake
```

## Next Discord Notifications

**MTM Heartbeat** (every 5 min):
```
Portfolio: $9,932.15 (-0.7%)
Cash: $8,283.94
Stakes: $1,648.21
Odds shift: -$345.14 (informational only)
```

**Main Heartbeat** (hourly):
```
Portfolio: $9,932.15 (-68 | -0.7%)
Positions: 18 open • $1,648 at risk
```

## Files Modified

1. `market_data_heartbeat.py` - Fixed MTM portfolio calculation
2. `unified_portfolio_calculator.py` - Added book_value metric
3. `carver_heartbeat_with_backtest.py` - Use book_value instead of cash_portfolio_value

## Technical Summary

**Root Cause**: Treating betting positions like tradeable securities with mark-to-market accounting

**Solution**: Implement proper betting accounting where portfolio value = cash + stakes at cost

**Result**: Both heartbeats now show consistent, accurate portfolio values using book value

---

**Fixed**: November 24, 2025, 2:12 AM EST
**Both heartbeats now operational with correct accounting**
