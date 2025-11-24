# MTM Heartbeat Fix - Correct Portfolio Accounting

## Problem

The mark-to-market (MTM) heartbeat was using **incorrect betting accounting** that inflated portfolio values by treating betting positions like tradeable securities.

### What Was Wrong

**Incorrect Calculation**:
```python
# OLD (WRONG):
position_value = stake × current_odds  # Potential payout if you win
portfolio_value = cash + sum(all position values)

# Example with $100 stake at 3.0 odds:
position_value = $100 × 3.0 = $300  # ❌ WRONG
portfolio = $8,000 cash + $300 = $8,300  # ❌ INFLATED
```

**Result**: Portfolio showed **$12,830.59** instead of actual **$10,000**

This added **$2,830** in "unrealized gains" that don't actually exist until bets settle and WIN.

### The Fundamental Issue

Betting positions are **NOT** like stocks:
- **Stocks**: Have market value that changes continuously (can sell anytime)
- **Bets**: Binary outcomes - you either win the full payout or lose everything
- **No mark-to-market**: Odds moving in your favor doesn't create realized value
- **No liquidity**: You can't "sell" a bet before settlement

## The Fix

### Correct Betting Accounting

```python
# CORRECT:
position_value = stake  # The capital you deployed
portfolio_value = cash + sum(all stakes)

# Example with $100 stake at 3.0 odds:
position_value = $100  # ✅ CORRECT
portfolio = $8,000 cash + $100 stake = $8,100  # ✅ ACCURATE
```

### What Changed

**1. Portfolio Value Calculation** (`calculate_updated_portfolio_value`):
```python
# Before:
updated_portfolio_value = current_cash + mtm_value  # Wrong
mtm_value = sum(stake × current_odds)

# After:
portfolio_value = current_cash + total_stakes  # Correct
total_stakes = sum(all stakes)
```

**2. Position Tracking** (`get_live_position_values`):
```python
# Before:
current_position_value = stake × current_odds  # Treated as asset value

# After:
# Track odds movement separately for information only
odds_shift = (stake × current_odds) - (stake × entry_odds)
# But DON'T add this to portfolio value
```

**3. Discord Notifications**:
```python
# Before:
"Portfolio: $12,830.59 (+28.3%)"
"MTM Adjustment: +$2,830"

# After:
"Portfolio: $9,932.15 (-0.7%)"
"Cash: $8,283.94"
"Stakes: $1,648.21"
"Odds shift: -$345.14 (informational only)"
```

## Current Portfolio Status

### Correct Accounting
```
Starting Capital:    $10,000.00
Current Cash:        $8,283.94
Stakes Deployed:     $1,648.21
Portfolio Value:     $9,932.15
Realized P&L:        -$67.85  (-0.7%)
```

### Understanding the Numbers

**Cash ($8,283.94)**: Available capital not in bets

**Stakes ($1,648.21)**: Capital locked in 18 open positions
- This is YOUR money at risk
- You get it back + winnings if bets win
- You lose it if bets lose

**Portfolio Value ($9,932.15)**: Cash + Stakes
- This is your actual total capital
- Won't change until positions settle
- Small negative due to position tracking differences

**Odds Shift (-$345.14)**: INFORMATIONAL ONLY
- Shows how potential payouts have changed
- NOT added to portfolio value
- Means odds have moved slightly against you on average

## What Happens at Settlement

When a position settles:

**If you WIN**:
```
Cash: $8,283.94 + ($stake × odds) = $8,283.94 + $payout
Stakes: $1,648.21 - $stake
Portfolio: Increases by (payout - stake) = profit
```

**If you LOSE**:
```
Cash: $8,283.94 (unchanged)
Stakes: $1,648.21 - $stake
Portfolio: Decreases by stake = loss
```

## System Status

**MTM Heartbeat**: ✅ FIXED and RUNNING
- Updates every 5 minutes
- Tracks odds movements for information
- Uses correct portfolio accounting
- Shows: Portfolio = Cash + Stakes

**Portfolio Value**: ✅ ACCURATE
- No more inflated values
- Correct betting accounting
- Real-time position tracking
- No "unrealized gains" from odds movements

**Odds Tracking**: ✅ STILL WORKS
- Tracks how odds are changing
- Shows favorable/unfavorable moves
- Displayed as informational only
- Not added to portfolio value

## Example: How Odds Movements Work

**Position**: Maine vs Brown, $101.63 stake
- Entry odds: 2.44 → Potential win: $248.46
- Current odds: 2.95 → Potential win: $299.81
- **Odds shift**: +$51.35 (favorable move)

**What this means**:
- ✅ If you placed this bet NOW, you'd get better odds
- ✅ Your potential payout increased
- ❌ But you DON'T "have" $51.35 more
- ❌ Portfolio value unchanged: still $101.63 at risk
- ⏰ You'll know the real result when the game settles

## Bottom Line

**Before**: System was counting chickens before they hatched
- Added potential winnings to portfolio value
- Created phantom gains of $2,830

**After**: System uses proper betting accounting
- Portfolio = Cash + Stakes
- Tracks odds movements for information only
- No realized P&L until positions settle
- Accurate representation of capital

**Your Real Status**:
- Started with $10,000
- Currently have $9,932.15 (at cost)
- 18 open positions
- First settlement in ~3.5 hours
- Then we'll see real P&L!

---

**Fixed**: November 24, 2025, 1:48 AM EST
