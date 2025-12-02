# Strategy Testing Pipeline Guide

## Overview

This guide explains how to safely test new trading strategies before risking real money. The pipeline has **6 progressive stages**, each designed to filter out bad strategies with increasing levels of realism and lower risk.

## The 6-Stage Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Backtest│ --> │ 2. Paper     │ --> │ 3. Shadow    │
│  Historical │     │ Trading      │     │ Trading      │
│  Data       │     │ Live Data    │     │ vs Production│
└─────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 v
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ 6. Gradual  │ <-- │ 5. Full A/B  │ <-- │ 4. Micro A/B │
│ Rollout     │     │ Test (20%)   │     │ Test (1-5%)  │
│ 100%        │     │ Real Money   │     │ Real Money   │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Stage 1: Backtest (Historical Data)

**Purpose**: Quick validation against historical data
**Risk**: None (simulated)
**Duration**: Minutes to hours
**Data**: Historical market and odds data from database

### How It Works

The backtester replays historical markets and simulates what trades your strategy would have made:

```python
from strategy_backtester import run_strategy_backtest
from datetime import datetime, timedelta, timezone

# Backtest over last 30 days
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=30)

results = run_strategy_backtest("1.1.0", start_date, end_date)
```

### Promotion Criteria

Must meet these requirements to advance:
- ✅ Minimum 100 trades
- ✅ ROI ≥ 5%
- ✅ Max drawdown ≤ 25%

### Pros & Cons

**Pros:**
- Fast (minutes to hours)
- No risk
- Can test many variants quickly
- Good for filtering obviously bad strategies

**Cons:**
- Historical data may not represent future
- Risk of overfitting
- No real market dynamics (liquidity, odds movements)
- Simulated outcomes (not actual results)

### Best Practices

1. **Use Recent Data**: Last 30-90 days is most relevant
2. **Test Multiple Periods**: Don't just test bull markets
3. **Watch for Overfitting**: If it's "too good to be true" it probably is
4. **Focus on Risk Metrics**: Drawdown matters more than returns

## Stage 2: Paper Trading (Live Data, No Execution)

**Purpose**: Test with live market data but no real money
**Risk**: None (simulated)
**Duration**: 1-2 weeks
**Data**: Live odds and markets from blockchain/APIs

### How It Works

Your strategy runs with real-time data but doesn't execute trades:

```python
from enhanced_trading_engine import EnhancedTradingEngine

# Paper trading with v1.1.0
engine = EnhancedTradingEngine(
    strategy_version="1.1.0",
    enable_ab_testing=False  # Just paper trading
)

await engine.run_trading_loop()
```

### Promotion Criteria

- ✅ Minimum 50 trades
- ✅ ROI ≥ 3%
- ✅ Max drawdown ≤ 20%
- ✅ Minimum 7 days duration

### Pros & Cons

**Pros:**
- Real market conditions
- Real odds movements
- No risk
- Tests full system integration

**Cons:**
- Still no real execution
- Can't validate blockchain integration
- May miss execution issues (gas, liquidity)

### Best Practices

1. **Run Full Duration**: Don't stop early even if looks good
2. **Monitor Daily**: Check for anomalies or bugs
3. **Compare to Baseline**: Is it actually better than v1.0.0?
4. **Check Edge Quality**: Are realized edges matching predictions?

## Stage 3: Shadow Trading (Parallel with Production)

**Purpose**: Run side-by-side with production to directly compare
**Risk**: None (logs only, no execution)
**Duration**: 1-2 weeks
**Data**: Same as production (live markets)

### How It Works

Shadow strategy runs in parallel with production and logs what it WOULD do:

```python
from shadow_trading_system import ShadowTradingSystem

# Run v1.1.0 in shadow mode vs production v1.0.0
shadow = ShadowTradingSystem(
    shadow_strategy_version="1.1.0",
    production_strategy_version="1.0.0"
)

await shadow.start()

# Generates comparison report showing divergences
print(shadow.generate_comparison_report())
```

### What It Logs

For every opportunity:
- Did shadow strategy see it? (Yes/No)
- Did production strategy see it? (Yes/No)
- Did shadow take it? (Yes/No + Reason)
- Did production take it? (Yes/No + Reason)
- Why did they differ? (Time horizon, edge threshold, etc.)

### Promotion Criteria

- ✅ Minimum 30 trades
- ✅ Must beat baseline (ROI > production)
- ✅ Max drawdown ≤ 20%
- ✅ Minimum 7 days duration

### Pros & Cons

**Pros:**
- Direct apples-to-apples comparison
- Same market conditions as production
- Identifies exact divergences
- Zero risk

**Cons:**
- Still not testing execution
- Can't validate real blockchain interactions

### Best Practices

1. **Analyze Divergences**: Why did strategies differ?
2. **Look for Patterns**: Is shadow consistently better/worse?
3. **Check Timing**: Do both see opportunities at same time?
4. **Validate Assumptions**: Are improvements real or luck?

## Stage 4: Micro A/B Test (1-5% Real Money)

**Purpose**: Validate execution works with minimal risk
**Risk**: Low (1-5% of capital)
**Duration**: 3-7 days
**Data**: Live with real execution

### How It Works

Very small allocation tests the full execution path:

```python
from ab_testing_framework import ABTestManager

test = test_manager.create_test(
    name="v1.1.0 Micro Test",
    control_version="1.0.0",
    variant_version="1.1.0",
    variant_allocation=5.0,  # Only 5% to variant
    duration_days=3
)
```

### What This Validates

- ✅ Blockchain transactions work
- ✅ Gas estimation correct
- ✅ Liquidity handling correct
- ✅ No execution bugs

### Promotion Criteria

- ✅ Minimum 20 trades
- ✅ Max drawdown ≤ 15% (tighter for real money)
- ✅ Minimum 3 days duration
- ✅ No execution errors

### Pros & Cons

**Pros:**
- Tests real execution
- Validates blockchain integration
- Minimal risk (5% max)

**Cons:**
- Small sample size
- Not statistically significant
- May not find rare edge cases

### Best Practices

1. **Monitor Closely**: Watch every trade
2. **Check Execution**: Gas, slippage, fees all correct?
3. **Look for Bugs**: Any errors or unexpected behavior?
4. **Don't Judge Performance**: Too small sample, just validate it works

## Stage 5: Full A/B Test (20% Real Money)

**Purpose**: Statistically validate performance improvement
**Risk**: Medium (20% of capital)
**Duration**: 14+ days
**Data**: Live with real execution

### How It Works

Standard A/B test with enough allocation for statistical significance:

```python
test = test_manager.create_test(
    name="v1.1.0 Full Test",
    control_version="1.0.0",
    variant_version="1.1.0",
    variant_allocation=20.0,  # 20% to variant
    duration_days=14
)

# After 14 days
analysis = test_manager.complete_test(test.test_id)
print(f"Winner: {analysis['winner']}")
print(f"P-value: {analysis['p_value']:.4f}")
print(f"Recommendation: {analysis['recommendation']}")
```

### Promotion Criteria

- ✅ Minimum 100 trades
- ✅ Must beat baseline by 5%+ (statistically significant)
- ✅ P-value < 0.05 (95% confidence)
- ✅ Max drawdown ≤ 20%
- ✅ Minimum 14 days duration

### Statistical Testing

Uses t-tests to determine if variant is truly better:
- **p-value < 0.05**: Variant is statistically different
- **Effect size > 0.2**: Difference is meaningful
- **Confidence**: 1 - p_value (e.g., 0.95 = 95% confident)

### Pros & Cons

**Pros:**
- Statistical validation
- Real performance data
- Sufficient sample size
- Tests all edge cases

**Cons:**
- 20% of capital at risk
- Takes 2+ weeks
- May still miss rare events

### Best Practices

1. **Don't Stop Early**: Even if winning/losing, complete full duration
2. **Check Auto-Termination**: System stops if variant loses >10%
3. **Review Stats**: Understand p-value and effect size
4. **Compare Metrics**: Not just P&L - check Sharpe, drawdown, win rate

## Stage 6: Gradual Rollout (20% → 100%)

**Purpose**: Gradually increase allocation while monitoring
**Risk**: Increasing (20% → 50% → 80% → 100%)
**Duration**: 1 week per stage
**Data**: Live production

### How It Works

If variant wins A/B test, gradually increase allocation:

```python
# Week 1: 20% (from A/B test)
# Week 2: Increase to 50%
# Week 3: Increase to 80%
# Week 4: Increase to 100% (full production)

# At each stage, monitor for issues
if performance_degraded():
    rollback_to_previous_allocation()
```

### Promotion Criteria (Each Stage)

- ✅ Minimum 50 trades at this allocation
- ✅ Still beats baseline
- ✅ Max drawdown ≤ 20%
- ✅ Minimum 7 days at this allocation

### Why Gradual?

Even if A/B test succeeded, gradual rollout catches:
- Scale-dependent issues
- Portfolio construction problems
- Rare edge cases that only appear at higher allocation

### Pros & Cons

**Pros:**
- Catches scale-dependent issues
- Can rollback quickly if problems
- Builds confidence incrementally

**Cons:**
- Takes 3-4 weeks total
- May seem slow if variant clearly better

### Best Practices

1. **Monitor Closely**: Watch for degradation at each step
2. **Don't Rush**: Take full week at each allocation
3. **Have Rollback Plan**: Know how to revert quickly
4. **Check All Metrics**: Not just P&L

## Complete Example: Testing v1.1.0

Let's walk through testing the v1.1.0 strategy (48h horizon, 50 markets):

### Step 1: Create Strategy Version

```python
from strategy_versions import StrategyRegistry, create_enhanced_version

registry = StrategyRegistry()
enhanced = create_enhanced_version()  # v1.1.0
registry.register_version(enhanced)
```

### Step 2: Create Testing Pipeline

```python
from strategy_testing_pipeline import PipelineManager

manager = PipelineManager()
pipeline = manager.create_pipeline(
    strategy_version="1.1.0",
    baseline_version="1.0.0"
)
```

### Step 3: Run Backtest

```python
from strategy_backtester import run_strategy_backtest
from datetime import datetime, timedelta, timezone

end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=30)

results = run_strategy_backtest("1.1.0", start_date, end_date)

# Check results
passed, reasons = pipeline.complete_stage(results)
if passed:
    pipeline.advance_to_next_stage()
```

**Expected Output:**
```
Backtest Results:
  Total Trades: 150
  Win Rate: 52%
  ROI: 12%
  Max Drawdown: 18%

Promotion Decision: PASS
  - All criteria met

Advanced to: paper_trading
```

### Step 4: Run Paper Trading

```python
from enhanced_trading_engine import EnhancedTradingEngine

engine = EnhancedTradingEngine(
    strategy_version="1.1.0",
    enable_ab_testing=False
)

# Run for 1-2 weeks
await engine.run_trading_loop()

# After 2 weeks, check results
paper_results = {
    'total_trades': 75,
    'win_rate': 0.49,
    'roi': 0.08,
    'max_drawdown': 0.15,
    'duration_days': 14
}

passed, reasons = pipeline.complete_stage(paper_results)
if passed:
    pipeline.advance_to_next_stage()
```

### Step 5: Run Shadow Trading

```python
from shadow_trading_system import ShadowTradingSystem

shadow = ShadowTradingSystem(
    shadow_strategy_version="1.1.0",
    production_strategy_version="1.0.0"
)

# Run for 1 week
await shadow.start()

# Check comparison
report = shadow.generate_comparison_report()
print(report)
```

**Example Report:**
```
SHADOW TRADING COMPARISON REPORT
================================

Shadow Strategy: 1.1.0 - Enhanced 48h
Production Strategy: 1.0.0 - Production Baseline

Opportunity Comparison:
  Shadow Opportunities: 45
  Production Opportunities: 20
  Overlapping: 18
  Shadow Only: 27 (from 48h horizon)
  Production Only: 2

Divergence Rate: 44% (expected due to horizon difference)

Strategy Differences:
  Time Horizon: 24h → 48h
  Market Limit: 20 → 50
```

If shadow consistently beats production, advance to micro test.

### Step 6: Run Micro A/B Test

```python
from ab_testing_framework import ABTestManager

test_manager = ABTestManager()

micro_test = test_manager.create_test(
    name="v1.1.0 Micro Test",
    control_version="1.0.0",
    variant_version="1.1.0",
    variant_allocation=5.0,  # 5% only
    duration_days=3
)

# Run for 3 days with integrated trading system
# (System automatically allocates 5% of trades to variant)

# After 3 days
status = test_manager.monitor_test(micro_test.test_id)

if status['variant']['trades'] >= 20 and no_execution_errors:
    # Advance to full A/B test
    pipeline.advance_to_next_stage()
```

### Step 7: Run Full A/B Test

```python
full_test = test_manager.create_test(
    name="v1.1.0 Full Test",
    control_version="1.0.0",
    variant_version="1.1.0",
    variant_allocation=20.0,  # 20%
    duration_days=14
)

# Run for 14 days
# System automatically allocates trades

# After 14 days, analyze
analysis = test_manager.complete_test(full_test.test_id)

print(f"Result: {analysis['result']}")
print(f"Winner: {analysis['winner']}")
print(f"P-value: {analysis['p_value']:.4f}")
print(f"Improvement: {analysis['metrics_comparison']['total_pnl']['diff_pct']:.1f}%")
print(f"Recommendation: {analysis['recommendation']}")
```

**Example Output:**
```
Result: variant_wins
Winner: 1.1.0
P-value: 0.0234
Improvement: 8.5%
Recommendation: Recommend promoting variant 1.1.0.
Statistically significant improvement with medium effect size (0.42).
```

### Step 8: Gradual Rollout

```python
# If A/B test passes, promote to production with gradual rollout

# Week 1: Already at 20% from A/B test
# Monitor for issues

# Week 2: Increase to 50%
test_manager.update_test_allocation(full_test.test_id, variant_pct=50.0)

# Week 3: Increase to 80%
test_manager.update_test_allocation(full_test.test_id, variant_pct=80.0)

# Week 4: Full production (100%)
registry.promote_version("1.1.0")
```

## Safety Features

### Auto-Termination

Tests automatically stop if variant performs poorly:

```python
# Checks on every trade:
if variant_drawdown > 10%:
    terminate_test("Variant exceeded max drawdown")

if variant_significantly_worse_than_control:
    terminate_test(f"Variant significantly underperforming (p={p_value:.4f})")
```

### Rollback

At any stage, can rollback to previous version:

```python
# Via StrategyRegistry
registry.rollback_to_version("1.0.0")

# Via git
git checkout v1.0.0
```

### Monitoring Alerts

Edge quality monitor alerts if models are miscalibrated:

```python
from monitoring.edge_quality_monitor import EdgeQualityMonitor

monitor = EdgeQualityMonitor()

# Automatic alerts on:
# - Win rate error > 5%
# - High-edge bets losing money
# - Probability miscalibration > 10%
# - Sport-specific issues
```

## Decision Matrix: Should I Advance?

Use this decision tree at each stage:

```
Did strategy pass promotion criteria?
├─ No → STOP, fix issues or abandon
└─ Yes
   ├─ Is improvement meaningful (>5%)?
   │  ├─ No → Consider abandoning (not worth risk)
   │  └─ Yes
   │     ├─ Are risks acceptable?
   │     │  ├─ No → Adjust parameters or abandon
   │     │  └─ Yes
   │     │     ├─ Is confidence high (p<0.05)?
   │     │     │  ├─ No → Continue testing longer
   │     │     │  └─ Yes → ADVANCE to next stage
```

## Common Pitfalls

### 1. Stopping Tests Early

**Problem**: Test looks good after 3 days, why wait 14?
**Solution**: Run full duration. Short-term luck != long-term edge.

### 2. Ignoring Risk Metrics

**Problem**: Strategy has 15% ROI but 30% drawdown
**Solution**: Risk-adjusted returns matter more. Use Sharpe ratio.

### 3. Overfitting Backtest

**Problem**: 50% ROI in backtest, 5% in paper trading
**Solution**: Backtest is just screening. Real validation is live data.

### 4. Not Comparing to Baseline

**Problem**: New strategy has 10% ROI. Is that good?
**Solution**: Always compare to v1.0.0 baseline (95% ROI). Is it better?

### 5. Skipping Shadow Trading

**Problem**: Jump from paper trading to real money
**Solution**: Shadow trading catches divergences and validates assumptions.

## Best Practices Summary

1. **Never Skip Stages**: Each stage catches different issues
2. **Run Full Duration**: Don't stop early even if looks good/bad
3. **Compare to Baseline**: Always benchmark against v1.0.0
4. **Monitor Risk Metrics**: Drawdown matters more than returns
5. **Use Statistical Tests**: Need 95% confidence (p<0.05) to promote
6. **Document Everything**: Track why decisions were made
7. **Have Rollback Plan**: Know how to revert quickly if issues
8. **Start Conservative**: Better to be slow and safe than fast and broke

## Files Reference

- `strategy_testing_pipeline.py`: Multi-stage pipeline manager
- `strategy_backtester.py`: Backtest against historical data
- `shadow_trading_system.py`: Run parallel with production
- `ab_testing_framework.py`: Statistical A/B testing
- `enhanced_trading_engine.py`: Version-aware trading engine
- `strategy_versions.py`: Strategy version control
- `monitoring/edge_quality_monitor.py`: Validate probability models

## Next Steps

1. Review your v1.1.0 and v2.0.0 strategy definitions
2. Run v1.1.0 through full pipeline
3. Document results at each stage
4. Make data-driven promotion decisions
5. Only advance to real money after passing all validation

Remember: **The goal isn't to deploy fast, it's to deploy confidently.** Taking 4-6 weeks to validate a new strategy is much better than losing money on a bad one.
