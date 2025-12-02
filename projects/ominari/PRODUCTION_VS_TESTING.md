# Production vs Testing: How It Works

## Overview

The system maintains **strict separation** between what's in production (actually trading) and what's being tested (no effect on production).

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR TRADING MACHINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  PRODUCTION (Actually Trading)                     │     │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │     │
│  │  Version: v1.0.0 (LOCKED)                         │     │
│  │  Status: ACTIVE                                   │     │
│  │  Trades: Real money, 100% of capital             │     │
│  │  Lock File: production_lock.json                 │     │
│  │                                                    │     │
│  │  Process: execute_trading_solution.py             │     │
│  │  PID: 12345                                       │     │
│  │  Uptime: 4 days                                   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  TESTING (Automated, Isolated)                     │     │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │     │
│  │  Orchestrator: automated_strategy_orchestrator.py  │     │
│  │  PID: 12346                                       │     │
│  │                                                    │     │
│  │  Currently Testing:                               │     │
│  │  ┌─────────────────────────────────────────┐     │     │
│  │  │ v1.1.0: Stage 3 (Shadow Trading)       │     │     │
│  │  │ Duration: 8 days                        │     │     │
│  │  │ Status: Beating production by 1.2%     │     │     │
│  │  └─────────────────────────────────────────┘     │     │
│  │  ┌─────────────────────────────────────────┐     │     │
│  │  │ v2.0.0: Stage 1 (Backtest)             │     │     │
│  │  │ Duration: Running...                    │     │     │
│  │  │ Status: In progress                     │     │     │
│  │  └─────────────────────────────────────────┘     │     │
│  │                                                    │     │
│  │  ZERO effect on production trades                │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Production Lock

There is **ONE and ONLY ONE** production version at any time, stored in `production_lock.json`:

```json
{
  "production_version": "1.0.0",
  "locked_at": "2025-12-02T05:00:00Z",
  "locked_by": "orchestrator"
}
```

**Rules:**
- ✅ Only this version actually trades with real money
- ✅ Cannot be changed except through completed testing pipeline
- ✅ Orchestrator enforces this lock
- ✅ Manual override requires explicit command (emergency only)

### 2. Strategy Status

Strategies have different statuses in `strategy_registry.json`:

- **`active`**: Current production version (v1.0.0)
- **`testing`**: Being validated through pipeline (v1.1.0, v2.0.0)
- **`retired`**: Replaced by better version
- **`deprecated`**: Failed testing, abandoned

### 3. Session Isolation

Each strategy version runs in its own session:

```json
{
  "20251202_050000_v1.0.0_production": {
    "strategy_version": "1.0.0",
    "session_type": "production",
    "positions": { ... }
  },
  "20251202_050000_v1.1.0_paper": {
    "strategy_version": "1.1.0",
    "session_type": "paper",
    "positions": { ... }
  },
  "20251202_050000_v1.1.0_shadow": {
    "strategy_version": "1.1.0",
    "session_type": "shadow",
    "positions": { ... }
  }
}
```

**Separation:**
- Production session affects real capital
- Paper/shadow sessions are simulated only
- Never mixed or confused

## How Testing Works

### Stage 1-3: Zero Risk (Automated)

These stages run **completely isolated** from production:

```
STAGE 1: BACKTEST
├─ Runs: Historical data only
├─ Automated: Yes
├─ Duration: Hours
├─ Effect on production: ZERO
└─ Advances: Automatically if passes

STAGE 2: PAPER TRADING
├─ Runs: Live data, no execution
├─ Automated: Yes
├─ Duration: 7-14 days
├─ Effect on production: ZERO
├─ Logs: Would-be trades to separate session
└─ Advances: Automatically if passes

STAGE 3: SHADOW TRADING
├─ Runs: Parallel with production
├─ Automated: Yes
├─ Duration: 7-14 days
├─ Effect on production: ZERO
├─ Logs: Divergences from production
└─ Advances: Automatically if passes
```

**Key Point**: Stages 1-3 are **completely safe**. They can't affect production.

### Stage 4-5: Low-Medium Risk (Automated A/B)

These stages actually trade, but with **limited allocation**:

```
STAGE 4: MICRO A/B TEST
├─ Allocation: 1-5% of capital
├─ Duration: 3-7 days
├─ Production: 95-99%
├─ Variant: 1-5%
├─ Effect: Minimal (< $500 risk with $10k bankroll)
├─ Purpose: Validate execution works
└─ Advances: Automatically if passes

STAGE 5: FULL A/B TEST
├─ Allocation: 20% of capital
├─ Duration: 14+ days
├─ Production: 80%
├─ Variant: 20%
├─ Effect: Limited ($2k risk with $10k bankroll)
├─ Purpose: Statistical validation
├─ Promotion: Automatic if statistically better
└─ Rollback: Automatic if underperforms
```

**A/B Test Allocation:**

For every trade opportunity:
```python
# Orchestrator decides allocation
if random.random() < 0.20:  # 20% to variant
    use_strategy_version = "1.1.0"  # Testing
else:
    use_strategy_version = "1.0.0"  # Production
```

This is **automatic and transparent**. Production still gets 80% of trades.

### Stage 6: Gradual Rollout (Automated)

If variant wins A/B test, automatically increase allocation:

```
Week 1: 20% variant, 80% production
Week 2: 50% variant, 50% production  (if performing well)
Week 3: 80% variant, 20% production  (if still performing well)
Week 4: 100% variant (promoted to production)
```

**Automatic Rollback**: If performance degrades at any stage, automatically rollback to previous allocation.

## Orchestrator Operation

### What It Does Automatically

The orchestrator runs in the background and:

1. **Detects new strategies** marked "testing" in registry
2. **Runs backtest** automatically (Stage 1)
3. **Starts paper trading** if backtest passes (Stage 2)
4. **Runs shadow trading** alongside production (Stage 3)
5. **Creates A/B tests** when shadow trading passes (Stage 4-5)
6. **Monitors performance** of all stages
7. **Advances stages** when criteria met
8. **Promotes winners** to production automatically
9. **Sends notifications** via Discord (no action required)
10. **Rollback on issues** automatically

### What It Doesn't Do

The orchestrator **does not**:
- Decide what strategies to test (you add them to registry)
- Override safety limits
- Skip stages
- Deploy untested code to production
- Require constant manual intervention

## Command Reference

### Check Status

```bash
# Check what's production
$ cat production_lock.json
{
  "production_version": "1.0.0",
  "locked_at": "2025-12-02T05:00:00Z"
}

# Check what's being tested
$ python -c "from automated_strategy_orchestrator import AutomatedStrategyOrchestrator; \
  o = AutomatedStrategyOrchestrator(); \
  import json; \
  print(json.dumps(o.get_status(), indent=2))"
```

### Add New Strategy to Test

```python
from strategy_versions import StrategyRegistry, StrategyVersion

registry = StrategyRegistry()

# Create new strategy version
new_strategy = StrategyVersion(
    version="1.2.0",
    name="Enhanced Multi-Sport",
    description="Testing multi-sport with 72h horizon",
    sports=["Soccer", "Tennis", "Basketball"],
    time_horizon_hours=72,
    market_query_limit=100,
    min_conservative_edge=2.0,
    kelly_fraction=0.25,
    max_position_pct=0.02,
    max_portfolio_pct=0.20,
    max_open_positions=50,
    status="testing"  # This triggers orchestrator to test it
)

registry.register_version(new_strategy)
```

Orchestrator will automatically detect it on next cycle (within 5 minutes) and start testing.

### Start/Stop Orchestrator

```bash
# Start orchestrator (systemd)
$ sudo systemctl start ominari-orchestrator

# Check status
$ sudo systemctl status ominari-orchestrator

# View logs
$ tail -f logs/orchestrator.log

# Stop orchestrator
$ sudo systemctl stop ominari-orchestrator
```

### Manual Promotion (Emergency Only)

```python
from automated_strategy_orchestrator import AutomatedStrategyOrchestrator

orchestrator = AutomatedStrategyOrchestrator()

# Emergency: Manually promote a version
await orchestrator._promote_to_production("1.1.0")
```

**Warning**: Only use in emergencies. Let the orchestrator handle promotion automatically.

### Rollback Production

```bash
# Emergency rollback to v1.0.0
$ python -c "from strategy_versions import StrategyRegistry; \
  r = StrategyRegistry(); \
  r.rollback_to_version('1.0.0')"

# Update production lock
$ python -c "from automated_strategy_orchestrator import *; \
  import json; \
  lock = ProductionLock(production_version='1.0.0', locked_at=datetime.now(timezone.utc).isoformat()); \
  with open('production_lock.json', 'w') as f: json.dump(lock.to_dict(), f, indent=2)"

# Restart production system
$ sudo systemctl restart ominari-trading
```

## Monitoring

### Discord Notifications

Orchestrator sends automatic notifications for:

- ✅ New strategy detected and testing started
- ✅ Backtest completed (pass/fail)
- ✅ Paper trading started
- ✅ Shadow trading results
- ✅ A/B test started
- ✅ A/B test completed with recommendation
- ✅ Promotion to production
- ⚠️ Issues detected (auto-rollback)
- 🚨 Critical errors

**No action required** - these are informational.

### Log Files

```bash
# Orchestrator logs
$ tail -f logs/orchestrator.log

# Production trading logs
$ tail -f logs/main_heartbeat.log

# Paper trading logs
$ tail -f logs/paper_trading.log

# Shadow trading logs
$ tail -f shadow_trading_logs/
```

### Dashboard (Future)

A web dashboard will show:
- Current production version
- Active tests and their progress
- Stage-by-stage results
- Performance comparisons
- Promotion history

## Safety Features

### 1. Auto-Termination

Tests automatically stop if:
- Variant drawdown > 10%
- Variant significantly underperforms (p < 0.05)
- Execution errors detected
- Criteria not met after maximum duration

### 2. Rollback on Degradation

If promoted variant starts underperforming:
- Automatic rollback to previous version
- Discord alert sent
- Manual review triggered

### 3. Production Lock

Production version cannot be changed except through:
- Completed testing pipeline (automated)
- Manual override with explicit command (emergency)

Lock file prevents accidental deployment.

### 4. Stage Gates

Cannot skip stages:
- Must pass Stage 1 to reach Stage 2
- Must pass Stage 2 to reach Stage 3
- etc.

Each gate has clear criteria.

### 5. Resource Isolation

Testing strategies use separate:
- Sessions
- Log files
- Database entries (with strategy version tag)
- Never interfere with production

## Best Practices

### DO:
- ✅ Let orchestrator run automatically
- ✅ Add new strategies to registry as "testing"
- ✅ Monitor Discord notifications
- ✅ Review promotion recommendations
- ✅ Trust the automated pipeline

### DON'T:
- ❌ Manually edit production_lock.json (except emergencies)
- ❌ Stop orchestrator without reason
- ❌ Skip stages manually
- ❌ Deploy untested strategies to production
- ❌ Override safety limits

## Troubleshooting

### "Why isn't my strategy being tested?"

Check:
1. Is it registered in `strategy_registry.json`?
2. Is status set to "testing"?
3. Is orchestrator running? (`sudo systemctl status ominari-orchestrator`)
4. Check logs: `tail -f logs/orchestrator.log`

### "Why is testing taking so long?"

Each stage has minimum durations:
- Backtest: Hours
- Paper Trading: 7-14 days
- Shadow Trading: 7-14 days
- Micro A/B: 3-7 days
- Full A/B: 14+ days
- Gradual Rollout: 3-4 weeks

**Total: 6-8 weeks** for full validation. This is intentional and ensures safety.

### "Can I speed it up?"

You can:
- Lower minimum duration requirements (edit `DEFAULT_CRITERIA` in `strategy_testing_pipeline.py`)
- Skip shadow trading (not recommended)
- Use shorter A/B test (minimum 7 days, not recommended)

But **don't compromise validation** to deploy faster. Better to be slow and confident.

### "How do I know what's production?"

```bash
$ cat production_lock.json
```

This is the single source of truth.

### "Can I run multiple strategies in production?"

No. By design, there is ONE production version at a time. This ensures:
- Clear responsibility
- No confusion about what's live
- Easy rollback if needed
- Simple monitoring

A/B tests allow testing variants, but production is always a single version.

## Example Timeline

Here's what happens when you add v1.1.0 for testing:

```
Day 0: Add v1.1.0 to registry with status="testing"
Day 0: Orchestrator detects new strategy, starts backtest
Day 0: Backtest completes in 2 hours → PASSES
Day 0: Orchestrator starts paper trading automatically

Day 7: Paper trading has 50 trades → meets criteria
Day 7: Orchestrator advances to shadow trading

Day 14: Shadow trading shows 1.2% improvement → PASSES
Day 14: Orchestrator starts micro A/B test (5%)

Day 17: Micro test complete, no errors → PASSES
Day 17: Orchestrator starts full A/B test (20%)

Day 31: Full A/B test complete
Day 31: Statistical analysis: p=0.023, variant wins
Day 31: Orchestrator starts gradual rollout (20%→50%)

Day 38: 50% allocation maintained performance
Day 38: Orchestrator increases to 80%

Day 45: 80% allocation maintained performance
Day 45: Orchestrator increases to 100%

Day 52: v1.1.0 promoted to production
Day 52: v1.0.0 retired
```

**Total time**: 52 days (~7.5 weeks)
**Human intervention required**: Zero (just monitored Discord notifications)
**Risk**: Minimal (progressively validated)
**Confidence**: High (statistical validation with 97.7% confidence)

## Files Reference

- `production_lock.json` - Current production version (single source of truth)
- `strategy_registry.json` - All strategy versions and their status
- `ab_tests.json` - Active A/B tests
- `strategy_testing_pipelines.json` - Pipeline state for each strategy
- `orchestrator_state.json` - Orchestrator internal state
- `systemd/ominari-orchestrator.service` - Systemd service definition

## Summary

**Production**: Locked, single version, actually trading
**Testing**: Automated, isolated, multiple strategies
**Orchestrator**: Runs in background, manages entire pipeline
**Notifications**: Automatic via Discord, no action required
**Promotion**: Automatic when statistically validated
**Rollback**: Automatic if issues detected

You set the strategies to test, the system validates them automatically, and promotes winners. It's designed to be **hands-off but safe**.
