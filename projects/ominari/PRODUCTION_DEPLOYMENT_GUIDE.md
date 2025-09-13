# Production Deployment Guide - Ominari Trading System

## Overview

This guide covers the complete production deployment of the Ominari trading system with comprehensive risk management, monitoring, and safety controls.

## System Components

### 1. Risk Management (`production_risk_config.py`, `risk_manager.py`)
- Configurable risk levels (Conservative, Moderate, Aggressive)
- Position and portfolio limits
- Time-based controls
- Kill switch for emergency stops
- Real-time monitoring and alerts

### 2. Signal Registry (Dynamic Signal Management)
- Hot-swappable signals
- Adaptive weight allocation (5 methods)
- Performance tracking
- Automatic optimization

### 3. Production Deployment (`production_deploy.py`)
- Integrated system orchestration
- Health checks
- Graceful shutdown
- Configuration management

### 4. Monitoring (`production_monitor.py`)
- Real-time dashboard
- System metrics
- Risk status
- Performance tracking

## Deployment Steps

### 1. Prerequisites

```bash
# Ensure all dependencies are installed
uv sync

# Create required directories
mkdir -p risk_reports sessions logs

# Ensure database is migrated
alembic upgrade head
```

### 2. Configure Risk Limits

#### Option A: Use Preset Configuration
```bash
# Conservative trading
python production_deploy.py --risk-preset conservative --bankroll 1000

# Moderate trading (default)
python production_deploy.py --risk-preset moderate --bankroll 5000

# Aggressive trading
python production_deploy.py --risk-preset aggressive --bankroll 10000
```

#### Option B: Custom Configuration
```python
# Create custom configuration
from production_risk_config import ProductionRiskConfig, PositionLimits

config = ProductionRiskConfig(
    position_limits=PositionLimits(
        max_single_bet_pct=0.02,    # 2% max per bet
        min_edge=0.015,             # 1.5% minimum edge
        max_odds=8.0                # Avoid extreme longshots
    ),
    kelly_limits=KellyLimits(
        kelly_fraction=0.3,         # 30% Kelly
        kelly_cap_per_game=0.15     # 15% max per game
    )
)

# Save configuration
save_config(config, "custom_risk_config.json")
```

### 3. Deploy Production System

#### Basic Deployment
```bash
# Start with moderate risk
python production_deploy.py \
    --risk-preset moderate \
    --bankroll 5000 \
    --session-name "prod_main"
```

#### Advanced Deployment with Overrides
```bash
python production_deploy.py \
    --risk-preset moderate \
    --bankroll 10000 \
    --kelly-fraction 0.4 \
    --max-exposure 0.3 \
    --max-drawdown 0.2 \
    --session-name "prod_advanced"
```

### 4. Monitor System

#### Interactive Dashboard (Recommended)
```bash
# Monitor all sessions
python production_monitor.py

# Monitor specific session
python production_monitor.py --session prod_main

# Text mode (for logs/remote)
python production_monitor.py --text --interval 10
```

#### Dashboard Features
- System-wide metrics
- Session performance
- Risk status indicators
- Real-time alerts
- Top performing sessions

### 5. Production Checklist

#### Pre-Launch
- [ ] Database backup completed
- [ ] Risk configuration validated
- [ ] Test run with --dry-run flag
- [ ] Monitoring dashboard accessible
- [ ] Alert webhooks configured
- [ ] Log rotation set up

#### Launch
- [ ] Start with conservative limits
- [ ] Monitor first trading cycle
- [ ] Verify risk checks working
- [ ] Check position execution
- [ ] Review initial reports

#### Post-Launch
- [ ] Monitor drawdown levels
- [ ] Review daily P&L
- [ ] Check signal performance
- [ ] Adjust weights if needed
- [ ] Scale bankroll gradually

## Risk Limit Summary

### Conservative Settings
```
Position Limits:
- Max bet: 1% / $50
- Min edge: 2%
- Max odds: 5.0

Portfolio Limits:
- Max exposure: 10%
- Max daily loss: 2%
- Max drawdown: 8%

Kelly Settings:
- Kelly fraction: 10%
- Signal confidence: 60%+
```

### Moderate Settings
```
Position Limits:
- Max bet: 2% / $100
- Min edge: 1%
- Max odds: 10.0

Portfolio Limits:
- Max exposure: 25%
- Max daily loss: 5%
- Max drawdown: 15%

Kelly Settings:
- Kelly fraction: 25%
- Signal confidence: 55%+
```

### Aggressive Settings
```
Position Limits:
- Max bet: 5% / $500
- Min edge: 0.5%
- Max odds: 20.0

Portfolio Limits:
- Max exposure: 50%
- Max daily loss: 10%
- Max drawdown: 25%

Kelly Settings:
- Kelly fraction: 50%
- Signal confidence: 52%+
```

## Safety Features

### 1. Kill Switch
Automatically stops trading when:
- Daily loss exceeds 8% (configurable)
- System errors exceed threshold
- Manual activation via monitor

### 2. Position Validation
Every bet must pass:
- Size limits (% and absolute)
- Edge requirements
- Time constraints
- Concentration limits
- Correlation checks

### 3. Rate Limiting
- Max 20 bets per hour
- Max 100 bets per day
- Cool-off after 5 consecutive losses
- Session timeout after 6 hours

### 4. Monitoring & Alerts
- Real-time risk metrics
- Drawdown warnings at 10%
- Daily loss alerts at 3%
- Webhook notifications
- Comprehensive logging

## Operational Procedures

### Daily Operations
1. Check morning risk report
2. Review overnight positions
3. Verify system health
4. Monitor throughout day
5. End-of-day reconciliation

### Weekly Tasks
1. Review signal performance
2. Analyze risk metrics
3. Adjust weights if needed
4. Backup database
5. Review logs for issues

### Emergency Procedures

#### Kill Switch Activated
1. System auto-stops trading
2. Review trigger cause
3. Assess portfolio status
4. Fix underlying issue
5. Manual restart required

#### High Drawdown
1. Reduce position sizes
2. Review signal accuracy
3. Check for system issues
4. Consider pause period
5. Adjust risk parameters

## Performance Optimization

### Signal Weight Updates
```python
# Update weights based on recent performance
from update_evaluate_with_registry import update_registry_weights

# Use different methods
update_registry_weights("bayesian")      # Probability-based
update_registry_weights("inverse_variance")  # Volatility-based
update_registry_weights("regime_based")  # Market condition
```

### Adding New Signals
```python
# Add custom signal to registry
from signal_registry import BaseSignalProvider

class MySignal(BaseSignalProvider):
    def get_probs(self, df):
        # Implementation
        return probabilities

registry.register(MySignal())
```

## Troubleshooting

### Common Issues

#### "Portfolio unhealthy" Error
- Check current drawdown
- Review daily losses
- Verify exposure limits
- May need to wait for cooldown

#### No Bets Placed
- Check market availability
- Verify signals working
- Review edge requirements
- Check time constraints

#### Performance Degradation
- Review signal weights
- Check data quality
- Verify API latency
- Consider signal refresh

### Log Files
- `ominari_production.log` - Main system log
- `monitor.log` - Monitoring events
- `risk_reports/` - Daily risk reports
- `sessions/` - Session configurations

## Best Practices

1. **Start Conservative**: Begin with conservative limits and scale up
2. **Monitor Actively**: Keep dashboard open during trading hours
3. **Review Daily**: Check risk reports and P&L daily
4. **Adjust Gradually**: Make small incremental changes
5. **Document Changes**: Log all configuration adjustments
6. **Test First**: Use dry-run mode for configuration changes
7. **Backup Regularly**: Daily database backups recommended

## Scaling Guidelines

### Bankroll Scaling
- Start: $1,000 (conservative)
- After 1 month profitable: $2,500
- After 3 months profitable: $5,000
- After 6 months profitable: $10,000

### Risk Scaling
- Weeks 1-4: Conservative settings
- Weeks 5-12: Moderate settings
- After 3 months: Consider aggressive

### Signal Scaling
- Start with 1-2 signals
- Add signals gradually
- Test each for 2 weeks
- Maximum 5-6 active signals

## Conclusion

The production deployment system provides institutional-grade risk management with:
- Comprehensive safety controls
- Real-time monitoring
- Adaptive optimization
- Complete audit trail

Always prioritize capital preservation over returns. The system is designed to protect against large losses while capturing consistent small edges.