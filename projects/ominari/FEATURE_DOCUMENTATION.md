# 📚 Ominari Trading Platform - Feature Documentation

**Version**: 2.0.0  
**Last Updated**: October 27, 2025

## Table of Contents

1. [Portfolio-Based Trading System](#portfolio-based-trading-system)
2. [Stop Loss System](#stop-loss-system)
3. [Performance Analytics](#performance-analytics)
4. [Market Filtering](#market-filtering)
5. [Market Links Integration](#market-links-integration)
6. [Testing Framework](#testing-framework)

---

## 1. Portfolio-Based Trading System

### Overview
The portfolio trading system uses Kelly criterion optimization to allocate capital across multiple betting opportunities simultaneously, maximizing long-term growth while managing risk.

### Key Features

#### Kelly Criterion Implementation
- **Optimal Sizing**: Calculates ideal bet sizes based on edge and probability
- **Multi-Market Optimization**: Allocates capital across all available opportunities
- **Risk Adjustment**: Applies fractional Kelly (default 25%) for conservative sizing

#### Portfolio Management
```python
# Configuration in STRATEGY_CONFIG
{
    'kelly_fraction': 0.25,      # Use 25% of full Kelly
    'min_edge': 0.02,            # Minimum 2% edge required
    'cap_per_bet': 0.01,         # Max 1% per individual bet
    'cap_per_game': 0.02,        # Max 2% exposure per game
    'min_bet': 10,               # Minimum bet size $10
    'bankroll': 5000             # Starting bankroll
}
```

#### How It Works
1. **Signal Generation**: Edge calculator evaluates all markets
2. **Portfolio Optimization**: Kelly criterion determines optimal allocations
3. **Risk Capping**: Apply per-bet and per-game limits
4. **Execution**: Place trades maintaining portfolio balance

### Usage
The system runs automatically via the portfolio trading engine. Monitor via dashboard:
- Portfolio value displayed in real-time
- Exposure percentage shown with warnings >50%
- Individual position tracking in Positions tab

---

## 2. Stop Loss System

### Overview
Multi-layered protection system that automatically halts trading and closes positions when risk thresholds are breached.

### Trigger Types

#### 1. Drawdown Protection
- **Threshold**: 5% portfolio loss
- **Action**: Immediate stop and position closure
- **Recovery**: 60-minute cooldown period

#### 2. Time-Window Loss
- **Threshold**: 5% loss within 10 minutes
- **Purpose**: Detect rapid deterioration
- **Action**: Emergency stop

#### 3. Consecutive Losses
- **Threshold**: 3 losses in a row
- **Purpose**: Break losing streaks
- **Action**: Pause trading for review

#### 4. Manual Override
- **Access**: Red STOP button on dashboard
- **Action**: Immediate halt and closure
- **Use Case**: User discretion

### Configuration
```python
stop_loss_config = {
    'drawdown_pct': 5,              # 5% max drawdown
    'time_window_minutes': 10,       # Rolling window
    'recovery_time_minutes': 60,     # Cooldown period
    'max_daily_loss_pct': 10,        # Daily limit
    'consecutive_losses': 3,         # Streak limit
    'volatility_threshold': 2.0      # Volatility multiplier
}
```

### Dashboard Integration
- **Status Indicator**: Shows "Protected" when active
- **Stop Button**: Always accessible for manual intervention
- **Activity Feed**: Real-time stop loss notifications
- **Recovery Timer**: Shows time until trading can resume

---

## 3. Performance Analytics

### Overview
Comprehensive analytics system providing real-time and historical performance metrics for informed decision-making.

### Key Metrics

#### Financial Performance
- **ROI**: Return on Investment percentage
- **Profit Factor**: Gross profit / Gross loss ratio
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

#### Trading Statistics
- **Win Rate**: Percentage of winning trades
- **Average Win/Loss**: Mean profit and loss amounts
- **Win/Loss Ratio**: Relative size of wins vs losses
- **Consecutive Wins/Losses**: Longest streaks

#### Dimensional Analysis
1. **By Sport**: Performance breakdown by sport type
2. **By Outcome**: Home/Draw/Away win rates
3. **By Hour**: Time-of-day performance patterns

### Accessing Analytics
1. Navigate to dashboard (http://localhost:8888)
2. Click "Analytics" tab in Positions section
3. View real-time updates as trades settle

### Interpreting Metrics
- **Good ROI**: >10% indicates profitable system
- **Profit Factor**: >1.5 suggests robust edge
- **Sharpe Ratio**: >1.0 indicates good risk-adjusted returns
- **Win Rate**: 45-55% typical for value betting

---

## 4. Market Filtering

### Overview
Advanced filtering system to focus on high-value betting opportunities based on confidence and edge criteria.

### Filter Types

#### Sport Filter
- **Options**: Soccer, Basketball, Football, Baseball, Hockey, Tennis, All
- **Purpose**: Focus on specific sports
- **Location**: Top dropdown in markets section

#### Confidence Filter
- **High (70%+)**: Very confident predictions
- **Medium (50-70%)**: Moderate confidence
- **Low (30-50%)**: Lower confidence
- **Positive Edge Only**: Any positive expected value

#### Edge Filter
- **>5%**: Premium opportunities
- **>3%**: Strong edges
- **>2%**: Standard threshold
- **>1%**: Marginal edges
- **>0%**: All positive EV

### Usage Examples

#### Conservative Approach
1. Set Confidence to "High (70%+)"
2. Set Edge to ">3%"
3. Result: Only highest quality bets shown

#### Volume Approach
1. Set Confidence to "All"
2. Set Edge to ">1%"
3. Result: More opportunities, lower average quality

#### Sport Specialist
1. Set Sport to specific sport
2. Set Confidence to "Medium"
3. Set Edge to ">2%"
4. Result: Focused domain betting

### Filter Persistence
- Filters remain active during session
- Market count updates in real-time
- Affects which markets trading engine considers

---

## 5. Market Links Integration

### Overview
Direct integration with external platforms for market verification and additional information.

### Supported Platforms

#### Overtime Markets
- **Link Format**: `https://overtimemarkets.xyz/markets/optimism/market/{id}`
- **Networks**: Optimism, Arbitrum
- **Access**: Click 🔗 icon next to market name

#### Blockchain Explorers
- **Optimism**: Optimistic Etherscan
- **Arbitrum**: Arbiscan
- **Purpose**: Verify on-chain market data

### Features

#### Source Identification
- **API**: Traditional sportsbook data
- **Overtime**: Decentralized sports markets
- **OP Chain**: Optimism blockchain
- **Arb Chain**: Arbitrum blockchain

#### Quick Access
- Hover over market for source tooltip
- Click 🔗 to open in new tab
- Source column shows data origin

### Use Cases
1. **Verification**: Confirm odds are accurate
2. **Research**: Deep dive into market details
3. **Arbitrage**: Compare with external platforms
4. **Transparency**: Understand data sources

---

## 6. Testing Framework

### Overview
Comprehensive testing system ensuring reliability and catching regressions before deployment.

### Test Levels

#### Quick Deployment Tests
```bash
just test-deployment
```
- **Duration**: <10 seconds
- **Coverage**: Critical paths only
- **Use**: Pre-deployment validation

**Tests**:
1. Database connectivity
2. Dashboard response
3. Paper trading system
4. Stop loss system

#### Comprehensive Test Suite
```bash
just test-comprehensive
```
- **Duration**: ~30 seconds
- **Coverage**: Full system validation
- **Use**: Major releases

**Categories**:
1. Environment Setup
2. Database Connectivity
3. Data Quality
4. Paper Trading System
5. Portfolio Trading Engine
6. Stop Loss System
7. Dashboard Functionality
8. WebSocket Connectivity
9. Edge Calculation
10. Risk Management
11. System Integration

### Test Reports
- **Format**: JSON with detailed results
- **Location**: `test_report_YYYYMMDD_HHMMSS.json`
- **Contents**: Pass/fail status, error messages, timing

### Running Specific Tests
```bash
# Test paper trading only
uv run python -c "from comprehensive_test_suite import TestSuite; TestSuite().test_paper_trading_system()"

# Test stop loss only
uv run python -c "from comprehensive_test_suite import TestSuite; TestSuite().test_stop_loss_system()"
```

### Interpreting Results
- **PASS**: Feature working correctly
- **FAIL**: Issue detected, check error message
- **Deployment Ready**: Core features operational

---

## Quick Reference

### Dashboard URL
```
http://localhost:8888
```

### Key Commands
```bash
# Start dashboard
just run-dashboard

# Run quick tests
just test-deployment

# Run all tests
just test-comprehensive

# Check logs
tail -f web_monitor.log
```

### Environment Variables
```bash
export PG_HOST=localhost
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
export USE_POSTGRESQL=1
```

### Safety Limits
- Max 50% portfolio exposure
- 5% drawdown stop loss
- 1% max per bet
- 2% max per game
- $10 minimum bet size

### Support Files
- `STOP_LOSS_IMPLEMENTATION.md` - Technical details
- `TESTING.md` - Testing guide
- `DEPLOYMENT_READINESS_REPORT.md` - Status report

---

## Troubleshooting

### Dashboard Won't Load
1. Check if process running: `ps aux | grep web_monitor`
2. Check logs: `tail -100 web_monitor.log`
3. Restart: `just run-dashboard`

### Trades Not Executing
1. Verify session active in dashboard
2. Check stop loss status (not triggered)
3. Confirm bankroll available
4. Review edge thresholds

### Performance Issues
1. Check database connection
2. Monitor CPU/memory usage
3. Review log file sizes
4. Restart services if needed

---

*This documentation covers all major features added to the Ominari Trading Platform. For technical implementation details, refer to the source code and inline comments.*