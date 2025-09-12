# Ominari Soccer Trading Dashboard

## 🌐 Access

### Unified Dashboard
**URL: http://localhost:8888**
- Clean, focused interface for soccer trading
- Essential trading information with system controls
- Real-time updates every 2 seconds
- Mobile-friendly responsive design
- Integrated logs with filtering

## 📊 Dashboard Sections

### 1. Account Balance
- **Total Balance**: Current account value including P&L
- **Today's Change**: Daily profit/loss in $ and %
- **Visual Indicators**: Green for profit, red for loss

### 2. Soccer Markets
- **Live Odds**: Home/Draw/Away with implied probabilities
- **League & Kickoff**: Competition and match time
- **Signal Strength**: Visual indicator when system has edge
- **Active Selections**: Highlighted when position exists

### 3. Open Positions
- **Match Details**: Teams and selection (Home/Draw/Away)
- **Entry Price**: Odds at time of bet
- **Size**: Amount wagered
- **Live P&L**: Current profit/loss

### 4. Recent Activity
- **Transaction Feed**: Last 10 trades/settlements
- **Timestamps**: When each action occurred
- **Amount Flow**: Money in/out of positions

## ⚙️ Configuration

### Sport Filtering
Edit `/home/ess/Documents/apps/ominari/projects/ominari/config.py`:

```python
# Sports filter - focused on soccer
allowed_sports: List[str] = Field(
    default=["Soccer", "Football", "EPL", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "UEFA", "FIFA"],
    env="ALLOWED_SPORTS"
)
sport_filter_enabled: bool = Field(default=True, env="SPORT_FILTER_ENABLED")
```

Or set via environment:
```bash
export TRADING__ALLOWED_SPORTS="Soccer,EPL,La Liga"
export TRADING__SPORT_FILTER_ENABLED=true
```

### Disable Sport Filter (Show All Sports)
```bash
export TRADING__SPORT_FILTER_ENABLED=false
```

## 🎯 Key Features

1. **Minimal Interface**
   - No clutter - only shows what matters for trading
   - Dark theme optimized for long viewing sessions
   - High contrast for important values

2. **Real-time Updates**
   - Automatic refresh every 2 seconds
   - No manual refresh needed
   - Smooth animations for value changes

3. **Soccer Focus**
   - Filters all non-soccer markets at data collection level
   - Supports major leagues: EPL, La Liga, Serie A, Bundesliga, etc.
   - Includes international competitions: UEFA, FIFA

4. **Essential Metrics Only**
   - Account balance and P&L
   - Open positions with live profit/loss
   - Active markets with best odds
   - Recent transaction history

## 📱 Mobile Access

The minimal dashboard is fully responsive and works on mobile devices:
- Access via phone/tablet using local network IP
- Example: `http://192.168.1.100:8891`
- Optimized touch targets and readable fonts

## 🔧 Troubleshooting

### Dashboard Not Loading
```bash
# Check if running
ps aux | grep monitor_unified

# Restart
pkill -f monitor_unified
python monitor_unified.py
```

### No Markets Showing
```bash
# Check database has soccer data
sqlite3 sport_odds.db "SELECT sport, league, COUNT(*) FROM market WHERE sport LIKE '%soccer%' OR league IN ('EPL','La Liga') GROUP BY sport, league;"

# Force data refresh
python free_data_pull.py
```

### Wrong Port
If port 8888 is busy:
```bash
# Use different port
sed -i 's/port=8888/port=8889/' monitor_unified.py
python monitor_unified.py
```

## 🚀 Quick Start

1. **Start the trading system**:
   ```bash
   ./start_ominari.sh
   ```

2. **Launch unified dashboard**:
   ```bash
   python monitor_unified.py
   ```

3. **Access in browser**:
   ```
   http://localhost:8888
   ```

4. **Monitor via CLI** (optional):
   ```bash
   ./ominari_logs.sh --trades
   ```

The minimal dashboard provides everything needed to monitor soccer trading activity without distraction!