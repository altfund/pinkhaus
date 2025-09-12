# Ominari V1 Architecture

## Overview

The Ominari V1 system is a simplified, production-ready architecture that uses:
- **Overtime API** for market data and quotes
- **Blockchain** for trade monitoring only
- **Paper trading** enabled by default
- **GraphQL** disabled (deprecated endpoints)

## Configuration

### Environment Variables (.env)
```bash
# API Keys
ODDS_API_KEY="your-key"
OVERTIME_API_KEY="your-key"

# V1 Settings
FEATURE_GRAPHQL_STREAMING=False
SPORTS_AMM_V2_ADDRESS=0xFb4e4811C7A811E098A556bD79B64c20b479E431

# Update Frequencies (minutes)
API_UPDATE_FREQ=5
BLOCKCHAIN_SCAN_FREQ=5

# Trading Mode
PAPER_TRADING=true
LIVE_TRADING=false
```

### V1 Configuration Module (config_v1.py)
- Locked configuration with type safety
- Validates correct setup on startup
- Provides clear data flow mapping

## Data Flow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Overtime API    │────▶│              │────▶│   Signal    │
│ (Market Data)   │     │   Database   │     │ Generation  │
└─────────────────┘     │   (SQLite)   │     └──────┬──────┘
                        │              │            │
┌─────────────────┐     │              │            ▼
│ Overtime API    │────▶│              │     ┌─────────────┐
│   (Quotes)      │     │              │     │   Paper     │
└─────────────────┘     │              │     │  Trading    │
                        │              │     └─────┬───────┘
┌─────────────────┐     │              │           │
│  Blockchain     │────▶│              │     ┌─────▼─────┐
│ (Trade Monitor) │     │              │     │ Portfolio │
└─────────────────┘     └──────────────┘     │ Tracking  │
                                             └───────────┘
```

## Key Files

### Core V1 Files
- `config_v1.py` - Locked v1 configuration
- `integrations_v1.py` - V1 system integrator
- `run_ominari_v1.py` - V1 system runner
- `start_v1.sh` - Startup script

### Service Files
- `ominari-v1.service` - Systemd service definition
- `ominari_v1.log` - Main system log
- `ominari_v1_error.log` - Error log

## Running the V1 System

### Manual Start
```bash
./start_v1.sh
```

### Service Start
```bash
sudo systemctl start ominari-v1
sudo systemctl enable ominari-v1  # Auto-start on boot
```

### Check Status
```bash
# Check processes
ps aux | grep ominari_v1

# View logs
tail -f ominari_v1.log

# Monitor web interface
http://localhost:8888/
```

## V1 Cycle Operations

Every 5 minutes, the system:

1. **Collects Market Data**
   - Fetches from Overtime API
   - Updates database with latest markets/odds
   - Supports Soccer, Football, and other sports

2. **Monitors Blockchain**
   - Scans for BoughtFromAmm/SoldToAmm events
   - Records on-chain trades
   - Uses correct contract: 0xFb4e4811C7A811E098A556bD79B64c20b479E431

3. **Generates Signals** (if enabled)
   - Runs signal providers
   - Calculates probabilities
   - Identifies betting opportunities

4. **Executes Paper Trades**
   - Simulates order execution
   - Tracks positions and P&L
   - Updates portfolio state

## Database Safety

The system uses `database_v2.py` with:
- Automatic connection pooling
- Retry logic for large database
- ORM-only queries (no raw SQL)
- Safe query patterns with limits

## Monitoring

### Web Dashboard (http://localhost:8888/)
- Live market view
- Active positions
- System health status
- Recent trades

### Log Files
- `ominari_v1.log` - Main operations
- `ominari_v1_error.log` - Errors only
- `v1_startup.log` - Startup sequence

## Troubleshooting

### Common Issues

1. **Web monitor fails to start**
   - Check if port 8888 is available
   - Verify Flask is installed

2. **No blockchain trades**
   - Normal during low activity
   - Check contract address is correct
   - Verify RPC connection

3. **Database timeouts**
   - Always use ORM queries
   - Add .limit() to queries
   - Use safe_query.py tool

### Health Checks

The system performs automatic health checks every 30 seconds:
- API configuration validation
- Blockchain connection status
- Paper trading status
- Web monitor process

## Future Enhancements

While V1 is locked and stable, potential v2 features include:
- GraphQL support (when endpoints return)
- Multiple blockchain networks
- Advanced signal aggregation
- Live trading execution

## Support

For issues or questions:
- Check logs first
- Review this documentation
- Use `/help` command in CLI
- Report issues at https://github.com/anthropics/claude-code/issues