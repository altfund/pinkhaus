# Blockchain Data Collection System - Complete Guide

## Overview

This guide explains the complete blockchain-based data collection system that replaces the need for external sports betting APIs. The system reads directly from Optimism and Arbitrum blockchains to get real-time sports market data.

## Architecture

```
Blockchain (Optimism/Arbitrum)
    ↓
MarketCreated Events & Contract Calls
    ↓
Blockchain Reader + RPC Manager
    ↓
Enhanced Tag Mappings + Team Metadata
    ↓
Market Enrichment Service
    ↓
Local Database (sport_odds.db)
    ↓
Trading System
```

## Components

### 1. Enhanced Tag Mappings (`enhanced_tag_mappings.py`)

Comprehensive mappings for all sports and leagues:

```python
from enhanced_tag_mappings import tag_mapper

# Decode tags from blockchain
tags = [5, 501]  # [sport_id, league_id]
sport, league = tag_mapper.decode_tags(tags)
# Returns: ("Soccer", "English Premier League")

# Get sport metadata
sport_meta = tag_mapper.get_sport_metadata(5)
# Returns: SportMetadata with positions, scoring type, etc.
```

**Sport IDs:**
- 1: American Football
- 2: Basketball
- 3: Baseball
- 4: Hockey
- 5: Soccer
- 6: MMA
- 7: Boxing
- 8: Tennis
- 9: Golf
- 10: Cricket

**League IDs (examples):**
- 101: NFL
- 201: NBA
- 301: MLB
- 401: NHL
- 501: English Premier League
- 502: La Liga
- 503: Serie A
- 601: UFC

### 2. Team Metadata Service (`team_metadata_service.py`)

Enriches team information with full names, venues, colors, etc:

```python
from team_metadata_service import TeamMetadataService

service = TeamMetadataService()

# Find team by name
team = service.find_team("Liverpool", sport="Soccer", league="EPL")
# Returns: TeamMetadata with full info

# Enrich market data
enriched = service.enrich_market_data({
    'game_label': 'Liverpool vs Manchester City',
    'sport': 'Soccer',
    'league': 'EPL'
})
```

### 3. Blockchain Sync Daemon (`blockchain_sync_daemon.py`)

Continuously syncs blockchain data to local database:

```python
# Run as a service
python blockchain_sync_daemon.py

# Or programmatically
from blockchain_sync_daemon import BlockchainSyncDaemon

daemon = BlockchainSyncDaemon(
    networks=['optimism', 'arbitrum'],
    sync_interval=60,  # Check for new markets every 60s
    odds_update_interval=300  # Update odds every 5 minutes
)

await daemon.run()
```

### 4. Market Enrichment Service (`market_enrichment.py`)

Combines blockchain data with metadata:

```python
from market_enrichment import MarketEnrichmentService

service = MarketEnrichmentService()

# Enrich a single market
enriched = service.enrich_market(
    market_address="0x...",
    network="optimism"
)

# Get active markets
markets = service.get_active_markets(
    sport="Soccer",
    league="EPL",
    limit=50
)

# Search markets
results = service.search_markets("Liverpool")
```

### 5. Database Migrator (`blockchain_to_db_migrator.py`)

Migrates blockchain data to existing database schema:

```bash
# Migrate markets from Optimism
python blockchain_to_db_migrator.py --network optimism

# Sync recent odds updates
python blockchain_to_db_migrator.py --sync-odds --network optimism

# Check migration status
python blockchain_to_db_migrator.py --status
```

## Setup Instructions

### 1. Initialize Metadata

```bash
# Run test to create default team metadata
python team_metadata_service.py

# Run test to verify tag mappings
python enhanced_tag_mappings.py
```

### 2. Start Blockchain Sync

```bash
# Set environment variables (optional)
export BLOCKCHAIN_NETWORKS="optimism,arbitrum"
export SYNC_INTERVAL=60
export ODDS_UPDATE_INTERVAL=300

# Run the sync daemon
python blockchain_sync_daemon.py
```

### 3. Migrate Existing Data

```bash
# Check current status
python blockchain_to_db_migrator.py --status

# Migrate from blockchain
python blockchain_to_db_migrator.py --network optimism --batch-size 100
python blockchain_to_db_migrator.py --network arbitrum --batch-size 100
```

### 4. Set Up as System Service

Create `/etc/systemd/system/ominari-blockchain-sync.service`:

```ini
[Unit]
Description=Ominari Blockchain Sync Daemon
After=network.target

[Service]
Type=simple
User=ominari
WorkingDirectory=/path/to/ominari
ExecStart=/usr/bin/python3 /path/to/ominari/blockchain_sync_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ominari-blockchain-sync
sudo systemctl start ominari-blockchain-sync
```

## Integration with Trading System

### Update Signal Providers

```python
# In your signal provider
from market_enrichment import MarketEnrichmentService

class EnhancedSignal(SignalProvider):
    def __init__(self):
        self.enrichment_service = MarketEnrichmentService()
    
    def get_probs(self, df):
        # Use enriched data
        for market_id in df['market_id']:
            enriched = self.enrichment_service.enrich_market(market_id)
            # Use team metadata, venue info, etc.
```

### Update Data Sources

In your configuration:

```python
DATA_SOURCES = {
    'primary': 'blockchain',
    'fallback': 'api'  # Keep API as fallback if needed
}
```

## Data Available from Blockchain

### Market Creation Event
- `market_address`: Unique market identifier
- `game_id`: Game identifier (bytes32)
- `game_label`: Human-readable game description (e.g., "Team A vs Team B")
- `maturity_date`: When the game occurs (Unix timestamp)
- `tags`: Array with [sport_id, league_id, ...]
- `normalized_odds`: Initial odds from market maker

### Real-Time Data (Contract Calls)
- Current buy/sell odds for each position
- Market liquidity
- Trading volume
- Market resolution status

### What's NOT on Blockchain
- Historical odds movements (expensive to reconstruct)
- Detailed team statistics
- Player information
- Weather data

## Advantages Over API

1. **No Rate Limits**: Read as much as you want
2. **Real-Time**: WebSocket events for instant updates
3. **Decentralized**: No single point of failure
4. **Free**: Use free RPC endpoints
5. **Trustless**: Data directly from smart contracts

## Running Your Own Node (Optional)

For production reliability:

```bash
# Optimism node
docker run -d \
  --name optimism-node \
  -v /data/optimism:/data \
  -p 8545:8545 \
  ethereumoptimism/l2geth

# Arbitrum node
docker run -d \
  --name arbitrum-node \
  -v /data/arbitrum:/data \
  -p 8546:8545 \
  offchainlabs/nitro-node
```

## Monitoring

Check sync status:
```python
from blockchain_sync_daemon import BlockchainSyncDaemon

daemon = BlockchainSyncDaemon()
stats = daemon.get_stats()
print(f"Active markets: {stats['active_markets']}")
print(f"Optimism connected: {stats['readers']['optimism']}")
```

## Troubleshooting

### RPC Connection Issues
```python
# Test RPC connection
from blockchain_reader import BlockchainReader

reader = BlockchainReader('optimism')
print(f"Connected: {reader.check_connection()}")
print(f"Block: {reader.w3.eth.block_number}")
```

### Missing Team Metadata
```python
# Add custom team
from team_metadata_service import TeamMetadataService, TeamMetadata

service = TeamMetadataService()
service.add_team(TeamMetadata(
    team_id="custom_team",
    sport="Soccer",
    league="Custom League",
    full_name="Custom Team FC",
    short_name="Custom",
    abbreviation="CTM",
    aliases=["Custom FC"]
))
```

### Tag Mapping Issues
```python
# Check unknown tags
tags = [99, 999]  # Unknown tags
sport, league = tag_mapper.decode_tags(tags)
print(f"Sport: {sport}, League: {league}")
# Will return: "Sport_99", "League_999"
```

## Future Enhancements

1. **GraphQL Integration**: Use The Graph for efficient historical queries
2. **IPFS Metadata**: Store team logos and additional data on IPFS
3. **Multi-Chain Support**: Add Polygon, Base, etc.
4. **Oracle Aggregation**: Combine multiple oracle sources
5. **ML-Based Enrichment**: Predict missing metadata

## Conclusion

This blockchain-based system provides all the data needed for sports betting without relying on external APIs. It's more reliable, cost-effective, and gives you complete control over your data pipeline.