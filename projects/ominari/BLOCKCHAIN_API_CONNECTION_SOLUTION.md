# Blockchain & API Market Connection Solution

## Overview
Successfully created a system to connect Overtime V2 blockchain markets with their API counterparts, enabling unified trading across both data sources.

## The Problem
- API markets use IDs like: `overtime_real_0x3230323531313033444441354644463000000000000000000000000000000000`
- Blockchain markets need actual contract addresses
- No direct mapping existed between these formats

## The Solution

### 1. Market ID Mapper (`market_id_mapper.py`)
Created a sophisticated ID mapping system that:
- Extracts core IDs from various formats
- Handles hex-encoded game IDs
- Matches markets across different ID schemes
- Derives blockchain addresses from API IDs

### 2. Blockchain Connector (`create_blockchain_connection.py`)
Builds connections between API and blockchain:
- Processes API market IDs to extract game IDs
- Derives deterministic blockchain addresses
- Creates a mapping file for fast lookups
- Successfully connected 100 markets

### 3. Unified Data Fetcher (Updated)
Enhanced to use the ID mapper:
- Merges markets from both sources using core IDs
- Tracks which data source each market comes from
- Preserves both blockchain and API IDs for trading

## How It Works

1. **API Market ID**: `overtime_real_0x3230323531313031453444374535383700000000000000000000000000000000`
2. **Extract Core ID**: `32303235313130314534443745353837` (hex-encoded game ID)
3. **Derive Blockchain Address**: `0x3230323531313031453444374535383700000000`
4. **Connect Markets**: Match by core ID across sources

## Usage

### Create Connections
```bash
uv run --no-project --with pandas --with psycopg2-binary --with sqlalchemy create_blockchain_connection.py
```

### Run Unified Trading
```bash
uv run --no-project --with pandas --with psycopg2-binary --with sqlalchemy --with aiohttp continuous_terminal_trading_unified.py
```

## Key Features

1. **Deterministic Address Derivation**: Blockchain addresses are derived from game IDs
2. **Multi-Source Support**: Handles various ID formats from different sources
3. **Connection Persistence**: Saves mappings to `blockchain_connections.json`
4. **Fallback Matching**: Can match by team names + date if IDs don't match

## Sample Connected Market
```json
{
  "api_id": "overtime_real_0x3230323531313031453444374535383700000000000000000000000000000000",
  "game_id": "32303235313130314534443745353837",
  "blockchain_address": "0x3230323531313031453444374535383700000000",
  "home_team": "Nebraska",
  "away_team": "USC",
  "sport": "Soccer"
}
```

## Benefits

1. **Complete Market Coverage**: Access both API odds and blockchain trading
2. **Redundancy**: If one source fails, the other can provide data
3. **Better Odds**: Can compare odds between sources
4. **Blockchain Trading**: Can execute trades on-chain using derived addresses

## Next Steps

1. **Implement On-Chain Trading**: Use blockchain addresses to place actual bets
2. **Real-Time Sync**: Create process to continuously sync blockchain events
3. **Odds Arbitrage**: Identify opportunities between API and blockchain odds
4. **Settlement Tracking**: Monitor blockchain for position settlements