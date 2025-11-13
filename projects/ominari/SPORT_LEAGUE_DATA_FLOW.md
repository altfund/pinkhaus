# Sport and League Data Flow in Ominari

## Overview
This document explains how sport and league data is sourced and updated in the Ominari system.

## Database Schema

### Market Table
- `sport` (String) - The sport category (e.g., "Soccer", "Basketball", "Baseball")
- `league_name` (String) - The league/competition name (e.g., "Premier League", "NBA", "MLB")

### Data Sources

1. **Overtime API** (`https://api.overtime.io/overtime-v2/`)
   - Provides real-time sports markets
   - Returns sport and league information with each market
   - Example: `fetch_overtime_markets.py`

2. **Blockchain Data**
   - Markets from blockchain contracts
   - Sport/league data extracted from market metadata
   - Example: `fetch_blockchain_odds_real.py`

3. **Manual Sync Scripts**
   - `sync_sports_from_api.py` - Maps sports using Overtime API definitions
   - `fix_sport_classification.py` - Corrects sport categorizations
   - `update_sports_with_api.py` - Updates sport data from API

## Data Flow

1. **Initial Fetch**
   ```python
   # From fetch_overtime_markets.py
   sport = market_data.get('sport', 'Soccer')
   if sport.lower() == 'football':
       sport = 'Soccer'  # Normalize football -> Soccer
   
   league_name = market_data.get('league', market_data.get('leagueName', f'{sport} League'))
   ```

2. **Database Insert**
   ```python
   market = Market(
       source_id=market_id,
       sport=sport,
       league_name=league_name,
       # ... other fields
   )
   ```

3. **Dashboard Display**
   ```python
   # From web_dashboard_real_odds.py
   'sport': market.sport,
   'league': market.league_name or 'Unknown',
   ```

## Sport Mapping Rules

### Common Mappings
- "football" → "Soccer" (for international football)
- "NCAAB" → "Basketball" (college basketball)
- "NCAAF" → "Football" (college American football)
- "CFB" → "Football" (college football)

### League Identification
Leagues are typically extracted from:
1. `league` field in API response
2. `leagueName` field as fallback
3. Default to `{sport} League` if not provided

## Sync Process

### Automatic Updates
- Data fetchers run periodically to get new markets
- Sport and league data is included with each market

### Manual Corrections
- Run `sync_sports_from_api.py` to correct mappings
- Use `fix_sport_classification.py` for bulk updates
- Database queries to update misclassified data

## Current Status

### Available Sports (from API)
- Soccer/Football
- Basketball (NBA, NCAAB)
- Baseball (MLB)
- American Football (NFL, NCAAF)
- Hockey (NHL)
- Tennis
- Golf
- MMA/UFC
- Cricket
- Others

### Data Quality
- Sport data: Generally reliable from API sources
- League data: Sometimes missing or needs normalization
- Blockchain data: May need manual mapping

## Maintenance

### Regular Tasks
1. Monitor for new sports/leagues
2. Update mapping rules as needed
3. Run sync scripts after major data imports
4. Check dashboard for "Unknown" leagues

### Common Issues
- Missing league names (defaults to "Unknown")
- Sport name variations (e.g., "football" vs "Soccer")
- Blockchain markets lacking metadata
- API changes requiring mapping updates

## API Endpoints

### Health Check
```bash
curl http://localhost:8888/health
```

### Metrics (includes market counts by sport)
```bash
curl http://localhost:8888/metrics
```

### Dashboard (displays sport/league)
```
http://localhost:8888/
```

## Recommendations

1. **Regular Sync**: Run sync scripts weekly to maintain data quality
2. **Validation**: Check for markets with NULL sport/league
3. **Monitoring**: Set up alerts for new unmapped sports/leagues
4. **Documentation**: Update mapping rules when changes are made

## Scripts for Data Management

```bash
# Check current sport/league data
python check_sport_league_data.py

# Sync sports from API
python sync_sports_from_api.py

# Fix classifications
python fix_sport_classification.py

# Update from API
python update_sports_with_api.py
```