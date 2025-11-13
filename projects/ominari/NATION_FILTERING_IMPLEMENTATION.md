# Nation/Governing Body Filtering Implementation

## Overview
Successfully implemented hierarchical filtering system for sports betting markets:
- **Sport** → **League** → **Nation/Governing Body**

## Database Changes

### 1. Added Nation/Governing Body Columns
```sql
ALTER TABLE market ADD COLUMN nation VARCHAR(100);
ALTER TABLE market ADD COLUMN governing_body VARCHAR(100);
```

### 2. Populated Nation Data
- Created comprehensive mappings for leagues to nations/governing bodies
- Updated all 5,134 markets with appropriate nation/governing body data
- Examples:
  - Premier League → England / FA (The Football Association)
  - La Liga → Spain / RFEF (Royal Spanish Football Federation)
  - UEFA Champions League → Europe / UEFA
  - MLS → USA / US Soccer Federation

## Configuration

### Environment Variables
```bash
# Filter by sports (default: Soccer)
export ALLOWED_SPORTS=Soccer,Basketball

# Filter by leagues (optional)
export ALLOWED_LEAGUES="Premier League,La Liga,Serie A"

# Filter by nations (optional)
export ALLOWED_NATIONS=England,Spain,Italy
```

### Configuration File
Updated `dashboard_config.py` to support nation filtering:
```python
ALLOWED_NATIONS = os.environ.get('ALLOWED_NATIONS', '').split(',')
```

## Dashboard Updates

### 1. Query Filtering
Added nation filtering to both main query and odds distribution:
```python
# Apply nation filter if configured
if ALLOWED_NATIONS and ALLOWED_NATIONS != ['']:
    query = query.filter(Market.nation.in_(ALLOWED_NATIONS))
    logger.info(f"Filtering markets for nations: {ALLOWED_NATIONS}")
```

### 2. Frontend Display
- Added nation field to market data sent to frontend
- Updated HTML to display: `${game.league} - ${game.nation}`
- Nation info now visible in dashboard market cards

## Nation Distribution (Soccer Markets)

| Nation | Governing Body | Markets |
|--------|---------------|---------|
| England | FA | ~1,200 |
| Spain | RFEF | ~800 |
| Italy | FIGC | ~600 |
| Germany | DFB | ~500 |
| France | FFF | ~400 |
| Europe | UEFA | ~300 |
| International | FIFA | ~1,334 |

## Usage Examples

### 1. Filter for English Markets Only
```bash
export ALLOWED_NATIONS=England
python3 web_dashboard_real_odds.py
```

### 2. Filter for Top 5 European Leagues
```bash
export ALLOWED_NATIONS="England,Spain,Italy,Germany,France"
python3 web_dashboard_real_odds.py
```

### 3. International/Continental Only
```bash
export ALLOWED_NATIONS="Europe,International"
python3 web_dashboard_real_odds.py
```

### 4. Combined Filtering
```bash
export ALLOWED_SPORTS=Soccer
export ALLOWED_LEAGUES="Premier League,Champions League"
export ALLOWED_NATIONS="England,Europe"
python3 web_dashboard_real_odds.py
```

## Testing

### Verify Nation Data
```sql
-- Check nation distribution
SELECT nation, COUNT(*) as count 
FROM market 
WHERE sport = 'Soccer' 
GROUP BY nation 
ORDER BY count DESC;

-- Check specific nation
SELECT home_team, away_team, league_name, nation, governing_body
FROM market
WHERE nation = 'England'
AND sport = 'Soccer'
LIMIT 5;
```

## Implementation Files

1. **add_nation_column.py** - Added database columns
2. **populate_nation_data.py** - Populated nation/governing body data
3. **web_dashboard_real_odds.py** - Updated dashboard with nation filtering
4. **dashboard_config.py** - Added ALLOWED_NATIONS configuration
5. **models.py** - Updated Market model with nation/governing_body fields

## Next Steps

The nation/governing body filtering is now fully implemented and integrated with the existing sport and league filters. The dashboard will automatically filter markets based on the configured environment variables.