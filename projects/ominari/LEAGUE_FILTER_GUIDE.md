# League Filtering Guide

## Overview
The Ominari dashboard supports filtering by both sport AND league, allowing precise control over which markets are displayed.

## League Data Status

### ✅ Fixed Leagues (4,058 markets updated)
- **Premier League** (28 markets)
- **La Liga** (27 markets)  
- **Serie A** (17 markets)
- **Ligue 1** (23 markets)
- **Primeira Liga** (40 markets)
- **Eredivisie** (18 markets)
- **NBA** (36 markets)
- **NFL** (28 markets)
- **NHL** (46 markets)
- **MLB** (187 markets)
- **Esports leagues** (CCT Europe, ESEA League, etc.)

### 📊 Generic Classifications
- **European Football** (932 markets) - General European leagues
- **International Football** (2,535 markets) - International matches
- **Group Stage**, **Round 1**, **Playoffs** - Tournament phases

## Configuration Options

### 1. Sport Only (Default)
```bash
# Show only Soccer
export ALLOWED_SPORTS=Soccer
python web_dashboard_real_odds.py
```

### 2. Sport + Specific Leagues
```bash
# Show only Premier League and La Liga
export ALLOWED_SPORTS=Soccer
export ALLOWED_LEAGUES="Premier League,La Liga"
python web_dashboard_real_odds.py
```

### 3. Multiple Sports, Specific Leagues
```bash
# NBA and Premier League only
export ALLOWED_SPORTS=Soccer,Basketball
export ALLOWED_LEAGUES="Premier League,NBA"
python web_dashboard_real_odds.py
```

### 4. All Soccer, Top Leagues Only
```bash
export ALLOWED_SPORTS=Soccer
export ALLOWED_LEAGUES="Premier League,La Liga,Serie A,Ligue 1,Bundesliga"
python web_dashboard_real_odds.py
```

## Examples

### European Top 5 Leagues
```bash
export ALLOWED_SPORTS=Soccer
export ALLOWED_LEAGUES="Premier League,La Liga,Serie A,Ligue 1,Bundesliga"
```

### US Sports Only
```bash
export ALLOWED_SPORTS=Basketball,Baseball,American Football,Hockey
export ALLOWED_LEAGUES="NBA,MLB,NFL,NHL"
```

### Esports Tournaments
```bash
export ALLOWED_SPORTS=Esports
export ALLOWED_LEAGUES="CCT Europe,ESEA League,BLAST Premier"
```

### International Soccer
```bash
export ALLOWED_SPORTS=Soccer
export ALLOWED_LEAGUES="International Football,World Cup,European Championship"
```

## Available Leagues by Sport

### Soccer
- Premier League
- La Liga
- Serie A
- Ligue 1
- Bundesliga (limited data)
- Primeira Liga
- Eredivisie
- European Football (generic)
- International Football (generic)

### Basketball
- NBA
- WNBA
- EuroLeague (limited)

### Baseball
- MLB
- KBO (Korean)

### American Football
- NFL

### Hockey
- NHL

### Esports
- CCT Europe
- ESEA League
- ESL Pro League
- BLAST Premier
- VCT Champions
- Mobile Legends Pro League

## Dashboard Display

The dashboard header shows active filters:
- Sport only: "Live Markets - Real Odds (Soccer)"
- Sport + League: "Live Markets - Real Odds (Soccer - Premier League, La Liga)"

## Performance Tips

1. **Specific leagues** are faster than generic ones
2. **Combine sport + league** filters for best performance
3. **Avoid generic leagues** like "Regular Season" or "N/A"

## Docker Usage

```yaml
# In docker-compose.yml
ominari-dashboard:
  environment:
    - ALLOWED_SPORTS=Soccer
    - ALLOWED_LEAGUES=Premier League,La Liga
```

## Troubleshooting

### No Markets Showing
- Check exact league names (case-sensitive)
- Verify league exists: `python check_league_quality.py`
- Try without league filter first

### Finding League Names
```bash
# List all soccer leagues
python show_soccer_leagues.py

# Check all leagues
python check_league_quality.py
```

### Common Issues
- League name must match exactly
- Some leagues have limited data
- Generic leagues have many markets but less specificity

## Future Improvements
- Fuzzy league matching
- League aliases (e.g., "EPL" → "Premier League")
- Automatic league detection from teams
- Real-time league updates from APIs