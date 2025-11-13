# Ominari Dashboard Filters Guide

## Current Status
The dashboard is now running at **http://localhost:8888** with the following filters:
- **Sports**: Soccer only
- **Nations**: England, International, Europe
- **Leagues**: All leagues from selected nations

This shows upcoming English soccer games along with international competitions like Champions League, Europa League, and World Cup matches.

## Available Filter Variables

### 1. Sport Filter (`ALLOWED_SPORTS`)
Filter by sport type. Available options:
- Soccer
- Basketball
- Baseball
- Hockey
- American Football
- Tennis
- Golf
- Cricket
- MMA
- Esports
- Handball

Example: `export ALLOWED_SPORTS="Soccer,Basketball"`

### 2. Nation Filter (`ALLOWED_NATIONS`)
Filter by country/region. Common options:
- **European**: England, Spain, Italy, Germany, France, Portugal, Netherlands, Belgium
- **Americas**: USA, Brazil, Argentina, Mexico, Canada
- **Asia**: Japan, China, South Korea, Australia
- **Continental**: Europe, International, North America, South America
- **Organizations**: UEFA, FIFA, CONMEBOL, CONCACAF

Example: `export ALLOWED_NATIONS="England,Spain,Italy"`

### 3. League Filter (`ALLOWED_LEAGUES`)
Filter by specific leagues:
- **Soccer**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League
- **US Sports**: NFL, NBA, MLB, NHL, MLS
- **International**: World Cup, European Championship, Copa America

Example: `export ALLOWED_LEAGUES="Premier League,Champions League"`

## Quick Filter Commands

### 1. Use the Filter Control Tool
```bash
./dashboard_filter_control.py
```

This provides an interactive menu with presets:
1. English Soccer (Default)
2. Top 5 European Leagues
3. All Soccer
4. US Sports
5. Premier League Only
6. Champions League & International
7. All Sports - No Filter

### 2. Manual Filter Setting
```bash
# English Premier League only
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England"
export ALLOWED_LEAGUES="Premier League"
./restart_dashboard_english_soccer.sh

# Top European soccer
export ALLOWED_SPORTS="Soccer"
export ALLOWED_NATIONS="England,Spain,Italy,Germany,France"
export ALLOWED_LEAGUES=""
./restart_dashboard_english_soccer.sh

# US Sports
export ALLOWED_SPORTS="Basketball,Baseball,American Football,Hockey"
export ALLOWED_NATIONS="USA"
export ALLOWED_LEAGUES=""
./restart_dashboard_english_soccer.sh
```

## Dashboard Features

### Real-time Updates
- Automatic refresh of odds data
- WebSocket connection for live updates
- 30-second cache for performance

### Performance Features
- Rate limiting: 60 requests/minute
- Caching layer for frequently accessed data
- Optimized queries with filtering

### Data Display
- Shows team matchups
- Current odds from various sources
- League and nation information
- Blockchain connection status

## Troubleshooting

### Check Current Filters
```bash
echo "Sports: $ALLOWED_SPORTS"
echo "Nations: $ALLOWED_NATIONS"
echo "Leagues: $ALLOWED_LEAGUES"
```

### View Dashboard Logs
```bash
tail -f real_odds_dashboard.out
```

### Restart Dashboard
```bash
./restart_dashboard_english_soccer.sh
```

## Default Configuration

The dashboard defaults to showing English and International soccer matches, which includes:
- All English leagues (Premier League, Championship, etc.)
- UEFA competitions (Champions League, Europa League)
- International competitions (World Cup, friendlies)

This provides a good balance of local English football and major international competitions.