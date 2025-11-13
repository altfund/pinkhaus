# Sport Filter Configuration Guide

## Overview
The Ominari dashboard now supports configurable sport filtering, allowing you to focus on specific sports. By default, it shows only Soccer markets, but this can be easily changed.

## Default Configuration
- **Default Sport**: Soccer only
- **Market Limit**: 150 markets
- **Cache TTL**: 30 seconds

## How to Configure Sports

### Method 1: Environment Variable
Set the `ALLOWED_SPORTS` environment variable before starting the dashboard:

```bash
# Soccer only (default)
export ALLOWED_SPORTS=Soccer
python web_dashboard_real_odds.py

# Multiple sports
export ALLOWED_SPORTS=Soccer,Basketball,Baseball
python web_dashboard_real_odds.py

# All major sports
export ALLOWED_SPORTS=Soccer,Basketball,Baseball,Hockey,American Football
python web_dashboard_real_odds.py
```

### Method 2: In Flox Environment
```bash
# Single sport
ALLOWED_SPORTS=Soccer flox activate -- python web_dashboard_real_odds.py

# Multiple sports
ALLOWED_SPORTS=Soccer,Basketball,Baseball flox activate -- python web_dashboard_real_odds.py
```

### Method 3: Edit Configuration File
Edit `dashboard_config.py` and change the default:
```python
# Change this line:
ALLOWED_SPORTS = os.environ.get('ALLOWED_SPORTS', 'Soccer').split(',')

# To something like:
ALLOWED_SPORTS = os.environ.get('ALLOWED_SPORTS', 'Soccer,Basketball,Baseball').split(',')
```

## Available Sports
The following sports are available in the system:
- Soccer (4,327 markets)
- Basketball (72 markets)
- Baseball (221 markets)
- Hockey (48 markets)
- American Football (39 markets)
- Cricket (37 markets)
- Tennis (1 market)
- Golf (10 markets)
- MMA (18 markets)
- Esports (350 markets)
- Handball (11 markets)

## Examples

### Soccer Only (Default)
```bash
python web_dashboard_real_odds.py
```
Dashboard will show: "Live Markets - Real Odds (Soccer)"

### Soccer and Basketball
```bash
export ALLOWED_SPORTS=Soccer,Basketball
python web_dashboard_real_odds.py
```
Dashboard will show: "Live Markets - Real Odds (Soccer, Basketball)"

### All Ball Sports
```bash
export ALLOWED_SPORTS=Soccer,Basketball,Baseball,American Football,Handball
python web_dashboard_real_odds.py
```

### Esports Only
```bash
export ALLOWED_SPORTS=Esports
python web_dashboard_real_odds.py
```

## Verification
1. The dashboard header will display which sports are being shown
2. Check the logs for: `Filtering markets for sports: ['Soccer']`
3. Only markets from allowed sports will appear in the dashboard

## Docker Usage
When using Docker, add the environment variable to docker-compose.yml:
```yaml
ominari-dashboard:
  environment:
    - ALLOWED_SPORTS=Soccer,Basketball
```

Or run with:
```bash
docker run -e ALLOWED_SPORTS=Soccer,Basketball ominari-dashboard
```

## Performance Note
Filtering by sport improves performance by:
- Reducing database query results
- Decreasing network transfer
- Improving cache efficiency
- Faster dashboard rendering

## Troubleshooting
- If no markets appear, check that the sport name matches exactly (case-sensitive)
- Valid sport names: Soccer, Basketball, Baseball, Hockey, etc.
- Invalid: soccer, SOCCER, Football (use "Soccer" not "Football")
- Check logs for actual sports in database if unsure