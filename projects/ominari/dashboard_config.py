"""
Dashboard configuration for Ominari
"""
import os

# Sport filtering configuration
# Default: Soccer only
# To enable multiple sports, set ALLOWED_SPORTS environment variable
# Example: export ALLOWED_SPORTS=Soccer,Basketball,Baseball,Hockey

# Get allowed sports from environment or use default
ALLOWED_SPORTS = os.environ.get('ALLOWED_SPORTS', 'Soccer').split(',')
ALLOWED_SPORTS = [sport.strip() for sport in ALLOWED_SPORTS if sport.strip()]

# Get allowed leagues from environment (optional filter)
# Example: export ALLOWED_LEAGUES=Premier League,La Liga,Serie A
ALLOWED_LEAGUES = os.environ.get('ALLOWED_LEAGUES', '').split(',')
ALLOWED_LEAGUES = [league.strip() for league in ALLOWED_LEAGUES if league.strip()]

# Get allowed nations from environment (optional filter)
# Example: export ALLOWED_NATIONS=England,Spain,Italy
ALLOWED_NATIONS = os.environ.get('ALLOWED_NATIONS', '').split(',')
ALLOWED_NATIONS = [nation.strip() for nation in ALLOWED_NATIONS if nation.strip()]

# Available sports in the system
AVAILABLE_SPORTS = [
    'Soccer',
    'Basketball', 
    'Baseball',
    'Hockey',
    'American Football',
    'Cricket',
    'Tennis',
    'Golf',
    'MMA',
    'Esports',
    'Handball'
]

# Dashboard settings
DASHBOARD_SETTINGS = {
    'markets_limit': 150,  # Maximum markets to display
    'cache_ttl': 30,       # Cache time-to-live in seconds
    'default_odds_filter': [2.5, 2.8, 3.0],  # Default odds to exclude
}

# League mappings for better display
LEAGUE_DISPLAY_NAMES = {
    'N/A': 'General',
    'Regular Season': 'Regular',
    'Playoffs': 'Playoffs',
    'UEFA Champions League': 'Champions League',
    'English Premier League': 'Premier League',
    'La Liga': 'La Liga',
    'Serie A': 'Serie A',
    'Bundesliga': 'Bundesliga',
    'Ligue 1': 'Ligue 1'
}

def get_display_league(league_name):
    """Get display name for league"""
    return LEAGUE_DISPLAY_NAMES.get(league_name, league_name)

# Validate allowed sports
def validate_sports():
    """Validate that allowed sports are available"""
    invalid_sports = [sport for sport in ALLOWED_SPORTS if sport not in AVAILABLE_SPORTS]
    if invalid_sports:
        print(f"Warning: Invalid sports in ALLOWED_SPORTS: {invalid_sports}")
        print(f"Available sports: {AVAILABLE_SPORTS}")
    return [sport for sport in ALLOWED_SPORTS if sport in AVAILABLE_SPORTS]

# Get validated sports
ALLOWED_SPORTS = validate_sports() or ['Soccer']  # Default to Soccer if no valid sports

print(f"Dashboard configured for sports: {ALLOWED_SPORTS}")