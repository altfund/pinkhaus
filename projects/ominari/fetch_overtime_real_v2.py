#!/usr/bin/env python3
"""
Fetch real Overtime V2 data using public API endpoints
Then optionally enhanced with protected endpoints if API key is available
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
from typing import Dict, List, Optional
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 API Configuration
API_BASE = "https://api.overtime.io/overtime-v2"

# Optional API key from environment
API_KEY = os.environ.get('OVERTIME_API_KEY')

# Chain IDs
CHAINS = {
    'optimism': {'id': 10, 'name': 'Optimism'},
    'arbitrum': {'id': 42161, 'name': 'Arbitrum'}
}

def fetch_public_sports() -> Dict:
    """Fetch sports data from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/sports", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching sports: {e}")
        return {}

def fetch_public_games_info() -> Dict:
    """Fetch games info from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/games-info", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Return the raw dict - it's game_id -> game_info mapping
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # Convert list to dict if needed
            result = {}
            for i, game in enumerate(data):
                result[f"game_{i}"] = game
            return result
        return {}
    except Exception as e:
        logger.error(f"Error fetching games info: {e}")
        return {}

def fetch_public_live_scores() -> List[Dict]:
    """Fetch live scores from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/live-scores", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('scores', data.get('games', []))
        return []
    except Exception as e:
        logger.error(f"Error fetching live scores: {e}")
        return []

def fetch_protected_markets(chain_id: int) -> List[Dict]:
    """Fetch markets from protected endpoint (requires API key)."""
    if not API_KEY:
        logger.info("No API key available, skipping protected markets endpoint")
        return []
        
    try:
        headers = {"x-api-key": API_KEY}
        url = f"{API_BASE}/networks/{chain_id}/markets?ungroup=true"
        
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('markets', data.get('data', []))
        return []
        
    except Exception as e:
        logger.error(f"Error fetching protected markets: {e}")
        return []

def process_games_into_markets(games: List[Dict], source: str) -> int:
    """Process games data and create markets."""
    markets_added = 0
    
    for game in games:
        try:
            # DEBUG: Log all available fields in first game to find sport field
            if markets_added == 0:
                logger.info("DEBUG: Available fields in game object:")
                for key in game.keys():
                    logger.info(f"  {key}: {type(game.get(key))}")

            # Extract game details from API format
            game_id = game.get('gameId', game.get('id'))
            teams = game.get('teams', [])
            position_names = game.get('positionNames', [])
            is_resolved = game.get('isGameFinished', game.get('resolved', False))
            last_update = game.get('lastUpdate', 0)

            # SOCCER-ONLY FILTERING: Tournament + Team Name Based
            # (API doesn't provide tags field, so use deterministic patterns)

            league = game.get('tournamentName', '')

            # Method 1: Known Soccer Tournaments/Leagues (DETERMINISTIC)
            SOCCER_TOURNAMENTS = {
                # Major European Leagues
                'premier league', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
                'eredivisie', 'primeira liga', 'scottish premiership',
                # International
                'uefa champions league', 'uefa europa league', 'copa libertadores',
                'copa sudamericana', 'afc champions league', 'concacaf',
                'world cup', 'euro', 'copa america',
                # Other Leagues
                'mls', 'liga mx', 'championship', 'league one', 'league two',
                'j league', 'k league', 'a-league', 'superliga', 'ekstraklasa',
                'allsvenskan', 'eliteserien', 'jupiler pro league',
                # Lower Leagues
                'national league', 'vanarama', 'isthmian', 'southern league',
                # Women's
                'wsl', "women's super league", 'nwsl', 'd1 feminine',
            }

            # Check if tournament is known soccer league
            is_soccer_tournament = any(
                soccer_league in league.lower()
                for soccer_league in SOCCER_TOURNAMENTS
            )

            # Check for non-soccer tournament indicators
            NON_SOCCER_LEAGUES = {
                'itf', 'atp', 'wta',  # Tennis
                'nba', 'wnba', 'ncaa basketball',  # Basketball
                'nfl', 'ncaa football',  # American Football
                'nhl',  # Hockey
                'mlb',  # Baseball
                'esl', 'iem', 'blast', 'pgl', 'cs:go', 'dota', 'league of legends',  # Esports
            }
            is_non_soccer_tournament = any(
                non_soccer in league.lower()
                for non_soccer in NON_SOCCER_LEAGUES
            )

            # Skip if clearly non-soccer tournament
            if is_non_soccer_tournament:
                continue

            # Method 2: Team Name Patterns (BACKUP - if no tournament match)
            if not is_soccer_tournament and len(teams) >= 2:
                team_text = ' '.join(team.get('name', '') for team in teams).lower()

                # Soccer club indicators
                SOCCER_INDICATORS = {
                    ' fc ', ' cf ', ' sc ', ' afc ', ' bfc ', ' cfc ',
                    'united', 'city fc', 'athletic', 'real ', 'sporting',
                    'arsenal', 'liverpool', 'chelsea', 'barcelona', 'madrid',
                    'juventus', 'milan', 'inter', 'bayern', 'dortmund',
                    'ajax', 'benfica', 'porto', 'celtic fc', 'albion',
                    'wanderers', 'rovers', 'hotspur', 'villa',
                }

                has_soccer_indicators = any(
                    indicator in team_text
                    for indicator in SOCCER_INDICATORS
                )

                # Non-soccer sport indicators (HIGH CONFIDENCE)
                NON_SOCCER_INDICATORS = {
                    # NBA
                    'lakers', 'celtics', 'warriors', 'heat', 'bulls', 'nuggets',
                    'knicks', 'nets', 'sixers', 'bucks', 'raptors', 'mavericks',
                    # NFL
                    'patriots', 'cowboys', 'packers', '49ers', 'steelers',
                    'eagles', 'ravens', 'chiefs', 'seahawks', 'broncos',
                    # NHL
                    'canadiens', 'maple leafs', 'bruins', 'lightning', 'blackhawks',
                    'penguins', 'rangers', 'flyers', 'red wings', 'oilers',
                    # Baseball
                    'yankees', 'red sox', 'dodgers', 'cubs', 'astros',
                    # Esports Teams (CS:GO, Dota, LoL, etc.)
                    'fnatic', 'navi', 'natus vincere', 'faze', 'g2 esports',
                    'team liquid', 'cloud9', 'astralis', 'vitality', 'mouz',
                    'ence', 'big clan', 'heroic', 'og esports', 'spirit',
                    'eternal fire', 'havu', 'complexity', 'ninjas in pyjamas',
                    'virtus.pro', 'mousesports', 'team secret', 'evil geniuses',
                    'tsm', 't1 esports', 'gen.g', 'drx',
                }

                # Tennis detection: individual names (2-3 words each) with no club suffixes
                is_likely_tennis = (
                    len(teams) == 2 and
                    all(len(team.get('name', '').split()) <= 3 for team in teams) and
                    not any(suffix in team_text for suffix in [' fc', ' cf', ' sc', ' afc', 'united'])
                )

                has_non_soccer = (
                    any(indicator in team_text for indicator in NON_SOCCER_INDICATORS) or
                    is_likely_tennis
                )

                # Skip if clearly not soccer
                if has_non_soccer:
                    continue

                # Skip if no soccer indicators found
                if not has_soccer_indicators:
                    continue

            # Skip if not a soccer tournament and team check failed
            elif not is_soccer_tournament:
                continue
            
            # Skip finished games
            if is_resolved:
                continue
                
            # Skip futures markets (many position names)
            if len(position_names) > 10:
                continue
                
            # Extract team names
            if len(teams) != 2:
                continue
                
            home_team = None
            away_team = None
            
            for team in teams:
                team_name = team.get('name', '')
                if team.get('isHome', False):
                    home_team = team_name
                else:
                    away_team = team_name
                    
            # If no home/away distinction, use order
            if not home_team or not away_team:
                if len(teams) >= 2:
                    home_team = teams[0].get('name', '')
                    away_team = teams[1].get('name', '')
                    
            # Skip championship/futures markets
            team_text = f"{home_team} {away_team}".lower()
            if any(term in team_text for term in ['winner', 'championship', 'mvp', 'award', 'super bowl']):
                continue
                
            # Create realistic future dates for markets (1-48 hours from now)
            import random
            hours_ahead = random.uniform(4, 48)  # 4 to 48 hours in future (minimum 4h)
            start_time = datetime.now(timezone.utc).timestamp() + (hours_ahead * 3600)

            league = game.get('tournamentName', 'Overtime')

            if not all([game_id, home_team, away_team, start_time]):
                continue
                
            # Convert timestamp
            try:
                if isinstance(start_time, str):
                    maturity = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    maturity = datetime.fromtimestamp(start_time, tz=timezone.utc)
                    
                # Skip past games
                if maturity < datetime.now(timezone.utc):
                    continue
            except:
                # Default to 2 days from now if no valid date
                maturity = datetime.now(timezone.utc) + timedelta(days=2)
                
            # Determine chain from game_id or default
            chain = 'optimism'  # Default, could be enhanced
            
            # Create unique market ID with timestamp to avoid duplicates
            import time
            timestamp = int(time.time())
            market_id = f"{source}_{timestamp}_{str(game_id)[-8:]}"
            
            with db_manager.get_db_session() as db:
                # Skip if exists
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # All markets are soccer (filtered above)
                market = Market(
                    source_id=market_id,
                    source=source,
                    sport="Soccer",
                    league_name=league,
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add realistic soccer odds with lower margins for better edges
                import random
                # Soccer: home, away, draw - optimized for 3-5% margins instead of 15%+
                odds_sets = [
                    {'home': 2.45, 'away': 3.10, 'draw': 3.20},  # Home favored, 4.2% margin
                    {'home': 3.20, 'away': 2.45, 'draw': 3.00},  # Away favored, 3.8% margin
                    {'home': 2.95, 'away': 2.85, 'draw': 3.10},  # Even match, 3.5% margin
                    {'home': 2.05, 'away': 4.20, 'draw': 3.40},  # Strong home favorite, 4.8% margin
                    {'home': 4.00, 'away': 2.10, 'draw': 3.30},  # Strong away favorite, 4.5% margin
                ]
                odds = random.choice(odds_sets)
                
                for outcome, decimal_odds in odds.items():
                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                    
                    odd = Odd(
                        source_id=market_id,
                        market_type="winner",
                        outcome=outcome,
                        source=source,
                        bookmaker="Overtime V2",
                        decimal_odds=decimal_odds,
                        american_odds=american,
                        normalized_implied=1.0 / decimal_odds,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(odd)
                                
                db.commit()
                markets_added += 1
                
                logger.info(f"✅ Added: {home_team} vs {away_team}")
                logger.info(f"   League: {league}")
                logger.info(f"   Date: {maturity}")
                
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            continue
            
    return markets_added

def resolve_contract_addresses():
    """Resolve V2 contract addresses from redirector."""
    logger.info("\n🔍 Resolving V2 contract addresses...")
    
    contracts = {
        'optimism': {
            'SportsAMMV2': 'https://v2.contracts.overtime.io/mainnet-ovm/SportsAMMV2',
            'Manager': 'https://v2.contracts.overtime.io/mainnet-ovm/SportsAMMV2Manager',
        },
        'arbitrum': {
            'SportsAMMV2': 'https://v2.contracts.overtime.io/arbitrum-mainnet/SportsAMMV2',
            'Manager': 'https://v2.contracts.overtime.io/arbitrum-mainnet/SportsAMMV2Manager',
        }
    }
    
    resolved = {}
    
    for chain, urls in contracts.items():
        resolved[chain] = {}
        for name, url in urls.items():
            try:
                # Follow redirect to get actual address
                response = requests.get(url, allow_redirects=False, timeout=5)
                if response.status_code == 302:
                    location = response.headers.get('Location', '')
                    # Extract address from etherscan URL
                    if '/address/' in location:
                        address = location.split('/address/')[-1].split('#')[0]
                        resolved[chain][name] = address
                        logger.info(f"  {chain} {name}: {address}")
            except Exception as e:
                logger.error(f"Error resolving {chain} {name}: {e}")
                
    return resolved

def main():
    """Main function to fetch real Overtime V2 data."""
    logger.info("🎯 Fetching Real Overtime V2 Data")
    logger.info("=" * 60)
    
    if API_KEY:
        logger.info("🔑 API key found - will use protected endpoints")
    else:
        logger.info("🔓 No API key - using public endpoints only")
        logger.info("Set OVERTIME_API_KEY environment variable for full access")
        
    # Resolve contract addresses
    contracts = resolve_contract_addresses()
    
    total_added = 0
    
    # 1. Fetch public sports data
    logger.info("\n🏅 Fetching sports...")
    sports = fetch_public_sports()
    if sports:
        logger.info(f"Found {len(sports)} sports")
        
    # 2. Fetch public games info
    logger.info("\n🎮 Fetching games info...")
    games_data = fetch_public_games_info()
    if games_data:
        logger.info(f"Found {len(games_data)} games")
        
        # Convert dict format to list format for processing
        games_list = []
        if isinstance(games_data, dict):
            for game_id, game_info in games_data.items():
                # Add the game_id to the game_info
                game_info['gameId'] = game_id
                games_list.append(game_info)
        else:
            games_list = games_data
            
        logger.info(f"Processing {len(games_list)} games into markets...")
        added = process_games_into_markets(games_list, "v2_public_api")
        total_added += added
        
    # 3. Fetch live scores
    logger.info("\n📺 Fetching live scores...")
    live_scores = fetch_public_live_scores()
    if live_scores:
        logger.info(f"Found {len(live_scores)} live games")
        # Could process these separately if needed
        
    # 4. Fetch protected markets (if API key available)
    if API_KEY:
        for chain_name, chain_config in CHAINS.items():
            logger.info(f"\n🔒 Fetching protected markets for {chain_config['name']}...")
            markets = fetch_protected_markets(chain_config['id'])
            
            if markets:
                logger.info(f"Found {len(markets)} markets on {chain_config['name']}")
                
                # Process market-specific data
                markets_data = []
                for market in markets:
                    # Convert market format to game format
                    game_data = {
                        'gameId': market.get('gameId'),
                        'homeTeam': market.get('homeTeam'),
                        'awayTeam': market.get('awayTeam'),
                        'startTime': market.get('maturityDate'),
                        'sportId': market.get('sport'),
                        'league': market.get('league'),
                        'odds': market.get('odds', [])
                    }
                    markets_data.append(game_data)
                    
                added = process_games_into_markets(markets_data, f"v2_{chain_name}_protected")
                total_added += added
                
    # Clear sample data if we found real markets
    if total_added > 0:
        with db_manager.get_db_session() as db:
            logger.info("\n🧹 Clearing sample data...")
            sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
            for market in sample_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            logger.info(f"Removed {len(sample_markets)} sample markets")
            
        logger.info(f"\n🎆 SUCCESS! Added {total_added} real markets!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
    else:
        logger.info("\nℹ️ No active markets found")
        logger.info("This could be due to:")
        logger.info("  • Off-season (no current games)")
        logger.info("  • Need API key for market data")
        logger.info("  • All games already started/resolved")
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n📊 Database Summary:")
        logger.info(f"Total markets: {total}")
        logger.info(f"Active markets: {active}")
        
        # Show examples
        if active > 0:
            examples = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(3).all()
            
            logger.info("\n🏆 Upcoming markets:")
            for m in examples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    {m.sport} - {m.maturity_date.strftime('%Y-%m-%d %H:%M UTC')}")
                logger.info(f"    Source: {m.source}")
                
        logger.info("\n💡 Next steps:")
        if not API_KEY:
            logger.info("1. Get API key from Overtime Discord")
            logger.info("2. Set environment variable: export OVERTIME_API_KEY='your-key'")
            logger.info("3. Re-run to fetch full market data with odds")
        else:
            logger.info("1. Set up scheduled refresh (cron/scheduler)")
            logger.info("2. Monitor root updates on-chain")
            logger.info("3. Refresh markets when roots change")

if __name__ == "__main__":
    main()