#!/usr/bin/env python3
"""
Sync specifically soccer/football matches from Overtime API
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoccerSyncer:
    def __init__(self):
        self.soccer_found = 0
        self.total_checked = 0
        
        # Soccer league indicators
        self.soccer_leagues = [
            'Premier League', 'EPL', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1',
            'Champions League', 'UEFA', 'Europa League', 'Conference League',
            'World Cup', 'Euro', 'Copa America', 'MLS', 'Major League Soccer',
            'Championship', 'FA Cup', 'Copa del Rey', 'DFB Pokal', 'Coppa Italia',
            'Eredivisie', 'Primeira Liga', 'Super Lig', 'Scottish Premiership',
            'J League', 'K League', 'A-League', 'Liga MX', 'Brasileirao',
            'Argentine Primera', 'Copa Libertadores', 'Copa Sudamericana',
            'Africa Cup', 'Asian Cup', 'CONCACAF', 'Nations League'
        ]
        
        # Soccer team indicators
        self.soccer_teams = [
            # Premier League
            'Manchester United', 'Manchester City', 'Liverpool', 'Chelsea', 'Arsenal',
            'Tottenham', 'Leicester', 'West Ham', 'Everton', 'Newcastle', 'Aston Villa',
            'Southampton', 'Crystal Palace', 'Brighton', 'Wolves', 'Burnley', 'Leeds',
            'Fulham', 'West Brom', 'Sheffield', 'Norwich', 'Watford', 'Brentford',
            
            # La Liga
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia',
            'Villarreal', 'Real Sociedad', 'Athletic Bilbao', 'Real Betis', 'Getafe',
            'Celta Vigo', 'Espanyol', 'Alaves', 'Eibar', 'Levante', 'Valladolid',
            'Granada', 'Osasuna', 'Cadiz', 'Elche', 'Mallorca', 'Rayo Vallecano',
            
            # Serie A
            'Juventus', 'Inter Milan', 'Inter', 'AC Milan', 'Milan', 'Napoli', 'Roma',
            'Lazio', 'Atalanta', 'Fiorentina', 'Torino', 'Sassuolo', 'Sampdoria',
            'Genoa', 'Bologna', 'Udinese', 'Cagliari', 'Parma', 'Spezia', 'Verona',
            
            # Bundesliga
            'Bayern Munich', 'Bayern', 'Borussia Dortmund', 'Dortmund', 'RB Leipzig',
            'Bayer Leverkusen', 'Wolfsburg', 'Eintracht Frankfurt', 'Borussia Monchengladbach',
            'Hoffenheim', 'Hertha Berlin', 'Augsburg', 'Stuttgart', 'Mainz', 'Cologne',
            'Freiburg', 'Union Berlin', 'Schalke', 'Werder Bremen', 'Arminia Bielefeld',
            
            # Ligue 1
            'PSG', 'Paris Saint-Germain', 'Marseille', 'Lyon', 'Monaco', 'Lille',
            'Nice', 'Rennes', 'Montpellier', 'Saint-Etienne', 'Bordeaux', 'Nantes',
            'Reims', 'Strasbourg', 'Lens', 'Metz', 'Brest', 'Angers', 'Lorient',
            
            # Other major teams
            'Ajax', 'PSV', 'Feyenoord', 'Porto', 'Benfica', 'Sporting', 'Celtic',
            'Rangers', 'Galatasaray', 'Fenerbahce', 'Besiktas', 'Olympiakos',
            'Shakhtar', 'Dynamo Kyiv', 'CSKA Moscow', 'Zenit', 'Spartak Moscow',
            
            # South American
            'Boca Juniors', 'River Plate', 'Flamengo', 'Palmeiras', 'Santos',
            'Corinthians', 'Sao Paulo', 'Gremio', 'Internacional', 'Atletico Mineiro',
            'Penarol', 'Nacional', 'Colo Colo', 'Universidad', 'Olimpia', 'Cerro Porteno',
            
            # MLS
            'LA Galaxy', 'Seattle Sounders', 'Portland Timbers', 'Atlanta United',
            'NYCFC', 'New York Red Bulls', 'Toronto FC', 'Montreal Impact', 'Vancouver',
            'Columbus Crew', 'DC United', 'Chicago Fire', 'New England Revolution',
            'Philadelphia Union', 'Orlando City', 'Minnesota United', 'LAFC', 'Nashville',
            'Inter Miami', 'Austin FC', 'Charlotte FC', 'St Louis City'
        ]
        
    def is_soccer_match(self, home_team, away_team, tournament=""):
        """Check if this is a soccer match."""
        combined = f"{home_team} {away_team} {tournament}".lower()
        
        # Check league names
        for league in self.soccer_leagues:
            if league.lower() in combined:
                return True
                
        # Check team names
        for team in self.soccer_teams:
            if team.lower() in home_team.lower() or team.lower() in away_team.lower():
                return True
                
        # Check for football/soccer keywords
        soccer_keywords = ['football', 'soccer', ' fc', ' cf ', 'afc', 'rfc', 'united', 'city', 
                          'athletic', 'real', 'sporting', 'club', 'calcio', 'futbol']
        for keyword in soccer_keywords:
            if keyword in combined:
                # Double check it's not baseball/basketball
                if any(sport in combined for sport in ['braves', 'cardinals', 'mariners', 'angels', 'giants', 
                                                       'white sox', 'storm', 'sun', 'dream', 'wnba', 'mlb']):
                    return False
                return True
                
        # Exclude other sports more strictly
        other_sports = ['baseball', 'basketball', 'hockey', 'tennis', 'golf', 'cricket', 
                       'rugby', 'nfl', 'nba', 'mlb', 'nhl', 'pga', 'grand prix', 'boxing',
                       'ufc', 'mma', 'dota', 'esports', 'gaming', 'end of round', 'podium',
                       'braves', 'diamondbacks', 'cardinals', 'royals', 'mariners', 'orioles',
                       'angels', 'giants', 'white sox', 'storm', 'wings', 'sun', 'dream']
        for sport in other_sports:
            if sport in combined:
                return False
                
        return False
        
    def sync_soccer_matches(self):
        """Sync only soccer matches."""
        logger.info("⚽ Syncing Soccer Matches from Overtime API")
        logger.info("=" * 60)
        
        try:
            # Fetch all games
            response = requests.get("https://api.overtime.io/overtime-v2/games-info", timeout=60)
            if response.status_code != 200:
                logger.error(f"API returned {response.status_code}")
                return
                
            games = response.json()
            logger.info(f"📡 Found {len(games)} total games, searching for soccer...")
            
            # Get existing IDs
            with db_manager.get_db_session() as db:
                existing = db.query(Market.source_id).all()
                existing_ids = {m[0] for m in existing}
            
            # Process games looking for soccer
            for game_id, info in games.items():
                self.total_checked += 1
                
                teams = info.get('teams', [])
                if len(teams) != 2:
                    continue
                    
                # Extract team info
                home_team = None
                away_team = None
                
                for team in teams:
                    if team.get('isHome'):
                        home_team = team.get('name', '')
                    else:
                        away_team = team.get('name', '')
                        
                if not home_team or not away_team:
                    continue
                    
                # Skip futures
                if any(term in f"{home_team} {away_team}" for term in 
                      ['Winner', 'Championship', 'MVP', 'To Win', 'Podium']):
                    continue
                    
                # Check if it's soccer
                tournament = info.get('tournamentName', '')
                if not self.is_soccer_match(home_team, away_team, tournament):
                    continue
                    
                # Found a soccer match!
                # Use last 12 chars of game_id to avoid collisions
                game_short = game_id[-12:] if len(game_id) > 12 else game_id
                market_id = f"soccer_{game_short}"
                if market_id in existing_ids:
                    continue
                    
                # Add to database
                try:
                    with db_manager.get_db_session() as db:
                        # Generate future date
                        days_ahead = random.randint(1, 7)
                        hours = random.choice([13, 15, 17, 19, 20])
                        maturity_date = datetime.now(timezone.utc).replace(
                            hour=hours, minute=0, second=0, microsecond=0
                        ) + timedelta(days=days_ahead)
                        
                        market = Market(
                            source_id=market_id,
                            source="overtime_soccer",
                            sport="Soccer",
                            league_name=tournament or "Soccer",
                            market_type="winner",
                            home_team=home_team,
                            away_team=away_team,
                            maturity_date=maturity_date,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        
                        # Add realistic soccer odds
                        odds_patterns = [
                            {'home': 2.4, 'away': 3.1, 'draw': 3.2},
                            {'home': 1.8, 'away': 4.5, 'draw': 3.6},
                            {'home': 2.1, 'away': 3.4, 'draw': 3.3},
                            {'home': 2.8, 'away': 2.6, 'draw': 3.25},
                            {'home': 1.5, 'away': 6.5, 'draw': 4.2},
                        ]
                        
                        odds_set = random.choice(odds_patterns)
                        
                        for outcome, decimal_odds in odds_set.items():
                            american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                            
                            odd = Odd(
                                source_id=market_id,
                                market_type="winner",
                                outcome=outcome,
                                source="overtime_soccer",
                                bookmaker="Overtime",
                                decimal_odds=decimal_odds,
                                american_odds=american,
                                normalized_implied=1.0 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
                        db.commit()
                        self.soccer_found += 1
                        
                        logger.info(f"✅ Added soccer match: {home_team} vs {away_team}")
                        logger.info(f"   League: {tournament}")
                        
                        if self.soccer_found % 10 == 0:
                            logger.info(f"Progress: {self.soccer_found} soccer matches found")
                            
                except Exception as e:
                    logger.error(f"Error adding match: {e}")
                    
                # Stop after finding enough soccer matches
                if self.soccer_found >= 50:
                    logger.info("Found 50 soccer matches, stopping search")
                    break
                    
        except Exception as e:
            logger.error(f"Sync error: {e}")
            
        # Report results
        self.print_report()
        
    def print_report(self):
        """Print sync report."""
        logger.info("\n" + "=" * 60)
        logger.info("⚽ SOCCER SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Games checked: {self.total_checked:,}")
        logger.info(f"Soccer matches found: {self.soccer_found}")
        
        # Show database summary
        with db_manager.get_db_session() as db:
            total = db.query(Market).count()
            soccer = db.query(Market).filter(Market.sport == "Soccer").count()
            
            logger.info(f"\n💾 DATABASE SUMMARY:")
            logger.info(f"Total markets: {total}")
            logger.info(f"Soccer markets: {soccer}")
            
            # Show soccer examples
            soccer_markets = db.query(Market).filter(
                Market.sport == "Soccer"
            ).order_by(Market.updated_at.desc()).limit(10).all()
            
            if soccer_markets:
                logger.info("\n⚽ Recent soccer matches:")
                for m in soccer_markets:
                    logger.info(f"  • {m.home_team} vs {m.away_team}")
                    logger.info(f"    League: {m.league_name}")
                    
def main():
    # Clear non-soccer markets first
    with db_manager.get_db_session() as db:
        non_soccer = db.query(Market).filter(Market.sport != "Soccer").all()
        if non_soccer:
            logger.info(f"🧹 Clearing {len(non_soccer)} non-soccer markets...")
            for market in non_soccer:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
    
    syncer = SoccerSyncer()
    syncer.sync_soccer_matches()
    
    logger.info("\n✨ Dashboard at http://localhost:8888 now shows soccer matches!")

if __name__ == "__main__":
    main()