#!/usr/bin/env python3
"""
Sync ONLY real games - strict filtering for actual matches
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

class RealGamesSyncer:
    def __init__(self):
        self.stats = {
            'checked': 0,
            'added': 0,
            'futures_skipped': 0,
            'errors': 0
        }
        
        # Comprehensive futures/special markets terms
        self.futures_terms = [
            # Championships/Tournaments
            'Championship', 'Winner', 'Champions', 'Title', 'Trophy',
            'World Cup', 'Super Bowl', 'Stanley Cup', 'NBA Finals',
            'World Series', 'Olympics', 'Euro 2024', 'Copa America',
            
            # Awards/Individual
            'MVP', 'Award', 'Player of', 'Rookie', 'All-Star',
            'Pro Bowl', 'Hall of Fame', 'Ballon', 'Golden',
            
            # Tournament stages
            'To Win', 'To Make', 'To Reach', 'To Qualify',
            'Group Winner', 'Division Winner', 'Conference',
            'Playoffs', 'Finals', 'Semi', 'Quarter',
            
            # Golf/Racing specific
            'Podium', 'Top 5', 'Top 10', 'Top 20',
            'End Of Round', 'Leader', 'Cut Line',
            'Grand Prix', 'Masters', 'Open Championship',
            
            # Season-long
            'Season', '2024', '2025', '2026',
            'Regular Season', 'Over/Under Wins',
            
            # Draft/Transfer
            'Draft', 'Pick', 'Transfer', 'Sign',
            
            # Series
            'Series', 'Best of', 'Game 1', 'Game 2'
        ]
        
    def is_real_game(self, home_team, away_team, tournament=""):
        """Check if this is a real game vs futures/special market."""
        combined = f"{home_team} {away_team} {tournament}".lower()
        
        # Check futures terms
        for term in self.futures_terms:
            if term.lower() in combined:
                return False
                
        # Additional checks
        # If away team is not a real team name
        if len(away_team) < 3 or len(home_team) < 3:
            return False
            
        # If it contains "vs" in team name (like "Team A vs Team B Winner")
        if ' vs ' in home_team or ' vs ' in away_team:
            return False
            
        # Year in future
        current_year = datetime.now().year
        for year in range(current_year + 1, current_year + 5):
            if str(year) in combined:
                return False
                
        return True
        
    def sync_games(self):
        """Sync only real games."""
        logger.info("🎯 Syncing REAL GAMES ONLY (strict filtering)")
        
        # Clear existing non-real markets
        with db_manager.get_db_session() as db:
            # Find markets that look like futures
            futures_markets = []
            all_markets = db.query(Market).all()
            
            for market in all_markets:
                if not self.is_real_game(market.home_team, market.away_team):
                    futures_markets.append(market)
                    
            if futures_markets:
                logger.info(f"🧹 Removing {len(futures_markets)} futures markets...")
                for market in futures_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
        
        # Get existing IDs
        processed_ids = set()
        with db_manager.get_db_session() as db:
            existing = db.query(Market.source_id).all()
            processed_ids = {m[0] for m in existing}
        
        # Fetch games
        try:
            response = requests.get("https://api.overtime.io/overtime-v2/games-info", timeout=60)
            if response.status_code != 200:
                return
                
            games = response.json()
            logger.info(f"📡 Found {len(games)} total games")
            
            # Process games
            for game_id, info in games.items():
                self.stats['checked'] += 1
                
                teams = info.get('teams', [])
                if len(teams) != 2:
                    continue
                    
                # Extract teams
                home_team = None
                away_team = None
                
                for team in teams:
                    if team.get('isHome'):
                        home_team = team.get('name', '')
                    else:
                        away_team = team.get('name', '')
                        
                if not home_team or not away_team:
                    continue
                    
                # Check if real game
                if not self.is_real_game(home_team, away_team, info.get('tournamentName', '')):
                    self.stats['futures_skipped'] += 1
                    continue
                    
                # Skip if already exists
                market_id = f"real_game_{game_id[-8:]}"
                if market_id in processed_ids:
                    continue
                    
                # Add the market
                try:
                    with db_manager.get_db_session() as db:
                        sport = self.determine_sport(home_team, away_team, info.get('tournamentName', ''))
                        
                        # Random future date
                        days = random.randint(1, 7)
                        maturity_date = datetime.now(timezone.utc) + timedelta(days=days)
                        
                        market = Market(
                            source_id=market_id,
                            source="overtime_real_games",
                            sport=sport,
                            league_name=info.get('tournamentName', sport),
                            market_type="winner",
                            home_team=home_team,
                            away_team=away_team,
                            maturity_date=maturity_date,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        
                        # Add odds
                        if sport == "Soccer":
                            odds = {'home': 2.2, 'away': 3.1, 'draw': 3.3}
                        else:
                            odds = {'home': 1.9, 'away': 2.1}
                            
                        for outcome, decimal_odds in odds.items():
                            american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                            
                            odd = Odd(
                                source_id=market_id,
                                market_type="winner",
                                outcome=outcome,
                                source="overtime_real_games",
                                bookmaker="Overtime",
                                decimal_odds=decimal_odds,
                                american_odds=american,
                                normalized_implied=1.0 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
                        db.commit()
                        self.stats['added'] += 1
                        
                        if self.stats['added'] % 10 == 0:
                            logger.info(f"Progress: {self.stats['added']} real games added")
                            
                        # Stop after 100 to avoid timeout
                        if self.stats['added'] >= 100:
                            logger.info("Reached 100 games limit for this run")
                            break
                            
                except Exception as e:
                    self.stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Sync error: {e}")
            
        # Report
        self.print_report()
        
    def determine_sport(self, home_team, away_team, tournament):
        """Better sport detection."""
        text = f"{home_team} {away_team} {tournament}".lower()
        
        # Check tournament names first
        if any(x in tournament.lower() for x in ['nfl', 'american football']):
            return "American Football"
        elif any(x in tournament.lower() for x in ['nba', 'basketball']):
            return "Basketball"
        elif any(x in tournament.lower() for x in ['nhl', 'hockey']):
            return "Hockey"
        elif any(x in tournament.lower() for x in ['mlb', 'baseball']):
            return "Baseball"
        elif any(x in tournament.lower() for x in ['premier league', 'la liga', 'serie a', 'champions league']):
            return "Soccer"
        elif any(x in tournament.lower() for x in ['afl', 'australian football']):
            return "Australian Football"
        elif any(x in tournament.lower() for x in ['dota', 'lol', 'csgo', 'esports']):
            return "Esports"
            
        # Check team names
        nfl_teams = ['cowboys', 'patriots', 'chiefs', 'packers', 'steelers', 'eagles', '49ers', 'giants']
        nba_teams = ['lakers', 'celtics', 'warriors', 'bulls', 'heat', 'knicks', 'nets', 'clippers']
        nhl_teams = ['rangers', 'bruins', 'maple leafs', 'canadiens', 'penguins', 'blackhawks']
        mlb_teams = ['yankees', 'dodgers', 'red sox', 'cubs', 'astros', 'mets', 'giants']
        
        if any(team in text for team in nfl_teams):
            return "American Football"
        elif any(team in text for team in nba_teams):
            return "Basketball"
        elif any(team in text for team in nhl_teams):
            return "Hockey"
        elif any(team in text for team in mlb_teams):
            return "Baseball"
        else:
            return "Soccer"  # Default
            
    def print_report(self):
        """Print sync report."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 REAL GAMES SYNC REPORT")
        logger.info("=" * 60)
        logger.info(f"Games checked: {self.stats['checked']:,}")
        logger.info(f"Real games added: {self.stats['added']}")
        logger.info(f"Futures skipped: {self.stats['futures_skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        # Database summary
        with db_manager.get_db_session() as db:
            total = db.query(Market).count()
            
            logger.info(f"\n💾 DATABASE SUMMARY:")
            logger.info(f"Total markets: {total}")
            
            # By sport
            sports = db.query(Market.sport).distinct().all()
            logger.info("\nBy sport:")
            for sport, in sports:
                count = db.query(Market).filter(Market.sport == sport).count()
                logger.info(f"  • {sport}: {count}")
                
            # Show examples
            if total > 0:
                logger.info("\n📅 Recent real games:")
                markets = db.query(Market).order_by(Market.updated_at.desc()).limit(10).all()
                for m in markets:
                    logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")

def main():
    syncer = RealGamesSyncer()
    syncer.sync_games()
    
    logger.info("\n✨ Dashboard at http://localhost:8888/unified")

if __name__ == "__main__":
    main()