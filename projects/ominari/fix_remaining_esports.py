#!/usr/bin/env python3
"""
More deterministic fix for remaining esports misclassifications
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_remaining_esports():
    """Comprehensive esports detection"""
    
    # Esports indicators - if ANY of these appear, it's esports
    ESPORTS_INDICATORS = {
        'team_patterns': [
            'Gaming', 'Esports', 'Esport', 'eSports', 'Team Liquid', 'Fnatic', 
            'G2', 'Cloud9', 'TSM', 'FaZe', 'NaVi', 'Vitality', 'Astralis',
            'OG ', ' OG', 'Evil Geniuses', 'Alliance', 'Invictus', 'T1',
            'Gen.G', 'DWG', 'RNG', 'EDG', 'FPX', 'MAD Lions', 'Rogue',
            'Misfits', 'Excel', 'Dignitas', 'Immortals', '100 Thieves',
            'Sentinels', 'OpTic', 'LOUD', 'DRX', 'ZETA', 'NiP', 'BDS',
            'Heroic', 'MOUZ', 'BIG', 'Virtus.pro', 'Spirit', 'Outsiders',
            'FORZE', 'ONSIDE', 'Xtreme', 'PSG Talon', 'Weibo', 'Talon',
            'OMBRA', 'SIBE', 'Verso', 'Wolves', 'Tricked', 'WOPA'
        ],
        'league_patterns': [
            'MLBB', 'OCS', 'The International', 'TI1', 'TI2', 'League of Legends',
            'LoL', 'LCS', 'LEC', 'LPL', 'LCK', 'MSI', 'Worlds', 'VCT',
            'VALORANT', 'CS:GO', 'CSGO', 'CS2', 'Dota', 'DOTA', 
            'Pro League', 'European Pro League', 'Asia-Pacific League',
            'Continental Championship', 'BLAST', 'ESL', 'IEM', 'DreamHack',
            'PGL', 'FACEIT', 'WePlay', 'Beyond the Summit', 'BTS',
            'Rocket League', 'RLCS', 'Overwatch', 'OWL', 'CDL', 'CWL',
            'Apex Legends', 'ALGS', 'PUBG', 'PCS', 'PGC', 'Fortnite',
            'FNCS', 'Rainbow Six', 'R6', 'Halo', 'HCS', 'Gears',
            'Europe MENA League', 'Americas League', 'Pacific League'
        ],
        'game_titles': [
            'Counter-Strike', 'Dota 2', 'League of Legends', 'VALORANT',
            'Overwatch', 'Call of Duty', 'Rocket League', 'Rainbow Six',
            'Apex Legends', 'Fortnite', 'PUBG', 'Halo', 'StarCraft',
            'Warcraft', 'Hearthstone', 'FIFA', 'NBA 2K', 'Madden',
            'Street Fighter', 'Tekken', 'Mortal Kombat', 'Smash Bros'
        ]
    }
    
    with db_manager.get_db_session() as db:
        # Find all potential esports markets
        all_markets = db.query(Market).filter(
            Market.sport == 'Soccer'  # Currently misclassified
        ).all()
        
        esports_markets = []
        
        for market in all_markets:
            # Check each field for esports indicators
            home = (market.home_team or '').lower()
            away = (market.away_team or '').lower()
            league = (market.league_name or '').lower()
            
            is_esports = False
            
            # Check team patterns
            for pattern in ESPORTS_INDICATORS['team_patterns']:
                if pattern.lower() in home or pattern.lower() in away:
                    is_esports = True
                    break
            
            # Check league patterns
            if not is_esports:
                for pattern in ESPORTS_INDICATORS['league_patterns']:
                    if pattern.lower() in league:
                        is_esports = True
                        break
            
            # Check for game titles
            if not is_esports:
                combined = f"{home} {away} {league}"
                for game in ESPORTS_INDICATORS['game_titles']:
                    if game.lower() in combined:
                        is_esports = True
                        break
            
            # Additional patterns
            if not is_esports:
                # Check for typical esports formatting
                if ('.' in home and not home.endswith('.')) or ('.' in away and not away.endswith('.')):
                    # Teams like "G2.Esports" or "Team.Liquid"
                    is_esports = True
                elif any(char in home + away for char in ['_', '[', ']', '|']):
                    # Esports teams often use underscores or brackets
                    is_esports = True
                elif 'vs' not in market.home_team and 'v' not in market.home_team:
                    # Check if it's a proper matchup format
                    if ' ' not in market.home_team and ' ' not in market.away_team:
                        # Single word teams are often esports
                        if len(market.home_team) < 15 and len(market.away_team) < 15:
                            is_esports = True
            
            if is_esports:
                esports_markets.append(market)
                market.sport = 'Esports'
        
        if esports_markets:
            db.commit()
            logger.info(f"\n✅ Fixed {len(esports_markets)} esports markets")
            
            # Show examples
            logger.info("\nExamples of fixed esports markets:")
            for market in esports_markets[:10]:
                logger.info(f"  {market.home_team} vs {market.away_team} ({market.league_name})")
        else:
            logger.info("\n✅ No additional esports markets found")
        
        # Check what's still marked as soccer
        logger.info("\n=== Remaining Soccer Markets Sample ===")
        remaining = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.nation.in_(['International', 'Europe', 'England'])
        ).limit(20).all()
        
        for market in remaining:
            logger.info(f"  {market.home_team} vs {market.away_team} ({market.league_name}) - {market.nation}")

if __name__ == "__main__":
    fix_remaining_esports()