#!/usr/bin/env python3
"""
Simple test to add a few Overtime markets for dashboard testing
Using known patterns from transaction analysis
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_test_markets():
    """Add some test markets based on Overtime patterns."""
    logger.info("Adding test Overtime markets...")
    
    markets_added = 0
    
    # Real upcoming matches that would be on Overtime
    test_markets = [
        # Premier League
        {
            'home': 'Manchester United',
            'away': 'Liverpool',
            'league': 'Premier League',
            'sport': 'Soccer',
            'days_ahead': 1,
            'network': 'optimism'
        },
        {
            'home': 'Chelsea',
            'away': 'Arsenal',
            'league': 'Premier League', 
            'sport': 'Soccer',
            'days_ahead': 2,
            'network': 'optimism'
        },
        # NFL
        {
            'home': 'Kansas City Chiefs',
            'away': 'Buffalo Bills',
            'league': 'NFL',
            'sport': 'American Football',
            'days_ahead': 1,
            'network': 'arbitrum'
        },
        {
            'home': 'Dallas Cowboys',
            'away': 'Philadelphia Eagles',
            'league': 'NFL',
            'sport': 'American Football',
            'days_ahead': 3,
            'network': 'arbitrum'
        },
        # NBA
        {
            'home': 'Los Angeles Lakers',
            'away': 'Golden State Warriors',
            'league': 'NBA',
            'sport': 'Basketball',
            'days_ahead': 1,
            'network': 'optimism'
        },
        # La Liga
        {
            'home': 'Real Madrid',
            'away': 'Barcelona',
            'league': 'La Liga',
            'sport': 'Soccer',
            'days_ahead': 2,
            'network': 'optimism'
        },
        # Serie A
        {
            'home': 'AC Milan',
            'away': 'Inter Milan',
            'league': 'Serie A',
            'sport': 'Soccer',
            'days_ahead': 3,
            'network': 'arbitrum'
        },
        # Bundesliga
        {
            'home': 'Bayern Munich',
            'away': 'Borussia Dortmund',
            'league': 'Bundesliga',
            'sport': 'Soccer',
            'days_ahead': 2,
            'network': 'optimism'
        }
    ]
    
    with db_manager.get_db_session() as db:
        for i, market_data in enumerate(test_markets):
            try:
                # Generate a realistic market address
                market_addr = f"0x{i:040x}"[0:42]
                market_id = f"blockchain_{market_data['network']}_test_{market_addr}"
                
                # Check if exists
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # Create market
                maturity = datetime.now(timezone.utc) + timedelta(days=market_data['days_ahead'])
                
                market = Market(
                    source_id=market_id,
                    source=f"blockchain_{market_data['network']}_test",
                    sport=market_data['sport'],
                    league_name=market_data['league'],
                    market_type="winner",
                    home_team=market_data['home'],
                    away_team=market_data['away'],
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                db.commit()
                
                # Add realistic odds
                if market_data['sport'] == 'Soccer':
                    # Soccer has draws
                    odds_data = [
                        ('Home', 2.20),
                        ('Draw', 3.30),
                        ('Away', 3.10)
                    ]
                else:
                    # No draw
                    odds_data = [
                        ('Home', 1.85),
                        ('Away', 2.10)
                    ]
                    
                for outcome, decimal_odds in odds_data:
                    odd = Odd(
                        source_id=market_id,
                        outcome=outcome,
                        decimal_odds=decimal_odds,
                        market_type='moneyline',
                        source=f"blockchain_{market_data['network']}_test",
                        bookmaker='overtime',
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(odd)
                    
                db.commit()
                markets_added += 1
                logger.info(f"✅ Added: {market_data['home']} vs {market_data['away']} ({market_data['sport']})")
                
            except Exception as e:
                logger.error(f"Error adding market: {e}")
                continue
                
    return markets_added

def main():
    """Main function."""
    logger.info("🎮 Overtime Test Markets")
    logger.info("=" * 60)
    logger.info("Adding test markets for dashboard functionality")
    
    # Check current state
    with db_manager.get_db_session() as db:
        before = db.query(Market).count()
        logger.info(f"Starting with {before} markets")
    
    # Add test markets
    added = add_test_markets()
    
    # Summary
    with db_manager.get_db_session() as db:
        after = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ TEST MARKETS ADDED ✨")
        logger.info(f"Markets: {before} → {after} (+{added})")
        logger.info(f"Active future markets: {active}")
        
        # Show all markets
        all_markets = db.query(Market).all()
        if all_markets:
            logger.info(f"\n📊 All markets ({len(all_markets)}):")
            for m in all_markets:
                odds = db.query(Odd).filter(Odd.source_id == m.source_id).all()
                odds_str = ", ".join([f"{o.outcome}: {o.decimal_odds}" for o in odds])
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.league_name} - {m.maturity_date}")
                logger.info(f"    Odds: {odds_str}")
                logger.info(f"    Source: {m.source}")

if __name__ == "__main__":
    main()