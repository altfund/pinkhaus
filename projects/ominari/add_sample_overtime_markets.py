#!/usr/bin/env python3
"""
Add some sample Overtime markets to demonstrate the system works
These are real contract structures but with example data
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample teams for different sports
SAMPLE_TEAMS = {
    'Soccer': [
        ('Manchester United', 'Chelsea'),
        ('Real Madrid', 'Barcelona'),
        ('Bayern Munich', 'Borussia Dortmund'),
        ('Liverpool', 'Manchester City'),
        ('PSG', 'Lyon'),
        ('Juventus', 'AC Milan'),
        ('Arsenal', 'Tottenham'),
        ('Inter Milan', 'AS Roma')
    ],
    'Basketball': [
        ('Lakers', 'Celtics'),
        ('Warriors', 'Suns'),
        ('Nets', 'Knicks'),
        ('Bulls', 'Pistons'),
        ('Heat', 'Magic'),
        ('Nuggets', 'Jazz')
    ],
    'American Football': [
        ('Patriots', 'Bills'),
        ('Cowboys', 'Eagles'),
        ('Packers', 'Bears'),
        ('49ers', 'Seahawks'),
        ('Chiefs', 'Raiders')
    ],
    'Baseball': [
        ('Yankees', 'Red Sox'),
        ('Dodgers', 'Giants'),
        ('Cubs', 'Cardinals'),
        ('Astros', 'Rangers')
    ]
}

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

def generate_market_address(chain, index):
    """Generate a realistic-looking market address."""
    # Real Overtime market addresses (40 chars total = 0x + 40 hex)
    prefix = '0x1b06' if chain == 'optimism' else '0x9b75'
    # Need 36 more hex chars after the 4-char prefix (excluding 0x)
    suffix = f"{index:08x}{random.randint(0, 0xffffffff):08x}"
    remaining = ''.join(random.choices('0123456789abcdef', k=20))
    return f"{prefix}{suffix}{remaining}"

def main():
    """Main function."""
    logger.info("🎯 Adding Sample Overtime Markets")
    logger.info("=" * 60)
    logger.info("These are example markets to demonstrate the system")
    
    markets_added = 0
    
    try:
        with db_manager.get_db_session() as db:
            # Add markets for each sport
            for sport, teams_list in SAMPLE_TEAMS.items():
                for i, (home, away) in enumerate(teams_list[:5]):  # 5 markets per sport
                    # Vary the maturity dates
                    days_ahead = random.randint(1, 14)
                    hours = random.randint(0, 23)
                    maturity = datetime.now(timezone.utc) + timedelta(days=days_ahead, hours=hours)
                    
                    # Alternate between chains
                    chain = 'optimism' if i % 2 == 0 else 'arbitrum'
                    market_addr = generate_market_address(chain, markets_added)
                    
                    # Shorten market_id to fit in 66 chars
                    # Use only last 8 chars of address to keep it unique but short
                    market_id = f"{chain[:3]}_sample_{market_addr[-8:].lower()}"
                    
                    # Check if already exists
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{chain}_sample",
                        sport=sport,
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=home,
                        away_team=away,
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.flush()  # Flush to get the market ID
                    
                    # Add some sample odds
                    # Home win odds
                    home_decimal = 1.8 + random.random()  # 1.8 - 2.8
                    home_odd = Odd(
                        source_id=market.source_id,
                        market_type="winner",
                        outcome="home",
                        source="overtime_sample",
                        bookmaker="Overtime",
                        decimal_odds=home_decimal,
                        american_odds=decimal_to_american(home_decimal),
                        normalized_implied=1.0 / home_decimal,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(home_odd)
                    
                    # Away win odds
                    away_decimal = 1.9 + random.random()  # 1.9 - 2.9
                    away_odd = Odd(
                        source_id=market.source_id,
                        market_type="winner",
                        outcome="away",
                        source="overtime_sample",
                        bookmaker="Overtime",
                        decimal_odds=away_decimal,
                        american_odds=decimal_to_american(away_decimal),
                        normalized_implied=1.0 / away_decimal,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(away_odd)
                    
                    # Draw odds for soccer
                    if sport == 'Soccer':
                        draw_decimal = 3.0 + random.random()  # 3.0 - 4.0
                        draw_odd = Odd(
                            source_id=market.source_id,
                            market_type="winner",
                            outcome="draw",
                            source="overtime_sample",
                            bookmaker="Overtime",
                            decimal_odds=draw_decimal,
                            american_odds=decimal_to_american(draw_decimal),
                            normalized_implied=1.0 / draw_decimal,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(draw_odd)
                        
                    markets_added += 1
                    logger.info(f"✅ Added: {home} vs {away} ({sport})")
                    logger.info(f"   Chain: {chain.capitalize()}")
                    logger.info(f"   Maturity: {maturity}")
                    
            db.commit()
            
            # Summary
            total = db.query(Market).count()
            active = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).count()
            
            logger.info(f"\n✨ SAMPLE DATA ADDED ✨")
            logger.info(f"Markets added: {markets_added}")
            logger.info(f"Total markets in database: {total}")
            logger.info(f"Active future markets: {active}")
            
            if markets_added > 0:
                logger.info("\n🎆 SUCCESS! Sample markets have been added.")
                logger.info("The dashboard at http://localhost:8888/unified should now display data.")
                logger.info("\n⚠️  Note: These are SAMPLE markets for demonstration purposes.")
                logger.info("To get real blockchain data, we need:")
                logger.info("  1. Active Overtime markets on chain (may be seasonal)")
                logger.info("  2. Correct contract addresses and ABIs")
                logger.info("  3. Recent trading activity to discover markets")
                
    except Exception as e:
        logger.error(f"Error adding sample markets: {e}")

if __name__ == "__main__":
    main()