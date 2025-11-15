#!/usr/bin/env python3
"""
Generate more realistic odds for testing
This simulates real market conditions with varying odds and edges
"""

import os
import sys
import random
import logging
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_odds(market):
    """Generate realistic odds based on team matchup"""
    
    # Simulate team strength (could be based on real data)
    team_strengths = {
        'Real Madrid': 0.85,
        'Barcelona': 0.82,
        'Bayern Munich': 0.80,
        'Manchester City': 0.83,
        'Liverpool': 0.78,
        'Chelsea': 0.75,
        'Arsenal': 0.73,
        'Manchester United': 0.70,
        'Valencia': 0.65,
        'Sevilla': 0.68,
        'Atletico Madrid': 0.77,
        'Borussia Dortmund': 0.72,
        'RB Leipzig': 0.70,
        'Inter Milan': 0.74,
        'AC Milan': 0.71,
        'Juventus': 0.76,
        'Napoli': 0.73,
        'Roma': 0.69,
        'PSG': 0.81,
        'Marseille': 0.66,
        'Lyon': 0.64,
    }
    
    # Get team strengths or use random if unknown
    home_strength = team_strengths.get(market.home_team, random.uniform(0.4, 0.7))
    away_strength = team_strengths.get(market.away_team, random.uniform(0.4, 0.7))
    
    # Add home advantage
    home_strength += 0.05
    
    # Calculate base probabilities
    total_strength = home_strength + away_strength
    home_prob_base = home_strength / total_strength
    away_prob_base = away_strength / total_strength
    
    # Add some randomness
    home_prob = max(0.15, min(0.80, home_prob_base + random.uniform(-0.1, 0.1)))
    away_prob = max(0.15, min(0.80, away_prob_base + random.uniform(-0.1, 0.1)))
    
    # Draw probability (more likely when teams are evenly matched)
    strength_diff = abs(home_strength - away_strength)
    draw_prob_base = 0.25 * (1 - strength_diff)
    draw_prob = max(0.15, min(0.40, draw_prob_base + random.uniform(-0.05, 0.05)))
    
    # Normalize probabilities
    total_prob = home_prob + away_prob + draw_prob
    home_prob /= total_prob
    away_prob /= total_prob
    draw_prob /= total_prob
    
    # Add bookmaker margin (varies by bookmaker and market)
    margin = random.uniform(0.03, 0.10)  # 3-10% margin
    
    # Apply margin proportionally
    home_prob_with_margin = home_prob * (1 + margin)
    away_prob_with_margin = away_prob * (1 + margin)
    draw_prob_with_margin = draw_prob * (1 + margin)
    
    # Convert to decimal odds
    home_odds = 1 / home_prob_with_margin
    away_odds = 1 / away_prob_with_margin
    draw_odds = 1 / draw_prob_with_margin
    
    # Round to realistic values
    home_odds = round(home_odds, 2)
    away_odds = round(away_odds, 2)
    draw_odds = round(draw_odds, 2)
    
    # Occasionally create positive edge opportunities (arbitrage or mispricing)
    if random.random() < 0.1:  # 10% chance of positive edge
        # Randomly boost one outcome's odds
        outcome = random.choice(['home', 'away', 'draw'])
        boost = random.uniform(1.05, 1.15)  # 5-15% boost
        
        if outcome == 'home':
            home_odds *= boost
        elif outcome == 'away':
            away_odds *= boost
        else:
            draw_odds *= boost
            
        logger.info(f"Created positive edge opportunity on {outcome} for {market.home_team} vs {market.away_team}")
    
    return {
        'home': home_odds,
        'away': away_odds,
        'draw': draw_odds
    }

def update_odds_with_realistic_values():
    """Update all active markets with realistic odds"""
    
    with db_manager.get_db_session() as db:
        # Get active markets
        markets = db.query(Market).filter(
            Market.maturity_date > datetime.now(timezone.utc)
        ).all()
        
        logger.info(f"Updating odds for {len(markets)} active markets")
        
        updated_count = 0
        positive_edge_count = 0
        
        for market in markets:
            odds = generate_realistic_odds(market)
            
            # Update each outcome
            for outcome, decimal_odds in odds.items():
                # Check if odd exists
                existing = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == outcome
                ).first()
                
                if existing:
                    # Update odds
                    existing.decimal_odds = decimal_odds
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new odd
                    new_odd = Odd(
                        source_id=market.source_id,
                        outcome=outcome,
                        decimal_odds=decimal_odds,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(new_odd)
                
                updated_count += 1
            
            # Check if any positive edges were created
            total_prob = sum(1/o for o in odds.values())
            if total_prob < 1.0:
                positive_edge_count += 1
                logger.info(f"Arbitrage opportunity: {market.home_team} vs {market.away_team}, total prob: {total_prob:.3f}")
            
        db.commit()
        
        logger.info(f"Updated {updated_count} odds")
        logger.info(f"Created {positive_edge_count} positive edge opportunities")
        
        # Show sample of new odds variety
        sample_odds = db.query(Odd).join(Market).filter(
            Market.maturity_date > datetime.now(timezone.utc)
        ).order_by(Odd.updated_at.desc()).limit(20).all()
        
        unique_odds = set()
        logger.info("\nSample of updated odds:")
        for odd in sample_odds:
            market = db.query(Market).filter(Market.source_id == odd.source_id).first()
            if market:
                logger.info(f"{market.home_team} vs {market.away_team}: {odd.outcome} @ {odd.decimal_odds}")
                unique_odds.add(odd.decimal_odds)
                
        logger.info(f"\nOdds variety: {len(unique_odds)} unique values in sample")

if __name__ == "__main__":
    logger.info("Generating realistic odds for all markets...")
    update_odds_with_realistic_values()
    logger.info("Realistic odds generation complete!")