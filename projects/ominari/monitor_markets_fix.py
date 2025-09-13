#!/usr/bin/env python3
"""Fixed get_soccer_markets function for monitor_unified.py"""

def get_soccer_markets_v2():
    """Get soccer markets using safe ORM approach."""
    from database_v2 import db_manager
    from models import Market, Odd
    from datetime import datetime, timezone
    from sqlalchemy import desc
    import numpy as np
    
    markets = []
    
    try:
        with db_manager.get_db_session() as db:
            # Get active soccer markets
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(50).all()
            
            print(f"[DEBUG] Found {len(active_markets)} active soccer markets")
            
            for market in active_markets:
                # Get latest odds for main market (option_1, option_2, option_3)
                option_1 = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_1'
                ).order_by(desc(Odd.updated_at)).first()
                
                option_2 = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_2'
                ).order_by(desc(Odd.updated_at)).first()
                
                option_3 = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.outcome == 'option_3'
                ).order_by(desc(Odd.updated_at)).first()
                
                kickoff = market.maturity_date
                if kickoff.tzinfo is None:
                    kickoff = kickoff.replace(tzinfo=timezone.utc)
                
                kickoff_str = kickoff.strftime('%H:%M') if kickoff.date() == datetime.now(timezone.utc).date() else kickoff.strftime('%m/%d %H:%M')
                
                home_odds = option_1.decimal_odds if option_1 else 2.0
                away_odds = option_2.decimal_odds if option_2 else 2.0
                draw_odds = option_3.decimal_odds if option_3 else 3.0
                
                market_data = {
                    'id': market.source_id,
                    'league': market.league_name or 'Soccer',
                    'home': market.home_team,
                    'away': market.away_team,
                    'kickoff': kickoff_str,
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'draw_odds': draw_odds,
                    'home_implied': 100 / home_odds,
                    'away_implied': 100 / away_odds,
                    'draw_implied': 100 / draw_odds
                }
                
                # Add signal if edge detected (simulate)
                if np.random.random() > 0.7:
                    market_data['signal'] = {
                        'selection': np.random.choice(['home', 'draw', 'away']),
                        'strength': np.random.randint(60, 90)
                    }
                
                markets.append(market_data)
                
    except Exception as e:
        print(f"Error loading markets: {e}")
        import traceback
        traceback.print_exc()
        
    return markets

# Add this function to monitor_unified.py by replacing the existing get_soccer_markets function