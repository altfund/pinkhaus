#!/usr/bin/env python3
"""Fix the get_soccer_markets function to use ORM."""

def get_soccer_markets_safe():
    """Get soccer markets using safe ORM approach."""
    from database_v2 import db_manager
    from models import Market, Odd
    from datetime import datetime, timezone
    from sqlalchemy import desc
    
    markets = []
    
    with db_manager.get_db_session() as db:
        # Get active soccer markets
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).order_by(Market.maturity_date).limit(50).all()
        
        for market in active_markets:
            # Get latest odds for this market
            home_odd = db.query(Odd).filter(
                Odd.source_id == market.source_id,
                Odd.outcome == market.home_team
            ).order_by(desc(Odd.updated_at)).first()
            
            away_odd = db.query(Odd).filter(
                Odd.source_id == market.source_id,
                Odd.outcome == market.away_team
            ).order_by(desc(Odd.updated_at)).first()
            
            draw_odd = db.query(Odd).filter(
                Odd.source_id == market.source_id,
                Odd.outcome == 'Draw'
            ).order_by(desc(Odd.updated_at)).first()
            
            kickoff = market.maturity_date
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            
            kickoff_str = kickoff.strftime('%H:%M') if kickoff.date() == datetime.now(timezone.utc).date() else kickoff.strftime('%m/%d %H:%M')
            
            market_data = {
                'id': market.source_id,
                'league': market.league_name or 'Soccer',
                'home': market.home_team,
                'away': market.away_team,
                'kickoff': kickoff_str,
                'home_odds': home_odd.decimal_odds if home_odd else 2.0,
                'away_odds': away_odd.decimal_odds if away_odd else 2.0,
                'draw_odds': draw_odd.decimal_odds if draw_odd else 3.0,
                'market_count': 1
            }
            
            markets.append(market_data)
    
    return markets

# Test it
if __name__ == "__main__":
    markets = get_soccer_markets_safe()
    print(f"Found {len(markets)} markets")
    for m in markets[:5]:
        print(f"- {m['home']} vs {m['away']} @ {m['kickoff']} (H:{m['home_odds']}, A:{m['away_odds']}, D:{m['draw_odds']})")