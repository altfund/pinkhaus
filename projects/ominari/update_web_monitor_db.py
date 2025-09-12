#!/usr/bin/env python3
"""Update web monitor to use database-backed paper trading."""

import json
from datetime import datetime, timezone
from flask import jsonify
from database_v2 import db_manager
from paper_trading_models_v2 import (
    PaperTradingSession, PaperTradingPosition, PaperTradingSnapshot,
    MarketName, PositionStatus, SessionStatus, Result
)
from sqlalchemy import func, and_
from models import Market

def get_active_session_id():
    """Get the active session ID."""
    with db_manager.get_db_session() as db:
        session = db.query(PaperTradingSession).filter_by(
            status=SessionStatus.ACTIVE
        ).order_by(PaperTradingSession.created_at.desc()).first()
        
        return session.session_id if session else None

def api_trading_closed_positions_db(limit=20):
    """Get closed positions from database."""
    try:
        closed_positions = []
        summary = {
            'total_count': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_execution_stake': 0.0
        }
        
        session_id = get_active_session_id()
        if not session_id:
            return jsonify({'positions': [], 'summary': summary, 'message': 'No active session'})
        
        with db_manager.get_db_session() as db:
            # Get all closed positions for summary
            all_closed = db.query(PaperTradingPosition).filter_by(
                session_id=session_id,
                status=PositionStatus.CLOSED
            ).all()
            
            summary['total_count'] = len(all_closed)
            
            # Calculate summary statistics
            for pos in all_closed:
                pnl = float(pos.pnl) if pos.pnl else 0
                summary['total_pnl'] += pnl
                summary['total_execution_stake'] += float(pos.execution_stake)
                
                if pos.result == Result.WON:
                    summary['wins'] += 1
                elif pos.result == Result.LOST:
                    summary['losses'] += 1
            
            # Calculate win rate
            total_settled = summary['wins'] + summary['losses']
            if total_settled > 0:
                summary['win_rate'] = (summary['wins'] / total_settled) * 100
            
            # Get recent positions with market names
            recent_closed = db.query(
                PaperTradingPosition,
                MarketName
            ).join(
                MarketName,
                PaperTradingPosition.market_id == MarketName.market_id
            ).filter(
                PaperTradingPosition.session_id == session_id,
                PaperTradingPosition.status == PositionStatus.CLOSED
            ).order_by(
                PaperTradingPosition.closed_at.desc()
            ).limit(limit).all()
            
            # Get market data from main database
            market_ids = [pos[0].market_id for pos in recent_closed]
            market_data = {}
            
            if market_ids:
                markets = db.query(Market).filter(Market.source_id.in_(market_ids)).all()
                for market in markets:
                    market_data[market.source_id] = {
                        'maturity_date': market.maturity_date,
                        'is_finished': market.is_finished,
                        'resolved_outcome': market.resolved_outcome,
                        'home_score': market.home_score,
                        'away_score': market.away_score,
                        'home_team': market.home_team,
                        'away_team': market.away_team
                    }
            
            # Format for display
            for pos, market_name in recent_closed:
                # Get market info
                mkt = market_data.get(pos.market_id, {})
                
                # Format closed time
                formatted_time = pos.closed_at.strftime('%Y-%m-%d %H:%M') if pos.closed_at else ''
                
                # Build position data
                position_data = {
                    'market_name': market_name.market_name,
                    'outcome': pos.outcome.value,
                    'stake': float(pos.stake),
                    'execution_stake': float(pos.execution_stake),
                    'odds': float(pos.avg_odds),
                    'pnl': float(pos.pnl) if pos.pnl else 0,
                    'result': pos.result.value if pos.result else 'pending',
                    'closed_at': formatted_time,
                    'market_id': pos.market_id,
                    'home_team': market_name.home_team,
                    'away_team': market_name.away_team,
                    'home_score': mkt.get('home_score'),
                    'away_score': mkt.get('away_score'),
                    'resolved_outcome': mkt.get('resolved_outcome'),
                    'total_fee_pct': pos.total_fee_pct
                }
                
                closed_positions.append(position_data)
        
        return jsonify({
            'positions': closed_positions,
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error in api_trading_closed_positions_db: {e}")
        return jsonify({
            'error': str(e),
            'positions': [],
            'summary': {
                'total_count': 0,
                'total_pnl': 0.0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0
            }
        })

# Test function
if __name__ == "__main__":
    result = api_trading_closed_positions_db()
    print(json.dumps(result.json, indent=2))