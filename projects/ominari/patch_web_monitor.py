#!/usr/bin/env python3
"""Patch web monitor to use database for closed positions."""

import re

# Read the file
with open('web_monitor.py', 'r') as f:
    content = f.read()

# Find the function to replace
start_pattern = r'@app\.route\(\'/api/trading/closed_positions\'\)\ndef api_trading_closed_positions\(\):'
end_pattern = r'@app\.route\(\'/api/trading/execute\''

# New function implementation
new_function = '''@app.route('/api/trading/closed_positions')
def api_trading_closed_positions():
    """Get closed positions from paper trading session."""
    try:
        limit = request.args.get('limit', 20, type=int)
        closed_positions = []
        summary = {
            'total_count': 0,
            'total_pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'total_execution_stake': 0.0
        }
        
        # Get active session from database
        with db_manager.get_db_session() as db:
            session = db.query(PaperTradingSession).filter_by(
                status=SessionStatus.ACTIVE
            ).order_by(PaperTradingSession.created_at.desc()).first()
            
            if not session:
                return jsonify({'positions': [], 'summary': summary, 'message': 'No active session'})
            
            session_id = session.session_id
            
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
            
            # Get recent positions with market names (order by closed_at desc)
            recent_query = db.query(
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
            ).limit(limit)
            
            recent_positions = recent_query.all()
            
            # Get market data from main database
            market_ids = [pos[0].market_id for pos in recent_positions]
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
            for pos, market_name in recent_positions:
                # Get market info
                mkt = market_data.get(pos.market_id, {})
                
                # Format closed time
                formatted_time = pos.closed_at.strftime('%Y-%m-%d %H:%M') if pos.closed_at else 'Unknown'
                
                # Get score info
                score = '-'
                if mkt.get('is_finished'):
                    score = f"{mkt.get('home_score', '?')}-{mkt.get('away_score', '?')}"
                
                # Get match time
                match_time = '-'
                if mkt.get('maturity_date'):
                    match_time = mkt['maturity_date'].strftime('%m/%d %H:%M')
                
                # Calculate P&L percentage
                pnl = float(pos.pnl) if pos.pnl else 0
                execution_stake = float(pos.execution_stake)
                pnl_pct = (pnl / execution_stake * 100) if execution_stake > 0 else 0
                
                closed_positions.append({
                    'closed_time': formatted_time,
                    'market_name': market_name.market_name,
                    'outcome': pos.outcome.value,
                    'stake': float(pos.stake),
                    'odds': float(pos.avg_odds),
                    'result': pos.result.value if pos.result else 'pending',
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'final_outcome': pos.resolved_outcome.value if pos.resolved_outcome else mkt.get('resolved_outcome', 'TBD'),
                    'score': score,
                    'match_time': match_time,
                    'fee_info': {
                        'safebox_fee_pct': pos.safebox_fee_pct,
                        'skew_fee_pct': pos.skew_fee_pct,
                        'total_fee_pct': pos.total_fee_pct
                    },
                    'execution_stake': execution_stake,
                    'home_team': market_name.home_team,
                    'away_team': market_name.away_team
                })
        
        return jsonify({
            'positions': closed_positions,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting closed positions: {e}")
        return jsonify({'positions': [], 'summary': summary})

'''

# Find where the function starts and ends
start_match = re.search(start_pattern, content, re.MULTILINE)
if not start_match:
    print("Could not find function start")
    exit(1)

end_match = re.search(end_pattern, content[start_match.start():], re.MULTILINE)
if not end_match:
    print("Could not find function end")
    exit(1)

# Calculate positions
func_start = start_match.start()
func_end = start_match.start() + end_match.start()

# Replace the function
new_content = content[:func_start] + new_function + content[func_end:]

# Write back
with open('web_monitor.py', 'w') as f:
    f.write(new_content)

print("Successfully patched web_monitor.py")