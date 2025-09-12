#!/usr/bin/env python3
"""Show results for specific matches mentioned by user."""

from paper_trading_sessions import PaperTradingSessionManager

session_manager = PaperTradingSessionManager()
current_session = session_manager.get_current_session()

if current_session:
    closed = current_session.get('closed_positions', [])
    
    # Find specific matches
    matches_to_find = [
        'Kosovo_vs_Sweden',
        'Greece_vs_Denmark', 
        'Ghana_vs_Mali',
        'Libya_vs_Eswatini',
        'Vila Nova FC_vs_Athletic Club',
        'Grêmio Novorizontino_vs_Atlético Goianiense'
    ]
    
    print('SPECIFIC MATCHES STATUS:')
    print('=' * 80)
    
    for match in matches_to_find:
        positions = [p for p in closed if match in p.get('market_name', '')]
        if positions:
            print(f'\n{match}:')
            
            # Get market result from first position
            market_result = positions[0].get('market_result', {})
            score = f"{market_result.get('home_score', '?')}-{market_result.get('away_score', '?')}"
            final_result = market_result.get('resolved_outcome', '?')
            
            print(f'  Final Score: {score} ({final_result})')
            print(f'  Positions:')
            
            for pos in positions:
                outcome = pos["outcome"]
                result = pos.get("result", "?")
                pnl = pos.get("pnl", 0)
                stake = pos.get("execution_stake", pos.get("total_stake", 0))
                
                result_symbol = "✓" if result == "won" else "✗" if result == "lost" else "?"
                pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}" if pnl < 0 else "$0"
                
                print(f'    {outcome:6} -> {result_symbol} {result:6} (Stake: ${stake:.2f}, P&L: {pnl_str})')