#!/usr/bin/env python3
"""Check the user's claim about $169 lost on $5648."""

from paper_trading_sessions import PaperTradingSessionManager

sm = PaperTradingSessionManager()
session = sm.get_current_session()

if session:
    print("CHECKING USER'S CLAIM")
    print("=" * 80)
    print("User said: '$169 lost on $5648 and $76 currently at risk'")
    print()
    
    # Calculate totals
    closed_positions = session.get('closed_positions', [])
    open_positions = session.get('positions', {})
    
    # Calculate total stakes and execution stakes
    total_stake_closed = sum(p.get('total_stake', 0) for p in closed_positions)
    total_exec_stake_closed = sum(p.get('execution_stake', p.get('total_stake', 0)) 
                                  for p in closed_positions)
    
    # Calculate P&L
    total_pnl = sum(p.get('pnl', 0) for p in closed_positions)
    wins = [p for p in closed_positions if p.get('result') == 'won']
    losses = [p for p in closed_positions if p.get('result') == 'lost']
    
    # Open positions
    open_stake = sum(p.get('total_stake', 0) for p in open_positions.values())
    open_exec_stake = sum(p.get('execution_stake', p.get('total_stake', 0)) 
                          for p in open_positions.values())
    
    print("CLOSED POSITIONS:")
    print(f"  Total positions: {len(closed_positions)}")
    print(f"  Wins: {len(wins)}")
    print(f"  Losses: {len(losses)}")
    print(f"  Total stake: ${total_stake_closed:,.2f}")
    print(f"  Total execution stake: ${total_exec_stake_closed:,.2f}")
    print(f"  Total P&L: ${total_pnl:,.2f}")
    
    print("\nOPEN POSITIONS:")
    print(f"  Total positions: {len(open_positions)}")
    print(f"  Total stake: ${open_stake:,.2f}")
    print(f"  Total execution stake: ${open_exec_stake:,.2f}")
    
    print("\nPOSSIBLE INTERPRETATIONS:")
    print("-" * 60)
    
    # Maybe user is looking at different numbers?
    print(f"1. Total closed stake: ${total_stake_closed:,.2f}")
    print(f"   Total closed exec stake: ${total_exec_stake_closed:,.2f}")
    print(f"   If user said '$5648', closest is total closed exec stake")
    
    # Check if $169 relates to something
    print(f"\n2. Checking for ~$169 values:")
    print(f"   Total fees on closed: ${total_exec_stake_closed - total_stake_closed:,.2f}")
    
    # Check portfolio
    portfolio_value = session.get('portfolio_value', 0)
    initial = session.get('initial_bankroll', 10000)
    print(f"\n3. Portfolio change: ${portfolio_value - initial:,.2f}")
    
    # Maybe it's about win amounts?
    total_won = sum(p.get('pnl', 0) for p in wins)
    total_lost = sum(p.get('pnl', 0) for p in losses)
    print(f"\n4. Win/Loss breakdown:")
    print(f"   Total won: ${total_won:,.2f}")
    print(f"   Total lost: ${total_lost:,.2f}")
    
    # Check if user meant something about returns
    gross_returns = sum(p.get('final_value', 0) for p in wins)
    print(f"\n5. Gross returns from wins: ${gross_returns:,.2f}")
    
    # Net vs gross
    print(f"\n6. If we calculate differently:")
    net_from_wins = gross_returns - sum(p.get('execution_stake', 0) for p in wins)
    print(f"   Net from wins: ${net_from_wins:,.2f}")
    money_lost_on_losses = sum(p.get('execution_stake', 0) for p in losses)
    print(f"   Money lost on losses: ${money_lost_on_losses:,.2f}")
    print(f"   Difference: ${net_from_wins - money_lost_on_losses:,.2f}")