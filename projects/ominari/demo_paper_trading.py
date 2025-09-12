#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Paper Trading Script
Simulates some trades to demonstrate the trading system.
"""

import json
import pandas as pd
from datetime import datetime, timezone, timedelta
import random

def create_demo_trades():
    """Create some demo trades for testing."""
    # Create portfolio
    portfolio = {
        'cash': 8500.0,  # Started with 10k, placed 1500 in bets
        'positions': {},
        'total_value': 10000.0,
        'trades': 15,
        'wins': 9,
        'losses': 6,
        'pending_bets': {}
    }
    
    # Add some pending bets
    pending_matches = [
        {
            'match_id': 'match_001',
            'match': 'Liverpool vs Manchester United',
            'bet_on': 'Liverpool',
            'odds': 2.10,
            'stake': 100.0,
            'potential_return': 210.0,
            'signal_prob': 0.55,
            'market_prob': 0.476,
            'edge': 0.074,
            'confidence': 0.8,
            'kick_off': (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        },
        {
            'match_id': 'match_002',
            'match': 'Barcelona vs Real Madrid',
            'bet_on': 'Draw',
            'odds': 3.25,
            'stake': 75.0,
            'potential_return': 243.75,
            'signal_prob': 0.35,
            'market_prob': 0.308,
            'edge': 0.042,
            'confidence': 0.7,
            'kick_off': (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
        }
    ]
    
    for bet in pending_matches:
        portfolio['pending_bets'][bet['match_id']] = bet
    
    # Save portfolio
    with open('paper_portfolio.json', 'w') as f:
        json.dump(portfolio, f, indent=2)
    
    # Create trade history
    trades = []
    base_time = datetime.now(timezone.utc) - timedelta(days=7)
    
    matches = [
        ('Arsenal vs Chelsea', 'Arsenal', 'Chelsea'),
        ('Bayern Munich vs Dortmund', 'Bayern Munich', 'Draw'),
        ('Juventus vs AC Milan', 'Draw', 'AC Milan'),
        ('PSG vs Lyon', 'PSG', 'Lyon'),
        ('Atletico Madrid vs Sevilla', 'Atletico Madrid', 'Draw'),
        ('Inter Milan vs Roma', 'Inter Milan', 'Roma'),
        ('Manchester City vs Tottenham', 'Manchester City', 'Draw'),
        ('Ajax vs PSV', 'Ajax', 'PSV'),
        ('Porto vs Benfica', 'Draw', 'Benfica'),
        ('Celtic vs Rangers', 'Celtic', 'Rangers'),
        ('Valencia vs Villarreal', 'Valencia', 'Draw'),
        ('Napoli vs Lazio', 'Napoli', 'Lazio'),
        ('Bayer Leverkusen vs RB Leipzig', 'Draw', 'RB Leipzig'),
        ('Monaco vs Marseille', 'Monaco', 'Marseille'),
        ('Sporting CP vs Braga', 'Sporting CP', 'Draw')
    ]
    
    for i, (match, home_team, away_team) in enumerate(matches):
        timestamp = base_time + timedelta(hours=i*12)
        kick_off = timestamp - timedelta(hours=1)
        
        # Randomly choose who to bet on
        bet_options = [home_team, away_team]
        if 'Draw' in bet_options:
            bet_on = 'Draw'
            odds = random.uniform(3.0, 3.8)
        else:
            bet_on = random.choice(bet_options)
            odds = random.uniform(1.8, 2.8)
        
        stake = random.uniform(50, 150)
        signal_prob = random.uniform(0.35, 0.65)
        
        trades.append({
            'timestamp': timestamp.isoformat(),
            'match_id': f'hist_{i:03d}',
            'match': match,
            'bet_on': bet_on,
            'odds': round(odds, 2),
            'stake': round(stake, 2),
            'potential_return': round(stake * odds, 2),
            'signal_prob': round(signal_prob, 3),
            'market_prob': round(1/odds, 3),
            'edge': round(signal_prob - 1/odds, 3),
            'confidence': round(random.uniform(0.6, 0.9), 2),
            'kick_off': kick_off.isoformat()
        })
    
    # Save trades
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('paper_trades.csv', index=False)
    
    print("✅ Demo data created successfully!")
    print(f"   - Portfolio: {portfolio['wins']} wins, {portfolio['losses']} losses")
    print(f"   - Pending bets: {len(portfolio['pending_bets'])}")
    print(f"   - Historical trades: {len(trades)}")

if __name__ == "__main__":
    create_demo_trades()