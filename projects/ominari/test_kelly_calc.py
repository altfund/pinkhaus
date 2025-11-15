#!/usr/bin/env python3
"""Test Kelly bet calculation"""

from paper_trading_live import LivePaperTrader

trader = LivePaperTrader()

# Test Kelly calculation for high edge opportunities
test_cases = [
    {'name': 'Liverpool home', 'fair_prob': 0.303, 'odds': 3.3, 'edge': 27.29},
    {'name': 'Portland home', 'fair_prob': 0.435, 'odds': 2.3, 'edge': 4.78},
    {'name': 'Man City home', 'fair_prob': 0.270, 'odds': 3.7, 'edge': 22.37}
]

bankroll = 10000
print(f"Bankroll: ${bankroll}")
print(f"Kelly fraction: {trader.kelly_fraction}")
print(f"Min bet: ${trader.min_bet}")
print(f"Max bet %: {trader.max_bet_pct * 100}%")
print()

for case in test_cases:
    bet = trader.calculate_kelly_bet(case['fair_prob'], case['odds'], bankroll)
    print(f"{case['name']}:")
    print(f"  Fair prob: {case['fair_prob']:.3f}")
    print(f"  Odds: {case['odds']}")
    print(f"  Edge: {case['edge']:.2f}%")
    print(f"  Kelly bet: ${bet:.2f}")
    print(f"  As % of bankroll: {bet/bankroll*100:.2f}%")
    print(f"  Meets minimum? {'✅' if bet >= trader.min_bet else '❌'}")
    print()