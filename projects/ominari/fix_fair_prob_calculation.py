#!/usr/bin/env python3
"""Fix fair probability calculation for Kelly betting"""

# Read the file
with open('paper_trading_live.py', 'r') as f:
    content = f.read()

# Fix the arbitrage fair probability calculation
old_code = """                        if total_prob < 1.0:
                            # Arbitrage opportunity - all outcomes have positive edge
                            fair_prob = implied_prob  # Use implied as fair
                            edge = ((1/total_prob) - 1) * 100  # Arbitrage edge"""

new_code = """                        if total_prob < 1.0:
                            # Arbitrage opportunity - all outcomes have positive edge
                            edge = ((1/total_prob) - 1) * 100  # Arbitrage edge
                            # Calculate fair probability from edge: fair_odds = market_odds / (1 + edge/100)
                            fair_odds = odd.decimal_odds / (1 + edge/100)
                            fair_prob = 1 / fair_odds"""

content = content.replace(old_code, new_code)

# Write back
with open('paper_trading_live.py', 'w') as f:
    f.write(content)

print("✅ Fixed fair probability calculation for arbitrage opportunities")
print("Fair probability is now calculated correctly from the edge")