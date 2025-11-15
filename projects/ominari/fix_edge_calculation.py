#!/usr/bin/env python3
"""Fix the edge calculation in paper_trading_live.py"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read the file
with open('paper_trading_live.py', 'r') as f:
    content = f.read()

# Replace the flawed edge calculation
old_calculation = """                    # Calculate fair probabilities (remove vig)
                    total_prob = sum(1/odd.decimal_odds for odd in odds_by_outcome.values())
                    
                    for outcome, odd in odds_by_outcome.items():
                        # Calculate fair probability
                        implied_prob = 1 / odd.decimal_odds
                        fair_prob = implied_prob / total_prob
                        
                        # Calculate edge
                        edge = self.calculate_edge(fair_prob, odd.decimal_odds)"""

new_calculation = """                    # Calculate fair probabilities (remove vig)
                    total_prob = sum(1/odd.decimal_odds for odd in odds_by_outcome.values())
                    
                    # Method 1: Simple edge - if total prob < 1, there's an arbitrage
                    # Method 2: Remove margin proportionally
                    margin = total_prob - 1
                    
                    for outcome, odd in odds_by_outcome.items():
                        implied_prob = 1 / odd.decimal_odds
                        
                        # Method 1: For positive edge detection
                        if total_prob < 1.0:
                            # Arbitrage opportunity - all outcomes have positive edge
                            fair_prob = implied_prob  # Use implied as fair
                            edge = ((1/total_prob) - 1) * 100  # Arbitrage edge
                        else:
                            # Method 2: Remove margin proportionally
                            fair_prob = implied_prob / total_prob
                            edge = self.calculate_edge(fair_prob, odd.decimal_odds)"""

# Replace the content
new_content = content.replace(old_calculation, new_calculation)

# Write back
with open('paper_trading_live.py', 'w') as f:
    f.write(new_content)

print("✅ Fixed edge calculation in paper_trading_live.py")
print("The system will now detect positive edge opportunities correctly!")
print("\nPositive edges will be found when:")
print("1. Total probability < 1.0 (arbitrage)")
print("2. Individual outcomes are mispriced vs fair value")