#!/usr/bin/env python3
"""Debug Kelly formula calculation"""

# Liverpool example: 27% edge, odds 3.3
odds = 3.3
implied_prob = 1 / odds  # 0.303
edge = 0.2729  # 27.29%

# Calculate true fair probability from edge
# Edge = (odds / fair_odds - 1)
# 0.2729 = (3.3 / fair_odds - 1)
# 1.2729 = 3.3 / fair_odds
# fair_odds = 3.3 / 1.2729 = 2.592
fair_odds = 3.3 / (1 + edge)
fair_prob = 1 / fair_odds

print(f"Odds: {odds}")
print(f"Implied probability: {implied_prob:.3f}")
print(f"Edge: {edge:.2%}")
print(f"Fair odds: {fair_odds:.3f}")
print(f"Fair probability: {fair_prob:.3f}")

# Kelly formula: f = (p * b - q) / b
b = odds - 1
p = fair_prob
q = 1 - p
f = (p * b - q) / b

print(f"\nKelly calculation:")
print(f"b (odds - 1): {b}")
print(f"p (fair prob): {p:.3f}")
print(f"q (1 - p): {q:.3f}")
print(f"f = ({p:.3f} * {b} - {q:.3f}) / {b}")
print(f"f = {f:.3f}")
print(f"Kelly fraction (25%): {f * 0.25:.3f}")
print(f"Bet on $10,000: ${10000 * f * 0.25:.2f}")

# The issue is the fair_prob being passed is actually implied_prob
# Let's see what happens with implied prob
print(f"\n--- Using implied prob (wrong) ---")
p_wrong = implied_prob
q_wrong = 1 - p_wrong
f_wrong = (p_wrong * b - q_wrong) / b
print(f"f = ({p_wrong:.3f} * {b} - {q_wrong:.3f}) / {b}")
print(f"f = {f_wrong:.3f}")
print(f"Kelly fraction (25%): {f_wrong * 0.25:.3f}")
print(f"Bet on $10,000: ${10000 * f_wrong * 0.25:.2f}")