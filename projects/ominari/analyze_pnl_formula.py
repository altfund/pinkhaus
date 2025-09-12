#!/usr/bin/env python3
"""Analyze P&L formula with a specific example."""

# Example: Burgos CF_vs_UD Las Palmas - Draw
stake = 31.91
odds = 3.03
fee_pct = 0.03
fee_amount = stake * fee_pct
execution_stake = stake + fee_amount

print("ANALYZING P&L FORMULA")
print("=" * 60)
print(f"Stake: ${stake:.2f}")
print(f"Odds: {odds}")
print(f"Fee %: {fee_pct*100:.1f}%")
print(f"Fee amount: ${fee_amount:.2f}")
print(f"Execution stake: ${execution_stake:.2f}")

print("\nIF BET WINS:")
print("-" * 40)

# Method 1: Current formula
gross_payout = stake * odds
pnl_method1 = gross_payout - execution_stake
print(f"Method 1 (current): P&L = (stake × odds) - execution_stake")
print(f"  = (${stake:.2f} × {odds}) - ${execution_stake:.2f}")
print(f"  = ${gross_payout:.2f} - ${execution_stake:.2f}")
print(f"  = ${pnl_method1:.2f}")

# Method 2: Net profit formula
net_profit = stake * (odds - 1) - fee_amount
print(f"\nMethod 2 (net profit): P&L = stake × (odds - 1) - fees")
print(f"  = ${stake:.2f} × ({odds} - 1) - ${fee_amount:.2f}")
print(f"  = ${stake * (odds - 1):.2f} - ${fee_amount:.2f}")
print(f"  = ${net_profit:.2f}")

# Check if they're the same
print(f"\nMethods equal? {abs(pnl_method1 - net_profit) < 0.01}")

# Check bankroll
print("\nBANKROLL CHECK:")
print("-" * 40)
initial_bankroll = 10000
after_bet = initial_bankroll - execution_stake
after_win = after_bet + gross_payout

print(f"Initial: ${initial_bankroll:.2f}")
print(f"After bet placed: ${after_bet:.2f} (deduct execution stake)")
print(f"After win: ${after_win:.2f} (add gross payout)")
print(f"Net change: ${after_win - initial_bankroll:.2f}")
print(f"Matches P&L? {abs((after_win - initial_bankroll) - pnl_method1) < 0.01}")

print("\nIF BET LOSES:")
print("-" * 40)
pnl_loss = -execution_stake
after_loss = after_bet  # Nothing returned

print(f"P&L: -${execution_stake:.2f}")
print(f"Bankroll after loss: ${after_loss:.2f}")
print(f"Net change: ${after_loss - initial_bankroll:.2f}")
print(f"Matches P&L? {abs((after_loss - initial_bankroll) - pnl_loss) < 0.01}")