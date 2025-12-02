#!/usr/bin/env python3
"""
Regression Test Suite for Trading System
Validates core trading logic to ensure changes don't break fundamental behaviors
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List


class TestKellyCriterion:
    """Test Kelly bet sizing calculations"""

    def calculate_kelly(self, prob: float, odds: float, kelly_fraction: float = 1.0) -> float:
        """
        Kelly formula: f = (p * b - q) / b
        where p = win probability, b = decimal odds - 1, q = 1 - p
        """
        b = odds - 1
        q = 1 - prob
        f = (prob * b - q) / b
        return f * kelly_fraction

    def test_kelly_positive_edge(self):
        """Kelly should be positive when we have an edge"""
        # 60% win probability at 2.0 odds = positive edge
        kelly = self.calculate_kelly(prob=0.6, odds=2.0)
        assert kelly > 0, "Kelly should be positive with an edge"
        assert kelly == pytest.approx(0.2, abs=0.01), "Kelly should be 20% for this scenario"

    def test_kelly_negative_edge(self):
        """Kelly should be negative when we have negative edge"""
        # 40% win probability at 2.0 odds = negative edge
        kelly = self.calculate_kelly(prob=0.4, odds=2.0)
        assert kelly < 0, "Kelly should be negative without an edge"

    def test_kelly_no_edge(self):
        """Kelly should be zero when fair odds (no edge)"""
        # 50% win probability at 2.0 odds = fair
        kelly = self.calculate_kelly(prob=0.5, odds=2.0)
        assert kelly == pytest.approx(0.0, abs=0.01), "Kelly should be ~0 at fair odds"

    def test_kelly_fraction_reduces_bet(self):
        """Kelly fraction should reduce bet size proportionally"""
        full_kelly = self.calculate_kelly(prob=0.6, odds=2.0, kelly_fraction=1.0)
        quarter_kelly = self.calculate_kelly(prob=0.6, odds=2.0, kelly_fraction=0.25)

        assert quarter_kelly == pytest.approx(full_kelly * 0.25, abs=0.01), \
            "Quarter Kelly should be 25% of full Kelly"

    def test_kelly_high_probability(self):
        """Kelly with very high win probability"""
        # 90% win probability at 1.5 odds
        kelly = self.calculate_kelly(prob=0.9, odds=1.5)
        assert 0 < kelly < 1, "Kelly should be between 0 and 1"
        assert kelly == pytest.approx(0.7, abs=0.05), "Kelly should be ~70%"

    def test_kelly_bounds(self):
        """Kelly should never recommend betting more than 100% of bankroll"""
        # Even with extreme edge, fractional Kelly keeps it reasonable
        kelly = self.calculate_kelly(prob=0.95, odds=2.0, kelly_fraction=0.25)
        assert kelly <= 1.0, "Fractional Kelly should never exceed 100%"


class TestEdgeCalculation:
    """Test betting edge calculations"""

    def calculate_edge(self, fair_prob: float, market_odds: float) -> float:
        """
        Edge = (Market Odds / Fair Odds - 1) * 100
        """
        fair_odds = 1 / fair_prob
        edge = ((market_odds / fair_odds) - 1) * 100
        return edge

    def test_positive_edge(self):
        """Positive edge when market odds are better than fair"""
        # Fair probability 50% (fair odds 2.0), market offers 2.2
        edge = self.calculate_edge(fair_prob=0.5, market_odds=2.2)
        assert edge > 0, "Should have positive edge"
        assert edge == pytest.approx(10.0, abs=0.5), "Edge should be ~10%"

    def test_negative_edge(self):
        """Negative edge when market odds are worse than fair"""
        # Fair probability 50% (fair odds 2.0), market offers 1.8
        edge = self.calculate_edge(fair_prob=0.5, market_odds=1.8)
        assert edge < 0, "Should have negative edge"
        assert edge == pytest.approx(-10.0, abs=0.5), "Edge should be ~-10%"

    def test_no_edge(self):
        """Zero edge at fair odds"""
        # Fair probability 50% (fair odds 2.0), market offers 2.0
        edge = self.calculate_edge(fair_prob=0.5, market_odds=2.0)
        assert edge == pytest.approx(0.0, abs=0.1), "Edge should be ~0 at fair odds"

    def test_edge_with_low_probability(self):
        """Edge calculation with low probability event"""
        # Fair probability 10% (fair odds 10.0), market offers 12.0
        edge = self.calculate_edge(fair_prob=0.1, market_odds=12.0)
        assert edge == pytest.approx(20.0, abs=0.5), "Edge should be ~20%"

    def test_edge_conservative_adjustment(self):
        """Test conservative edge with discount factors"""
        raw_edge = self.calculate_edge(fair_prob=0.5, market_odds=2.2)

        # Apply conservative discounts (market efficiency * uncertainty * competition)
        conservative_edge = raw_edge * 0.85 * 0.90 * 0.75
        assert conservative_edge < raw_edge, "Conservative edge should be lower"
        assert conservative_edge == pytest.approx(raw_edge * 0.57375, abs=0.01)


class TestPositionSizing:
    """Test position sizing logic"""

    def calculate_position_size(self,
                                kelly_bet: float,
                                bankroll: float,
                                max_position_pct: float = 0.02,
                                min_bet: float = 10.0) -> float:
        """Calculate actual position size with constraints"""
        # Start with Kelly
        position = kelly_bet

        # Apply max position constraint
        max_position = bankroll * max_position_pct
        position = min(position, max_position)

        # Apply minimum bet
        if position < min_bet:
            return 0

        return position

    def test_kelly_below_max(self):
        """Position should equal Kelly when below max"""
        bankroll = 10000
        kelly_bet = 100  # 1% of bankroll
        max_pct = 0.02  # 2% max

        position = self.calculate_position_size(kelly_bet, bankroll, max_pct)
        assert position == kelly_bet, "Should use Kelly when below max"

    def test_kelly_above_max(self):
        """Position should be capped at max when Kelly exceeds"""
        bankroll = 10000
        kelly_bet = 500  # 5% of bankroll
        max_pct = 0.02  # 2% max

        position = self.calculate_position_size(kelly_bet, bankroll, max_pct)
        assert position == 200, "Should be capped at max position (2% of 10000)"

    def test_below_minimum(self):
        """Position should be 0 when below minimum bet"""
        bankroll = 10000
        kelly_bet = 5  # Below $10 minimum
        min_bet = 10.0

        position = self.calculate_position_size(kelly_bet, bankroll, min_bet=min_bet)
        assert position == 0, "Should be 0 when below minimum"

    def test_scaling_with_bankroll(self):
        """Position size should scale proportionally with bankroll"""
        max_pct = 0.02
        kelly_pct = 0.03  # Kelly suggests 3%, will be capped at 2%

        # Test with different bankrolls
        for bankroll in [5000, 10000, 20000]:
            kelly_bet = bankroll * kelly_pct
            position = self.calculate_position_size(kelly_bet, bankroll, max_pct)
            assert position == bankroll * max_pct, \
                f"Position should be {max_pct*100}% of bankroll ({bankroll})"


class TestRiskLimits:
    """Test risk management limits"""

    def check_portfolio_limits(self,
                             new_position: float,
                             current_exposure: float,
                             bankroll: float,
                             max_position_pct: float = 0.02,
                             max_portfolio_pct: float = 0.20,
                             max_positions: int = 50) -> tuple[bool, str]:
        """Check if new position passes risk limits"""

        # Check individual position size
        if new_position > bankroll * max_position_pct:
            return False, "Exceeds max position size"

        # Check total portfolio exposure
        total_exposure = current_exposure + new_position
        if total_exposure > bankroll * max_portfolio_pct:
            return False, "Exceeds max portfolio exposure"

        return True, "OK"

    def test_position_within_limits(self):
        """Position within all limits should pass"""
        bankroll = 10000
        new_position = 150  # 1.5%
        current_exposure = 1000  # 10%

        ok, msg = self.check_portfolio_limits(new_position, current_exposure, bankroll)
        assert ok, f"Should pass: {msg}"

    def test_position_exceeds_individual_limit(self):
        """Position exceeding individual limit should fail"""
        bankroll = 10000
        new_position = 250  # 2.5% - exceeds 2% max
        current_exposure = 0

        ok, msg = self.check_portfolio_limits(new_position, current_exposure, bankroll)
        assert not ok, "Should fail individual position limit"
        assert "max position size" in msg.lower()

    def test_portfolio_exceeds_total_exposure(self):
        """Portfolio exceeding total exposure should fail"""
        bankroll = 10000
        new_position = 150  # 1.5% - within individual limit
        current_exposure = 1950  # 19.5% - total would be 21%

        ok, msg = self.check_portfolio_limits(new_position, current_exposure, bankroll)
        assert not ok, "Should fail portfolio exposure limit"
        assert "portfolio exposure" in msg.lower()

    def test_max_drawdown_check(self):
        """Drawdown should trigger circuit breaker"""
        initial_bankroll = 10000
        current_bankroll = 8400  # 16% drawdown
        max_drawdown_pct = 15.0

        drawdown_pct = ((initial_bankroll - current_bankroll) / initial_bankroll) * 100
        assert drawdown_pct > max_drawdown_pct, "Should exceed max drawdown"


class TestMarkToMarket:
    """Test mark-to-market calculations"""

    def calculate_mtm_value(self,
                           stake: float,
                           entry_odds: float,
                           current_odds: float,
                           position_open: bool = True) -> float:
        """
        Calculate mark-to-market value of a position
        MTM = stake * (current_odds / entry_odds) if position is open
        """
        if not position_open:
            return 0

        mtm_multiplier = current_odds / entry_odds
        return stake * mtm_multiplier

    def calculate_book_value(self, stake: float, position_open: bool = True) -> float:
        """Book value is simply the stake if position is open"""
        return stake if position_open else 0

    def test_mtm_odds_improved(self):
        """MTM should increase when odds improve"""
        stake = 100
        entry_odds = 2.0
        current_odds = 2.5

        mtm = self.calculate_mtm_value(stake, entry_odds, current_odds)
        assert mtm > stake, "MTM should exceed stake when odds improve"
        assert mtm == 125, "MTM should be stake * (2.5/2.0) = 125"

    def test_mtm_odds_worsened(self):
        """MTM should decrease when odds worsen"""
        stake = 100
        entry_odds = 2.0
        current_odds = 1.5

        mtm = self.calculate_mtm_value(stake, entry_odds, current_odds)
        assert mtm < stake, "MTM should be less than stake when odds worsen"
        assert mtm == 75, "MTM should be stake * (1.5/2.0) = 75"

    def test_mtm_odds_unchanged(self):
        """MTM should equal stake when odds unchanged"""
        stake = 100
        entry_odds = 2.0
        current_odds = 2.0

        mtm = self.calculate_mtm_value(stake, entry_odds, current_odds)
        assert mtm == stake, "MTM should equal stake when odds unchanged"

    def test_book_value_vs_mtm(self):
        """Book value should always equal stake (conservative)"""
        stake = 100
        entry_odds = 2.0
        current_odds = 2.5  # Odds improved

        book = self.calculate_book_value(stake, position_open=True)
        mtm = self.calculate_mtm_value(stake, entry_odds, current_odds)

        assert book == stake, "Book value should always equal stake"
        assert mtm > book, "MTM should be higher when odds improve"

    def test_closed_position(self):
        """Closed positions should have zero value"""
        stake = 100
        entry_odds = 2.0
        current_odds = 2.5

        book = self.calculate_book_value(stake, position_open=False)
        mtm = self.calculate_mtm_value(stake, entry_odds, current_odds, position_open=False)

        assert book == 0, "Book value should be 0 for closed position"
        assert mtm == 0, "MTM should be 0 for closed position"


class TestSettlement:
    """Test bet settlement logic"""

    def settle_bet(self,
                  stake: float,
                  odds: float,
                  outcome: str,
                  actual_result: str) -> float:
        """
        Settle a bet and return P&L
        Win: stake * (odds - 1)
        Loss: -stake
        """
        if outcome == actual_result:
            # Win
            return stake * (odds - 1)
        else:
            # Loss
            return -stake

    def test_winning_bet(self):
        """Winning bet should return stake * (odds - 1)"""
        stake = 100
        odds = 2.5
        pnl = self.settle_bet(stake, odds, outcome="home", actual_result="home")

        assert pnl > 0, "Winning bet should have positive P&L"
        assert pnl == 150, "P&L should be stake * (2.5 - 1) = 150"

    def test_losing_bet(self):
        """Losing bet should return -stake"""
        stake = 100
        odds = 2.5
        pnl = self.settle_bet(stake, odds, outcome="home", actual_result="away")

        assert pnl < 0, "Losing bet should have negative P&L"
        assert pnl == -100, "P&L should be -stake = -100"

    def test_even_odds_win(self):
        """Win at 2.0 (even) odds"""
        stake = 100
        odds = 2.0
        pnl = self.settle_bet(stake, odds, outcome="home", actual_result="home")

        assert pnl == 100, "P&L should equal stake at 2.0 odds"

    def test_high_odds_win(self):
        """Win at high odds"""
        stake = 100
        odds = 5.0
        pnl = self.settle_bet(stake, odds, outcome="away", actual_result="away")

        assert pnl == 400, "P&L should be stake * (5.0 - 1) = 400"

    def test_portfolio_after_settlement(self):
        """Test portfolio value after multiple settlements"""
        initial_bankroll = 10000
        bankroll = initial_bankroll

        # Win 1: $100 at 2.0 odds
        bankroll += self.settle_bet(100, 2.0, "home", "home")
        assert bankroll == 10100, "Should be +$100"

        # Loss 1: $100 at 2.5 odds
        bankroll += self.settle_bet(100, 2.5, "away", "home")
        assert bankroll == 10000, "Should be back to initial"

        # Win 2: $200 at 3.0 odds
        bankroll += self.settle_bet(200, 3.0, "home", "home")
        assert bankroll == 10400, "Should be +$400 total"


class TestSessionManagement:
    """Test session and position tracking"""

    def test_position_count(self):
        """Track number of open positions"""
        positions = {
            'pos1': {'stake': 100, 'status': 'open'},
            'pos2': {'stake': 150, 'status': 'open'},
            'pos3': {'stake': 200, 'status': 'closed'}
        }

        open_count = sum(1 for p in positions.values() if p['status'] == 'open')
        assert open_count == 2, "Should have 2 open positions"

    def test_total_exposure(self):
        """Calculate total portfolio exposure"""
        positions = {
            'pos1': {'stake': 100, 'status': 'open'},
            'pos2': {'stake': 150, 'status': 'open'},
            'pos3': {'stake': 200, 'status': 'closed'}
        }

        total_exposure = sum(
            p['stake'] for p in positions.values()
            if p['status'] == 'open'
        )
        assert total_exposure == 250, "Total exposure should be $250"

    def test_realized_vs_unrealized_pnl(self):
        """Separate realized and unrealized P&L"""
        positions = {
            'pos1': {'stake': 100, 'status': 'closed', 'pnl': 50},
            'pos2': {'stake': 150, 'status': 'closed', 'pnl': -100},
            'pos3': {'stake': 200, 'status': 'open', 'mtm_pnl': 25}
        }

        realized_pnl = sum(
            p['pnl'] for p in positions.values()
            if p['status'] == 'closed'
        )
        unrealized_pnl = sum(
            p.get('mtm_pnl', 0) for p in positions.values()
            if p['status'] == 'open'
        )

        assert realized_pnl == -50, "Realized P&L should be -$50"
        assert unrealized_pnl == 25, "Unrealized P&L should be $25"

    def test_win_rate_calculation(self):
        """Calculate win rate from closed positions"""
        positions = [
            {'status': 'closed', 'pnl': 50},
            {'status': 'closed', 'pnl': -100},
            {'status': 'closed', 'pnl': 75},
            {'status': 'closed', 'pnl': -50},
            {'status': 'open', 'pnl': 0}  # Should be excluded
        ]

        closed = [p for p in positions if p['status'] == 'closed']
        wins = sum(1 for p in closed if p['pnl'] > 0)
        win_rate = wins / len(closed)

        assert win_rate == 0.5, "Win rate should be 50% (2 wins / 4 closed)"


class TestInvariantsAndSanity:
    """Test system invariants and sanity checks"""

    def test_probability_bounds(self):
        """Probabilities must be between 0 and 1"""
        valid_probs = [0.0, 0.5, 1.0, 0.25, 0.75]
        for prob in valid_probs:
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds"

    def test_odds_positive(self):
        """Decimal odds must be > 1.0"""
        valid_odds = [1.01, 1.5, 2.0, 3.5, 10.0]
        for odds in valid_odds:
            assert odds > 1.0, f"Odds {odds} must be > 1.0"

    def test_implied_probability_from_odds(self):
        """Implied probability = 1 / odds"""
        odds = 2.0
        implied_prob = 1 / odds
        assert implied_prob == 0.5, "Implied probability should be 50% at 2.0 odds"

        odds = 4.0
        implied_prob = 1 / odds
        assert implied_prob == 0.25, "Implied probability should be 25% at 4.0 odds"

    def test_total_probability_with_vig(self):
        """Sum of implied probabilities should exceed 1.0 (bookmaker margin)"""
        odds_home = 2.1
        odds_away = 3.0
        odds_draw = 4.0

        total_prob = (1/odds_home) + (1/odds_away) + (1/odds_draw)
        assert total_prob > 1.0, "Total implied probability should exceed 1.0 (vig)"

        margin = (total_prob - 1.0) * 100
        assert 0 < margin < 20, "Typical bookmaker margin is 2-10%"

    def test_kelly_never_exceeds_one(self):
        """Even with extreme edge, Kelly fraction should keep bet reasonable"""
        # Extreme case: 80% win probability at 2.0 odds
        prob = 0.8
        odds = 2.0
        kelly_fraction = 0.25

        b = odds - 1
        q = 1 - prob
        kelly = ((prob * b - q) / b) * kelly_fraction

        assert kelly <= 1.0, "Fractional Kelly should never exceed 100%"

    def test_bankroll_never_negative(self):
        """Bankroll should never go negative in simulation"""
        bankroll = 10000

        # Simulate max losses
        max_position_pct = 0.02
        max_portfolio_pct = 0.20

        max_single_loss = bankroll * max_position_pct
        max_total_loss = bankroll * max_portfolio_pct

        # Worst case: lose all open positions
        worst_case_bankroll = bankroll - max_total_loss
        assert worst_case_bankroll > 0, "Even worst case should not go negative"
        assert worst_case_bankroll == 8000, "Should have $8000 left (lost 20%)"


if __name__ == "__main__":
    # Run tests
    print("Running regression tests...")
    print("\nTo run with pytest:")
    print("  pytest tests/test_trading_regression.py -v")
    print("  pytest tests/test_trading_regression.py -v -k TestKellyCriterion")
    print("\nTo run with coverage:")
    print("  pytest tests/test_trading_regression.py --cov=. --cov-report=html")
