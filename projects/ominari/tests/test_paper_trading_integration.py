#!/usr/bin/env python3
"""
Integration tests for paper trading system
"""

import pytest
import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Set up test environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from paper_trading_live import LivePaperTrader


class TestPaperTradingIntegration:
    """Integration tests for paper trading"""
    
    @pytest.fixture
    def bankroll_config(self):
        """Create test bankroll config"""
        config = BankrollConfig(config_file="test_bankroll.json")
        config.config["initial_bankroll"] = 10000.0
        config.config["current_bankroll"] = 10000.0
        config.save_config()
        yield config
        # Cleanup
        if os.path.exists("test_bankroll.json"):
            os.remove("test_bankroll.json")
            
    @pytest.fixture
    def trader(self, bankroll_config):
        """Create test trader"""
        trader = LivePaperTrader()
        trader.bankroll_config = bankroll_config
        trader.min_edge = -10.0  # Allow negative edge for testing
        return trader
        
    @pytest.mark.asyncio
    async def test_find_betting_opportunities(self, trader):
        """Test finding betting opportunities"""
        opportunities = await trader.find_betting_opportunities()
        
        # Should find some opportunities (even if negative edge)
        assert isinstance(opportunities, list)
        
        if opportunities:
            opp = opportunities[0]
            assert 'market' in opp
            assert 'outcome' in opp
            assert 'odds' in opp
            assert 'fair_prob' in opp
            assert 'edge' in opp
            
    @pytest.mark.asyncio
    async def test_kelly_bet_calculation(self, trader):
        """Test Kelly bet sizing"""
        # Test with positive edge
        bet_size = trader.calculate_kelly_bet(0.55, 2.0, 10000)
        assert bet_size > 0
        assert bet_size <= 500  # Max 5% of bankroll
        
        # Test with negative edge
        bet_size = trader.calculate_kelly_bet(0.45, 2.0, 10000)
        assert bet_size == 0  # Should not bet
        
    @pytest.mark.asyncio
    async def test_place_bet(self, trader):
        """Test placing a bet"""
        # Create test session
        with db_manager.get_db_session() as db:
            session = BettingSession(
                as_of=datetime.now(timezone.utc),
                session_type='paper',
                strategy_name="Test",
                kelly_bankroll=10000,
                execution_bankroll=10000,
                kelly_fraction=0.25,
                cap_per_game=1000,
                cap_per_bet=500,
                cap_per_game_market=500,
                min_bet_abs=10,
                min_bet_pct=0.001,
                abs_game_limit=None,
                min_break_minutes=0,
                avg_game_duration_minutes=120
            )
            db.add(session)
            db.commit()
            trader.session_id = session.id
            
        # Find an opportunity
        opportunities = await trader.find_betting_opportunities()
        
        if opportunities:
            # Place bet on first opportunity
            bet = await trader.place_bet(opportunities[0])
            
            if bet:
                assert bet.id is not None
                assert bet.session_id == trader.session_id
                assert bet.stake > 0
                assert bet.odds > 0
                
                # Check bankroll was NOT updated yet (only on settle)
                assert trader.bankroll_config.get_current_bankroll() == 10000.0
                
    @pytest.mark.asyncio
    async def test_session_creation(self, trader):
        """Test trading session creation"""
        # Start trading loop briefly
        task = asyncio.create_task(trader.run_trading_loop())
        await asyncio.sleep(2)
        trader.is_running = False
        await task
        
        # Check session was created
        assert trader.session_id is not None
        
        with db_manager.get_db_session() as db:
            session = db.query(BettingSession).filter(
                BettingSession.id == trader.session_id
            ).first()
            
            assert session is not None
            assert session.session_type == 'paper'
            assert session.kelly_bankroll == 10000.0
            
    def test_edge_calculation(self, trader):
        """Test edge calculation"""
        # Positive edge
        edge = trader.calculate_edge(0.5, 2.1)  # Fair odds 2.0, market 2.1
        assert edge == 5.0
        
        # Negative edge
        edge = trader.calculate_edge(0.5, 1.9)  # Fair odds 2.0, market 1.9
        assert edge == -5.0
        
    def test_bankroll_tracking(self, trader):
        """Test bankroll configuration"""
        initial = trader.bankroll_config.get_current_bankroll()
        assert initial == 10000.0
        
        # Record a win
        trader.bankroll_config.record_bet_result(True, 100.0)
        assert trader.bankroll_config.get_current_bankroll() == 10100.0
        
        # Record a loss
        trader.bankroll_config.record_bet_result(False, -50.0)
        assert trader.bankroll_config.get_current_bankroll() == 10050.0
        
        # Check stats
        stats = trader.bankroll_config.get_performance_stats()
        assert stats['total_bets'] == 2
        assert stats['wins'] == 1
        assert stats['losses'] == 1
        assert stats['win_rate'] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])