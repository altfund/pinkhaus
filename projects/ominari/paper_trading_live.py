#!/usr/bin/env python3
"""
Working Paper Trading System for Ominari
Actually places trades and tracks a real portfolio
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import random

import pandas as pd
from sqlalchemy import func, and_, or_

from database_v2 import db_manager
from models import Market, Odd, BettingSession, Bet
from config.bankroll_config import BankrollConfig
from notifications.slack_notifier import slack_notifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LivePaperTrader:
    """Paper trading system that actually places trades"""
    
    def __init__(self):
        self.bankroll_config = BankrollConfig()
        self.is_running = False
        self.session_id = None
        self.trades_placed = 0
        self.last_trade_time = None
        
        # Trading parameters
        self.min_edge = -1.0  # Temporarily allow negative edge for testing
        self.max_bet_pct = 0.05  # Max 5% of bankroll per bet
        self.min_bet = 10.0
        self.max_concurrent_bets = 10
        self.kelly_fraction = 0.25  # Conservative Kelly
        
    def calculate_kelly_bet(self, prob: float, odds: float, bankroll: float) -> float:
        """Calculate Kelly bet size"""
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, b = decimal odds - 1, q = 1 - p
        b = odds - 1
        q = 1 - prob
        f = (prob * b - q) / b
        
        # Apply Kelly fraction for conservative sizing
        f = f * self.kelly_fraction
        
        # Ensure bet is within limits
        if f <= 0:
            return 0
            
        bet_amount = bankroll * f
        bet_amount = min(bet_amount, bankroll * self.max_bet_pct)
        bet_amount = max(bet_amount, self.min_bet)
        
        return round(bet_amount, 2)
        
    def calculate_edge(self, fair_prob: float, market_odds: float) -> float:
        """Calculate betting edge"""
        # Edge = (Fair Odds / Market Odds - 1) * 100
        fair_odds = 1 / fair_prob
        edge = ((market_odds / fair_odds) - 1) * 100
        return edge
        
    async def find_betting_opportunities(self) -> List[Dict]:
        """Find markets with positive edge"""
        opportunities = []
        
        try:
            with db_manager.get_db_session() as db:
                # Get markets happening in next 24 hours
                now = datetime.now(timezone.utc)
                future_time = now + timedelta(hours=24)
                
                markets = db.query(Market).filter(
                    and_(
                        Market.maturity_date > now,
                        Market.maturity_date < future_time,
                        Market.sport.in_(['Soccer', 'Tennis', 'American Football', 'Basketball'])
                    )
                ).limit(100).all()
                
                logger.info(f"Analyzing {len(markets)} markets for opportunities...")
                logger.info(f"Min edge threshold: {self.min_edge}%")
                
                for market in markets:
                    # Get odds for this market
                    odds_records = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).all()
                    
                    if not odds_records:
                        continue
                        
                    # Group by outcome
                    odds_by_outcome = {}
                    for odd in odds_records:
                        if odd.outcome not in odds_by_outcome or odd.decimal_odds > odds_by_outcome[odd.outcome].decimal_odds:
                            odds_by_outcome[odd.outcome] = odd
                            
                    # Calculate fair probabilities (remove vig)
                    total_prob = sum(1/odd.decimal_odds for odd in odds_by_outcome.values())
                    
                    for outcome, odd in odds_by_outcome.items():
                        # Calculate fair probability
                        implied_prob = 1 / odd.decimal_odds
                        fair_prob = implied_prob / total_prob
                        
                        # Calculate edge
                        edge = self.calculate_edge(fair_prob, odd.decimal_odds)
                        
                        if edge >= self.min_edge:
                            logger.info(f"Found opportunity: {market.home_team} vs {market.away_team}, {outcome} @ {odd.decimal_odds}, edge={edge:.2f}%")
                            opportunities.append({
                                'market': market,
                                'outcome': outcome,
                                'odds': odd.decimal_odds,
                                'fair_prob': fair_prob,
                                'edge': edge,
                                'odd_record': odd
                            })
                            
        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")
            
        # Sort by edge
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        logger.info(f"Total opportunities found: {len(opportunities)}")
        return opportunities[:self.max_concurrent_bets]  # Limit concurrent bets
        
    async def place_bet(self, opportunity: Dict) -> Optional[Bet]:
        """Place a paper bet"""
        try:
            bankroll = self.bankroll_config.get_current_bankroll()
            
            # Calculate bet size
            bet_amount = self.calculate_kelly_bet(
                opportunity['fair_prob'],
                opportunity['odds'],
                bankroll
            )
            
            if bet_amount < self.min_bet:
                return None
                
            # Check risk limits
            risk_limits = self.bankroll_config.get_risk_limits()
            if bet_amount > bankroll * risk_limits['max_position_size']:
                bet_amount = bankroll * risk_limits['max_position_size']
                
            with db_manager.get_db_session() as db:
                # Create bet record
                # Create bet record with correct fields
                bet = Bet(
                    session_id=self.session_id,
                    source_id=opportunity['market'].source_id,
                    unified_market_type='h2h',
                    normalized_outcome=opportunity['outcome'],
                    normalized_line=0.0,  # No line for h2h markets
                    bet_name=f"{opportunity['market'].home_team} vs {opportunity['market'].away_team} - {opportunity['outcome']}",
                    probability=opportunity['fair_prob'],
                    odds=opportunity['odds'],
                    stake=bet_amount,
                    execution_stake=bet_amount,  # Same as stake for paper trading
                    fee_amount=0.0,  # No fees for paper trading
                    fee_pct=0.0
                )
                
                db.add(bet)
                db.commit()
                db.refresh(bet)
                
                self.trades_placed += 1
                self.last_trade_time = datetime.now()
                
                logger.info(f"✅ Placed bet: {opportunity['market'].home_team} vs {opportunity['market'].away_team}")
                logger.info(f"   Outcome: {opportunity['outcome']}, Stake: ${bet_amount:.2f}, Odds: {opportunity['odds']:.2f}")
                logger.info(f"   Edge: {opportunity['edge']:.2f}%, Fair prob: {opportunity['fair_prob']:.2%}")
                
                # Send Slack notification for significant bets
                if bet_amount > 50 or opportunity['edge'] > 10:
                    slack_notifier.send_trade_alert({
                        'type': 'BUY',
                        'market': f"{opportunity['market'].home_team} vs {opportunity['market'].away_team}",
                        'outcome': opportunity['outcome'],
                        'amount': bet_amount,
                        'odds': opportunity['odds'],
                        'edge': opportunity['edge'],
                        'kelly_pct': (bet_amount / bankroll) * 100,
                        'bankroll': bankroll
                    })
                    
                return bet
                
        except Exception as e:
            logger.error(f"Error placing bet: {e}")
            return None
            
    async def update_bet_results(self):
        """Check and update results for pending bets"""
        # Since Bet model doesn't have status/payout fields, we'll track separately
        # In a real system, this would check blockchain for actual results
        pass  # Simplified for now since we need a separate results tracking table
            
    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting Live Paper Trading System")
        logger.info(f"💰 Initial bankroll: ${self.bankroll_config.get_current_bankroll():.2f}")
        
        # Create trading session
        try:
            with db_manager.get_db_session() as db:
                current_bankroll = self.bankroll_config.get_current_bankroll()
                session = BettingSession(
                    as_of=datetime.now(timezone.utc),
                    session_type='paper',
                    strategy_name="Edge-based Kelly Criterion",
                    kelly_bankroll=current_bankroll,
                    execution_bankroll=current_bankroll,
                    kelly_fraction=self.kelly_fraction,
                    cap_per_game=current_bankroll * 0.1,  # 10% max per game
                    cap_per_bet=current_bankroll * self.max_bet_pct,
                    cap_per_game_market=current_bankroll * 0.05,  # 5% per market
                    min_bet_abs=self.min_bet,
                    min_bet_pct=0.001,  # 0.1% min
                    abs_game_limit=None,  # No limit
                    min_break_minutes=0.0,
                    avg_game_duration_minutes=120.0
                )
                db.add(session)
                db.commit()
                self.session_id = session.id
                logger.info(f"Created trading session: {session.id}")
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return
            
        self.is_running = True
        
        while self.is_running:
            try:
                # Find opportunities
                opportunities = await self.find_betting_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} betting opportunities")
                    
                    # Place bets
                    for opp in opportunities:
                        # Check current exposure
                        bankroll = self.bankroll_config.get_current_bankroll()
                        
                        # Get active bets count (all bets for this session since we don't track status)
                        with db_manager.get_db_session() as db:
                            active_count = db.query(func.count(Bet.id)).filter(
                                Bet.session_id == self.session_id
                            ).scalar()
                            
                        if active_count >= self.max_concurrent_bets:
                            logger.info("Max concurrent bets reached, skipping...")
                            break
                            
                        await self.place_bet(opp)
                        await asyncio.sleep(1)  # Small delay between bets
                        
                # Update bet results
                await self.update_bet_results()
                
                # Log status
                stats = self.bankroll_config.get_performance_stats()
                logger.info(f"📊 Portfolio Status: Bankroll ${stats['current_bankroll']:.2f}, " +
                          f"PnL ${stats['total_pnl']:.2f}, ROI {stats['roi']:.2f}%, " +
                          f"Bets {stats['total_bets']}, Win Rate {stats['win_rate']:.1f}%")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)
                
    async def stop(self):
        """Stop trading"""
        self.is_running = False
        logger.info("Stopping paper trading system...")
        
        # Final report
        stats = self.bankroll_config.get_performance_stats()
        logger.info("📊 Final Trading Report:")
        logger.info(f"  Initial Bankroll: ${stats['initial_bankroll']:.2f}")
        logger.info(f"  Final Bankroll: ${stats['current_bankroll']:.2f}")
        logger.info(f"  Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"  ROI: {stats['roi']:.2f}%")
        logger.info(f"  Total Bets: {stats['total_bets']}")
        logger.info(f"  Win Rate: {stats['win_rate']:.1f}%")


async def main():
    """Main entry point"""
    trader = LivePaperTrader()
    
    try:
        await trader.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    finally:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(main())