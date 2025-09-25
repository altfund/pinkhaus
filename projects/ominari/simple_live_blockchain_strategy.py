#!/usr/bin/env python3
"""
Simple Live Blockchain Trading Strategy for Ominari
Uses PostgreSQL blockchain data without complex dependencies.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleTrade:
    """Simple trade record."""
    market_id: str
    position: str  # 'Home', 'Away', 'Draw'
    stake: float
    odds: float
    entry_time: datetime
    reasoning: str
    expected_edge: float


@dataclass
class TradeResult:
    """Result of a completed trade."""
    trade: SimpleTrade
    exit_time: datetime
    result: str  # 'win', 'loss', 'push'
    pnl: float


class SimpleBlockchainAnalyzer:
    """Simple analyzer for blockchain betting data."""
    
    def __init__(self):
        self.min_edge = 0.05  # 5% minimum expected edge
        
    def analyze_markets(self, markets: List[Market]) -> List[SimpleTrade]:
        """Analyze markets for trading opportunities."""
        trades = []
        
        for market in markets:
            trade = self._analyze_market(market)
            if trade:
                trades.append(trade)
                
        return trades
    
    def _analyze_market(self, market: Market) -> Optional[SimpleTrade]:
        """Analyze a single market for trading opportunity."""
        try:
            # Get current odds
            with db_manager.get_db_session() as db:
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(10).all()
                
                if len(odds) < 2:
                    return None
                
                # Group odds by outcome
                odds_by_outcome = {}
                for odd in odds:
                    if odd.outcome not in odds_by_outcome:
                        odds_by_outcome[odd.outcome] = []
                    odds_by_outcome[odd.outcome].append(odd)
                
                # Strategy 1: Look for odds movements
                movement_trade = self._analyze_odds_movement(market, odds_by_outcome)
                if movement_trade:
                    return movement_trade
                
                # Strategy 2: Look for value in market timing
                timing_trade = self._analyze_market_timing(market, odds_by_outcome)
                if timing_trade:
                    return timing_trade
                
                # Strategy 3: Look for basic arbitrage opportunities
                arb_trade = self._analyze_basic_arbitrage(market, odds_by_outcome)
                if arb_trade:
                    return arb_trade
                    
        except Exception as e:
            logger.warning(f"Error analyzing market {market.source_id}: {e}")
            
        return None
    
    def _analyze_odds_movement(self, market: Market, odds_by_outcome: Dict) -> Optional[SimpleTrade]:
        """Look for significant odds movements."""
        for outcome, outcome_odds in odds_by_outcome.items():
            if len(outcome_odds) >= 3:
                # Calculate recent vs older odds
                recent_odds = outcome_odds[0].decimal_odds
                older_odds = outcome_odds[-1].decimal_odds
                
                if older_odds > 0:
                    movement = (recent_odds - older_odds) / older_odds
                    
                    # If odds increased significantly (less favored), might be value
                    if movement > 0.15:  # 15% increase in odds
                        return SimpleTrade(
                            market_id=market.source_id,
                            position=outcome,
                            stake=50.0,  # Fixed stake for simplicity
                            odds=recent_odds,
                            entry_time=datetime.now(timezone.utc),
                            reasoning=f"Odds movement: {movement:.1%} increase suggests value",
                            expected_edge=min(movement / 2, 0.2)
                        )
        
        return None
    
    def _analyze_market_timing(self, market: Market, odds_by_outcome: Dict) -> Optional[SimpleTrade]:
        """Look for late-market opportunities."""
        now = datetime.now(timezone.utc)
        if not market.maturity_date or market.maturity_date <= now:
            return None
            
        hours_remaining = (market.maturity_date - now).total_seconds() / 3600
        
        # Look for value in markets closing within 1-3 hours
        if 1.0 <= hours_remaining <= 3.0:
            # Find the underdog (highest odds)
            best_outcome = None
            best_odds = 0
            
            for outcome, outcome_odds in odds_by_outcome.items():
                if outcome_odds and outcome_odds[0].decimal_odds > best_odds:
                    best_odds = outcome_odds[0].decimal_odds
                    best_outcome = outcome
            
            # If underdog has odds > 3.0, consider betting
            if best_outcome and best_odds > 3.0:
                return SimpleTrade(
                    market_id=market.source_id,
                    position=best_outcome,
                    stake=25.0,  # Smaller stake for underdog bets
                    odds=best_odds,
                    entry_time=datetime.now(timezone.utc),
                    reasoning=f"Late market underdog bet at {best_odds:.2f}",
                    expected_edge=0.08  # Assume 8% edge for underdog value
                )
        
        return None
    
    def _analyze_basic_arbitrage(self, market: Market, odds_by_outcome: Dict) -> Optional[SimpleTrade]:
        """Look for basic value opportunities."""
        # Calculate total implied probability
        total_implied_prob = 0
        outcome_probs = {}
        
        for outcome, outcome_odds in odds_by_outcome.items():
            if outcome_odds:
                odds_value = outcome_odds[0].decimal_odds
                implied_prob = 1 / odds_value
                outcome_probs[outcome] = implied_prob
                total_implied_prob += implied_prob
        
        # If total probability is very high (>1.2), market has high vig
        # Look for the most undervalued outcome
        if total_implied_prob > 1.2 and len(outcome_probs) >= 2:
            # Find outcome with best value (lowest implied probability relative to fair value)
            fair_prob = 1 / len(outcome_probs)  # Assume equal probability for simplicity
            
            best_outcome = None
            best_value = 0
            
            for outcome, implied_prob in outcome_probs.items():
                value = fair_prob - implied_prob
                if value > best_value:
                    best_value = value
                    best_outcome = outcome
            
            if best_outcome and best_value > 0.05:  # 5% edge
                odds_value = 1 / outcome_probs[best_outcome]
                return SimpleTrade(
                    market_id=market.source_id,
                    position=best_outcome,
                    stake=30.0,
                    odds=odds_value,
                    entry_time=datetime.now(timezone.utc),
                    reasoning=f"Value bet: {best_value:.1%} edge over fair odds",
                    expected_edge=best_value
                )
        
        return None


class SimpleLiveTradingStrategy:
    """Simple live trading strategy without complex dependencies."""
    
    def __init__(self, initial_capital: float = 1000):
        self.analyzer = SimpleBlockchainAnalyzer()
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.active_trades: List[SimpleTrade] = []
        self.completed_trades: List[TradeResult] = []
        self.max_position = 0.1  # 10% max per position
        
    def run_strategy(self, duration_minutes: int = 30):
        """Run the strategy for specified duration."""
        logger.info(f"🚀 Starting Simple Live Blockchain Trading Strategy")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Max Position Size: {self.max_position:.1%}")
        
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        cycle_count = 0
        
        while datetime.now(timezone.utc) < end_time:
            cycle_count += 1
            logger.info(f"\n🔄 Trading Cycle {cycle_count}")
            
            try:
                # 1. Scan for opportunities
                opportunities = self._scan_opportunities()
                
                # 2. Execute trades
                if opportunities:
                    logger.info(f"   Found {len(opportunities)} opportunities")
                    for trade in opportunities[:3]:  # Limit to 3 trades per cycle
                        self._execute_trade(trade)
                else:
                    logger.info("   No opportunities found")
                
                # 3. Check existing positions
                self._check_positions()
                
                # 4. Report status
                self._report_status()
                
                # Wait 2 minutes between cycles
                time.sleep(120)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(30)
        
        # Final report
        self._generate_final_report()
    
    def _scan_opportunities(self) -> List[SimpleTrade]:
        """Scan for trading opportunities."""
        try:
            with db_manager.get_db_session() as db:
                # Get markets starting in next 4 hours
                now = datetime.now(timezone.utc)
                cutoff = now + timedelta(hours=4)
                
                markets = db.query(Market).filter(
                    Market.maturity_date > now,
                    Market.maturity_date < cutoff,
                    Market.is_finished == False
                ).limit(20).all()
                
                if not markets:
                    return []
                
                # Analyze markets
                opportunities = self.analyzer.analyze_markets(markets)
                
                # Filter opportunities based on capital and existing positions
                filtered_opportunities = []
                for trade in opportunities:
                    if self._can_execute_trade(trade):
                        filtered_opportunities.append(trade)
                
                return filtered_opportunities
                
        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
            return []
    
    def _can_execute_trade(self, trade: SimpleTrade) -> bool:
        """Check if we can execute the trade based on risk limits."""
        # Check capital
        if trade.stake > self.capital:
            return False
            
        # Check position size limit
        max_stake = self.capital * self.max_position
        if trade.stake > max_stake:
            trade.stake = max_stake  # Adjust stake to limit
            
        # Check if we already have a position in this market
        for active_trade in self.active_trades:
            if active_trade.market_id == trade.market_id:
                return False  # Don't double up
                
        return True
    
    def _execute_trade(self, trade: SimpleTrade):
        """Execute a trade."""
        try:
            # Get market details for logging
            with db_manager.get_db_session() as db:
                market = db.query(Market).filter(Market.source_id == trade.market_id).first()
                
                if market:
                    logger.info(f"   ✅ Executing trade: {market.home_team} vs {market.away_team}")
                    logger.info(f"      Position: {trade.position}")
                    logger.info(f"      Stake: ${trade.stake:.2f}")
                    logger.info(f"      Odds: {trade.odds:.2f}")
                    logger.info(f"      Expected Edge: {trade.expected_edge:.1%}")
                    logger.info(f"      Reasoning: {trade.reasoning}")
                    
                    # Reduce capital
                    self.capital -= trade.stake
                    
                    # Add to active trades
                    self.active_trades.append(trade)
                    
                else:
                    logger.warning(f"   ❌ Market not found: {trade.market_id}")
                    
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _check_positions(self):
        """Check and potentially close positions."""
        # For demo purposes, randomly "settle" some old positions
        now = datetime.now(timezone.utc)
        
        settled_trades = []
        for trade in self.active_trades:
            # If trade is more than 10 minutes old, simulate settlement
            age_minutes = (now - trade.entry_time).total_seconds() / 60
            
            if age_minutes > 10:  # Settle after 10 minutes for demo
                # Simulate random outcome (in real trading, this would come from actual results)
                import random
                won = random.random() < (1 / trade.odds + trade.expected_edge)
                
                if won:
                    pnl = trade.stake * (trade.odds - 1)
                    self.capital += trade.stake + pnl
                    result = 'win'
                else:
                    pnl = -trade.stake
                    result = 'loss'
                
                trade_result = TradeResult(
                    trade=trade,
                    exit_time=now,
                    result=result,
                    pnl=pnl
                )
                
                self.completed_trades.append(trade_result)
                settled_trades.append(trade)
                
                logger.info(f"   🎯 Position settled: {result.upper()} - P&L: ${pnl:+.2f}")
        
        # Remove settled trades from active list
        for trade in settled_trades:
            self.active_trades.remove(trade)
    
    def _report_status(self):
        """Report current strategy status."""
        total_value = self.capital
        
        # Add value of active positions (using stake as book value)
        for trade in self.active_trades:
            total_value += trade.stake
            
        logger.info(f"   💰 Capital: ${self.capital:.2f}")
        logger.info(f"   📊 Active Positions: {len(self.active_trades)}")
        logger.info(f"   📈 Total Value: ${total_value:.2f}")
    
    def _generate_final_report(self):
        """Generate final performance report."""
        logger.info("\n" + "="*50)
        logger.info("📊 SIMPLE LIVE TRADING FINAL REPORT")
        logger.info("="*50)
        
        total_pnl = sum(result.pnl for result in self.completed_trades)
        total_trades = len(self.completed_trades)
        wins = len([r for r in self.completed_trades if r.result == 'win'])
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        final_capital = self.capital + sum(trade.stake for trade in self.active_trades)
        
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Capital: ${final_capital:.2f}")
        logger.info(f"Total P&L: ${total_pnl:+.2f}")
        logger.info(f"Return: {((final_capital / self.initial_capital) - 1) * 100:+.1f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Active Positions: {len(self.active_trades)}")
        
        if self.completed_trades:
            avg_win = sum(r.pnl for r in self.completed_trades if r.result == 'win') / max(wins, 1)
            losses = [r for r in self.completed_trades if r.result == 'loss']
            avg_loss = sum(r.pnl for r in losses) / max(len(losses), 1) if losses else 0
            
            logger.info(f"Average Win: ${avg_win:.2f}")
            logger.info(f"Average Loss: ${avg_loss:.2f}")


def main():
    """Run the simple live trading strategy."""
    strategy = SimpleLiveTradingStrategy(initial_capital=1000)
    
    # Run for 10 minutes as a demo
    strategy.run_strategy(duration_minutes=10)


if __name__ == "__main__":
    main()