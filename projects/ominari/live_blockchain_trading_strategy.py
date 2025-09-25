#!/usr/bin/env python3
"""
Live Blockchain Trading Strategy for Ominari
Integrates PostgreSQL blockchain data with live trading execution.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from paper_trading_engine import PaperTradingEngine, PaperOrder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal based on blockchain data analysis."""
    market_id: str
    signal_strength: float  # -1 to 1, negative = sell, positive = buy
    confidence: float  # 0 to 1
    recommended_position: str  # 'Home', 'Away', 'Draw'
    expected_edge: float
    reasoning: str


@dataclass
class TradingOpportunity:
    """Trading opportunity with risk/reward analysis."""
    signal: TradingSignal
    market: Market
    current_odds: Dict[str, float]
    recommended_stake: float
    max_loss: float
    expected_return: float


class BlockchainDataAnalyzer:
    """Analyzes blockchain data to generate trading signals."""
    
    def __init__(self):
        self.min_confidence = 0.6
        self.min_edge = 0.05  # 5% minimum edge
        
    def analyze_market_inefficiencies(self, markets_df: pd.DataFrame) -> List[TradingSignal]:
        """Analyze markets for trading opportunities based on blockchain data."""
        signals = []
        
        if markets_df.empty:
            return signals
            
        # Group by sport for sport-specific analysis
        for sport, sport_markets in markets_df.groupby('sport'):
            sport_signals = self._analyze_sport_markets(sport, sport_markets)
            signals.extend(sport_signals)
            
        return signals
    
    def _analyze_sport_markets(self, sport: str, markets_df: pd.DataFrame) -> List[TradingSignal]:
        """Analyze markets within a specific sport."""
        signals = []
        
        for _, market in markets_df.iterrows():
            # Strategy 1: Odds momentum analysis
            momentum_signal = self._analyze_odds_momentum(market)
            if momentum_signal:
                signals.append(momentum_signal)
                
            # Strategy 2: Market timing analysis
            timing_signal = self._analyze_market_timing(market)
            if timing_signal:
                signals.append(timing_signal)
                
            # Strategy 3: Volume-based signals (if we have betting volume data)
            volume_signal = self._analyze_betting_volume(market)
            if volume_signal:
                signals.append(volume_signal)
                
        return signals
    
    def _analyze_odds_momentum(self, market: pd.Series) -> Optional[TradingSignal]:
        """Analyze odds momentum for trading signals."""
        try:
            # Get recent odds history for this market
            with db_manager.get_db_session() as db:
                odds_history = db.query(Odd).filter(
                    Odd.source_id == market['source_id'],
                    Odd.updated_at > datetime.now(timezone.utc) - timedelta(hours=6)
                ).order_by(Odd.updated_at.desc()).limit(20).all()
                
                if len(odds_history) < 5:
                    return None
                    
                # Analyze odds movement
                home_odds = [o.decimal_odds for o in odds_history if o.outcome == 'Home']
                away_odds = [o.decimal_odds for o in odds_history if o.outcome == 'Away']
                
                if len(home_odds) < 3 or len(away_odds) < 3:
                    return None
                
                # Calculate momentum (simple moving average change)
                home_trend = (home_odds[0] - home_odds[-1]) / home_odds[-1]
                away_trend = (away_odds[0] - away_odds[-1]) / away_odds[-1]
                
                # Strong trend detection
                if abs(home_trend) > 0.1:  # 10% odds movement
                    if home_trend > 0:  # Home odds increasing (less favored)
                        return TradingSignal(
                            market_id=market['source_id'],
                            signal_strength=min(abs(home_trend), 0.8),
                            confidence=0.7,
                            recommended_position='Away',
                            expected_edge=abs(home_trend) / 2,
                            reasoning=f"Home odds momentum: {home_trend:.2%} increase suggests Away value"
                        )
                    else:  # Home odds decreasing (more favored)
                        return TradingSignal(
                            market_id=market['source_id'],
                            signal_strength=min(abs(home_trend), 0.8),
                            confidence=0.7,
                            recommended_position='Home',
                            expected_edge=abs(home_trend) / 2,
                            reasoning=f"Home odds momentum: {abs(home_trend):.2%} decrease suggests Home value"
                        )
                        
        except Exception as e:
            logger.warning(f"Error analyzing odds momentum for {market['source_id']}: {e}")
            
        return None
    
    def _analyze_market_timing(self, market: pd.Series) -> Optional[TradingSignal]:
        """Analyze market timing for late-market inefficiencies."""
        try:
            now = datetime.now(timezone.utc)
            maturity = market['maturity_date']
            
            if not maturity or maturity <= now:
                return None
                
            # Time until maturity
            hours_remaining = (maturity - now).total_seconds() / 3600
            
            # Look for value in markets closing soon (last 2 hours)
            if 0.5 <= hours_remaining <= 2.0:
                # Get current odds distribution
                with db_manager.get_db_session() as db:
                    current_odds = db.query(Odd).filter(
                        Odd.source_id == market['source_id']
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                    
                    if len(current_odds) >= 2:
                        home_odd = next((o for o in current_odds if o.outcome == 'Home'), None)
                        away_odd = next((o for o in current_odds if o.outcome == 'Away'), None)
                        
                        if home_odd and away_odd:
                            # Calculate implied probabilities
                            home_prob = 1 / home_odd.decimal_odds
                            away_prob = 1 / away_odd.decimal_odds
                            total_prob = home_prob + away_prob
                            
                            # Look for markets with high overround (bookmaker edge)
                            overround = total_prob - 1.0
                            
                            if overround > 0.15:  # 15% overround suggests value opportunity
                                # Recommend the underdog (higher odds)
                                if home_odd.decimal_odds > away_odd.decimal_odds:
                                    return TradingSignal(
                                        market_id=market['source_id'],
                                        signal_strength=0.6,
                                        confidence=0.65,
                                        recommended_position='Home',
                                        expected_edge=overround / 3,
                                        reasoning=f"Late market with high overround ({overround:.1%}), betting underdog"
                                    )
                                else:
                                    return TradingSignal(
                                        market_id=market['source_id'],
                                        signal_strength=0.6,
                                        confidence=0.65,
                                        recommended_position='Away',
                                        expected_edge=overround / 3,
                                        reasoning=f"Late market with high overround ({overround:.1%}), betting underdog"
                                    )
                        
        except Exception as e:
            logger.warning(f"Error analyzing market timing for {market['source_id']}: {e}")
            
        return None
    
    def _analyze_betting_volume(self, market: pd.Series) -> Optional[TradingSignal]:
        """Analyze betting volume patterns (placeholder for future enhancement)."""
        # This would require integration with actual betting volume data
        # For now, return None as we don't have volume data
        return None


class LiveTradingStrategy:
    """Main live trading strategy orchestrator."""
    
    def __init__(self, initial_capital: float = 10000, max_position_size: float = 0.1):
        self.analyzer = BlockchainDataAnalyzer()
        self.paper_engine = PaperTradingEngine(
            initial_capital=initial_capital,
            commission_rate=0.002  # 0.2% commission
        )
        self.max_position_size = max_position_size
        self.active_positions = {}
        self.performance_history = []
        
    async def run_strategy(self, duration_hours: int = 24):
        """Run the live trading strategy for specified duration."""
        logger.info(f"🚀 Starting Live Blockchain Trading Strategy")
        logger.info(f"   Duration: {duration_hours} hours")
        logger.info(f"   Initial Capital: ${self.paper_engine.initial_capital:,.2f}")
        logger.info(f"   Session ID: {self.paper_engine.session_id}")
        
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=duration_hours)
        
        cycle_count = 0
        
        while datetime.now(timezone.utc) < end_time:
            cycle_count += 1
            logger.info(f"\n🔄 Trading Cycle {cycle_count}")
            
            try:
                # 1. Scan for trading opportunities
                opportunities = await self._scan_opportunities()
                
                if opportunities:
                    logger.info(f"   Found {len(opportunities)} trading opportunities")
                    
                    # 2. Execute trades
                    for opportunity in opportunities[:5]:  # Limit to 5 trades per cycle
                        await self._execute_trade(opportunity)
                else:
                    logger.info("   No trading opportunities found")
                
                # 3. Monitor existing positions
                await self._monitor_positions()
                
                # 4. Update performance metrics
                self._update_performance()
                
                # 5. Wait before next cycle (5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
        
        # Generate final report
        self._generate_final_report()
    
    async def _scan_opportunities(self) -> List[TradingOpportunity]:
        """Scan blockchain data for trading opportunities."""
        opportunities = []
        
        try:
            # Get active markets from PostgreSQL
            with db_manager.get_db_session() as db:
                # Markets starting in next 6 hours
                now = datetime.now(timezone.utc)
                cutoff = now + timedelta(hours=6)
                
                markets = db.query(Market).filter(
                    Market.maturity_date > now,
                    Market.maturity_date < cutoff,
                    Market.is_finished == False
                ).limit(50).all()
                
                if not markets:
                    return opportunities
                
                # Convert to DataFrame for analysis
                markets_data = []
                for market in markets:
                    markets_data.append({
                        'source_id': market.source_id,
                        'sport': market.sport,
                        'home_team': market.home_team,
                        'away_team': market.away_team,
                        'maturity_date': market.maturity_date,
                        'league_name': market.league_name
                    })
                
                markets_df = pd.DataFrame(markets_data)
                
                # Generate trading signals
                signals = self.analyzer.analyze_market_inefficiencies(markets_df)
                
                # Convert signals to opportunities
                for signal in signals:
                    if signal.confidence >= self.analyzer.min_confidence:
                        opportunity = await self._create_opportunity(signal, markets)
                        if opportunity:
                            opportunities.append(opportunity)
        
        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
        
        return opportunities
    
    async def _create_opportunity(self, signal: TradingSignal, markets: List[Market]) -> Optional[TradingOpportunity]:
        """Create trading opportunity from signal."""
        try:
            # Find the market
            market = next((m for m in markets if m.source_id == signal.market_id), None)
            if not market:
                return None
            
            # Get current odds
            with db_manager.get_db_session() as db:
                current_odds_query = db.query(Odd).filter(
                    Odd.source_id == signal.market_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                current_odds = {}
                for odd in current_odds_query:
                    current_odds[odd.outcome] = odd.decimal_odds
                
                if not current_odds:
                    return None
                
                # Calculate recommended stake using Kelly criterion
                if signal.recommended_position in current_odds:
                    odds = current_odds[signal.recommended_position]
                    implied_prob = 1 / odds
                    our_prob = implied_prob + signal.expected_edge
                    
                    # Kelly fraction
                    kelly_fraction = (our_prob * odds - 1) / (odds - 1)
                    kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
                    
                    recommended_stake = self.paper_engine.current_capital * kelly_fraction
                    max_loss = recommended_stake
                    expected_return = recommended_stake * (odds - 1) * our_prob - max_loss * (1 - our_prob)
                    
                    return TradingOpportunity(
                        signal=signal,
                        market=market,
                        current_odds=current_odds,
                        recommended_stake=recommended_stake,
                        max_loss=max_loss,
                        expected_return=expected_return
                    )
        
        except Exception as e:
            logger.error(f"Error creating opportunity: {e}")
        
        return None
    
    async def _execute_trade(self, opportunity: TradingOpportunity):
        """Execute a trade based on opportunity."""
        try:
            signal = opportunity.signal
            
            if opportunity.recommended_stake < 10:  # Minimum bet size
                logger.debug(f"Skipping small bet: ${opportunity.recommended_stake:.2f}")
                return
            
            # Create paper order
            order = PaperOrder(
                order_id=f"LIVE_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(timezone.utc),
                source_id=signal.market_id,
                market_type="winner",
                bet_name=signal.recommended_position,
                side="buy",
                size=opportunity.recommended_stake,
                limit_price=opportunity.current_odds.get(signal.recommended_position, 2.0),
                signal_name="blockchain_live_strategy",
                expected_edge=signal.expected_edge
            )
            
            # Execute order
            fill = await self.paper_engine.submit_order(order)
            
            if fill:
                logger.info(f"   ✅ Trade executed: {opportunity.market.home_team} vs {opportunity.market.away_team}")
                logger.info(f"      Position: {signal.recommended_position}")
                logger.info(f"      Stake: ${opportunity.recommended_stake:.2f}")
                logger.info(f"      Odds: {opportunity.current_odds.get(signal.recommended_position, 'N/A')}")
                logger.info(f"      Reasoning: {signal.reasoning}")
                
                # Track active position
                self.active_positions[signal.market_id] = {
                    'order': order,
                    'fill': fill,
                    'opportunity': opportunity,
                    'entry_time': datetime.now(timezone.utc)
                }
            else:
                logger.warning(f"   ❌ Trade failed: {opportunity.market.home_team} vs {opportunity.market.away_team}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _monitor_positions(self):
        """Monitor existing positions and manage risk."""
        # For now, just log position status
        # In a real system, this would check for closing opportunities
        if self.active_positions:
            logger.info(f"   📊 Monitoring {len(self.active_positions)} active positions")
    
    def _update_performance(self):
        """Update and log performance metrics."""
        try:
            metrics = self.paper_engine.calculate_performance()
            current_value = self.paper_engine.get_portfolio_value()
            
            performance = {
                'timestamp': datetime.now(timezone.utc),
                'capital': self.paper_engine.current_capital,
                'portfolio_value': current_value,
                'total_trades': metrics['total_trades'],
                'total_pnl': metrics['total_pnl'],
                'win_rate': metrics['win_rate']
            }
            
            self.performance_history.append(performance)
            
            # Log every 10th update
            if len(self.performance_history) % 10 == 0:
                logger.info(f"   💰 Portfolio Value: ${current_value:,.2f}")
                logger.info(f"   📈 Total P&L: ${metrics['total_pnl']:,.2f}")
                logger.info(f"   🎯 Win Rate: {metrics['win_rate']:.1%}")
        
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _generate_final_report(self):
        """Generate final performance report."""
        logger.info("\n" + "="*60)
        logger.info("📊 LIVE TRADING STRATEGY FINAL REPORT")
        logger.info("="*60)
        
        try:
            final_metrics = self.paper_engine.calculate_performance()
            final_value = self.paper_engine.get_portfolio_value()
            
            logger.info(f"Initial Capital: ${self.paper_engine.initial_capital:,.2f}")
            logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
            logger.info(f"Total Return: ${final_value - self.paper_engine.initial_capital:,.2f}")
            logger.info(f"Return %: {((final_value / self.paper_engine.initial_capital) - 1) * 100:.2f}%")
            logger.info(f"Total Trades: {final_metrics['total_trades']}")
            logger.info(f"Win Rate: {final_metrics['win_rate']:.1%}")
            logger.info(f"Total Commission: ${final_metrics['total_commission']:.2f}")
            
            # Position summary
            positions = self.paper_engine.get_open_positions()
            logger.info(f"Open Positions: {len(positions)}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")


async def main():
    """Run the live blockchain trading strategy."""
    strategy = LiveTradingStrategy(initial_capital=5000)
    
    # Run for 2 hours as a demo
    await strategy.run_strategy(duration_hours=2)


if __name__ == "__main__":
    asyncio.run(main())