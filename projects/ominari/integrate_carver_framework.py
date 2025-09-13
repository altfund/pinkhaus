#!/usr/bin/env python3
"""
Integration of Carver systematic trading framework with Ominari system.
"""

import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import Dict, List

from carver_framework import (
    CarverTradingSystem, 
    TradingRule, 
    Instrument,
    create_carver_system
)
from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from paper_trading_engine import PaperTradingEngine
from sqlalchemy import and_, func

logger = logging.getLogger(__name__)


class OminariCarverAdapter:
    """Adapts Carver framework to work with Ominari betting system."""
    
    def __init__(self, capital: float = 10000):
        self.system = create_carver_system(capital=capital)
        self.paper_trader = None
        self.active_markets = {}
        
    def load_active_markets(self):
        """Load currently active betting markets."""
        with db_manager.get_db_session() as db:
            # Get markets closing in next 24 hours
            now = datetime.now(timezone.utc)
            tomorrow = now + timedelta(days=1)
            
            markets = db.query(Market).filter(
                and_(
                    Market.maturity_date > now,
                    Market.maturity_date < tomorrow,
                    Market.is_finished == False
                )
            ).all()
            
            logger.info(f"Loaded {len(markets)} active markets")
            
            # Convert to Carver instruments
            for market in markets:
                instrument = Instrument(
                    market_id=market.source_id,
                    sport=market.sport,
                    league=market.league_name or market.sport,
                    volatility_target=0.16  # 16% default
                )
                self.system.add_instrument(instrument)
                self.active_markets[market.source_id] = market
                
    def fetch_market_data(self, market_id: str, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch historical odds data for a market."""
        with db_manager.get_db_session() as db:
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            
            # Get odds history
            odds_query = db.query(
                Odd.updated_at,
                func.min(Odd.decimal_odds).label('best_odds'),
                func.avg(Odd.decimal_odds).label('avg_odds'),
                func.count(Odd.id).label('n_bookmakers'),
                func.avg(Odd.normalized_implied).label('avg_implied')
            ).filter(
                and_(
                    Odd.source_id == market_id,
                    Odd.outcome == 'option_1',  # Home team
                    Odd.updated_at >= since
                )
            ).group_by(
                func.date_trunc('hour', Odd.updated_at)  # Hourly aggregation
            ).order_by(
                Odd.updated_at
            ).all()
            
            if not odds_query:
                return pd.DataFrame()
            
            # Convert to dataframe
            data = []
            for row in odds_query:
                data.append({
                    'timestamp': row.updated_at,
                    'close': row.avg_odds,
                    'best_odds': row.best_odds,
                    'volume': row.n_bookmakers,  # Proxy for volume
                    'implied_prob': row.avg_implied,
                    'spread': row.avg_odds - row.best_odds if row.best_odds else 0
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df = df.set_index('timestamp').sort_index()
                
                # Add technical features
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(24).std()  # 24hr vol
                df['momentum'] = df['returns'].rolling(12).mean()  # 12hr momentum
                
            return df
    
    def generate_signals(self) -> Dict[str, float]:
        """Generate Carver-style position signals for all markets."""
        signals = {}
        
        # Prepare market data for all instruments
        market_data = {}
        for market_id in self.active_markets:
            data = self.fetch_market_data(market_id)
            if not data.empty:
                market_data[market_id] = data
        
        if not market_data:
            logger.warning("No market data available")
            return signals
        
        # Run Carver system
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)
        
        try:
            results = self.system.run_system(
                market_data,
                start,
                now
            )
            
            # Extract final positions
            if 'positions' in results:
                final_positions = results['positions'].iloc[-1]
                
                for market_id, position in final_positions.items():
                    if abs(position) > 0.01:  # Minimum position threshold
                        signals[market_id] = position
                        
            logger.info(f"Generated {len(signals)} signals")
            
        except Exception as e:
            logger.error(f"Error running Carver system: {e}")
            
        return signals
    
    def execute_paper_trades(self, signals: Dict[str, float]):
        """Execute paper trades based on Carver signals."""
        if not self.paper_trader:
            from paper_trading_sessions import SessionManager
            session_manager = SessionManager()
            session = session_manager.get_or_create_session(
                session_name=f"Carver_{datetime.now().strftime('%Y%m%d')}",
                initial_bankroll=self.system.capital
            )
            self.paper_trader = PaperTradingEngine(
                db_manager=db_manager,
                session_id=session['session_id']
            )
        
        # Convert Carver positions to bet sizes
        total_capital = self.system.capital
        
        for market_id, position in signals.items():
            market = self.active_markets.get(market_id)
            if not market:
                continue
                
            # Position sizing: position * capital * volatility_target
            instrument = self.system.instruments.get(market_id)
            if not instrument:
                continue
                
            # Calculate bet size
            bet_size = abs(position) * total_capital * instrument.volatility_target
            
            # Limit to max 5% per bet
            bet_size = min(bet_size, total_capital * 0.05)
            
            if bet_size < 10:  # Minimum bet size
                continue
                
            # Determine bet direction
            if position > 0:
                outcome = 'option_1'  # Bet on home team
            else:
                outcome = 'option_2'  # Bet on away team
                
            logger.info(f"Placing bet: {market.home_team} vs {market.away_team}, "
                       f"Outcome: {outcome}, Size: ${bet_size:.2f}")
            
            # Execute paper trade
            self.paper_trader.place_bet(
                market_id=market_id,
                outcome=outcome,
                stake=bet_size,
                odds=2.0,  # Placeholder - would get actual odds
                strategy_name='carver_systematic'
            )
    
    def generate_performance_report(self):
        """Generate performance report for Carver strategy."""
        if not hasattr(self, 'system') or not self.system.performance_history:
            return "No performance data available"
            
        perf = self.system.performance_history[-1]
        
        report = f"""
Carver Systematic Trading Report
================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Capital: ${self.system.capital:,.2f}

Performance Metrics:
- Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}
- Volatility: {perf.get('volatility', 0):.2%}
- Max Drawdown: {perf.get('max_drawdown', 0):.2%}
- Turnover: {perf.get('turnover', 0):.1f}x annual

Position Summary:
- Active Markets: {len(self.active_markets)}
- Signals Generated: {perf.get('n_signals', 0)}
- Forecast Correlation: {perf.get('forecast_correlation', 0):.2f}
- Diversification Multiplier: {perf.get('div_multiplier', 1.0):.2f}

Trading Rules Performance:
"""
        
        # Add individual rule performance
        if 'rule_performance' in perf:
            for rule, metrics in perf['rule_performance'].items():
                report += f"\n{rule}:"
                report += f"\n  - Sharpe: {metrics.get('sharpe', 0):.2f}"
                report += f"\n  - Weight: {metrics.get('weight', 0):.1%}"
                
        return report


def run_carver_integration():
    """Run the integrated Carver system."""
    logger.info("Starting Carver framework integration...")
    
    # Initialize adapter
    adapter = OminariCarverAdapter(capital=10000)
    
    # Load active markets
    adapter.load_active_markets()
    
    if not adapter.active_markets:
        logger.warning("No active markets found")
        return
    
    # Generate signals
    signals = adapter.generate_signals()
    
    if not signals:
        logger.info("No trading signals generated")
        return
        
    logger.info(f"Generated signals for {len(signals)} markets:")
    for market_id, position in signals.items():
        market = adapter.active_markets.get(market_id)
        if market:
            logger.info(f"  {market.home_team} vs {market.away_team}: {position:+.3f}")
    
    # Execute paper trades
    adapter.execute_paper_trades(signals)
    
    # Generate report
    report = adapter.generate_performance_report()
    print(report)
    
    # Save report
    with open('carver_integration_report.txt', 'w') as f:
        f.write(report)
        f.write("\n\nDetailed Signals:\n")
        for market_id, position in signals.items():
            market = adapter.active_markets.get(market_id)
            if market:
                f.write(f"\n{market.home_team} vs {market.away_team}:")
                f.write(f"\n  Position: {position:+.3f}")
                f.write(f"\n  Sport: {market.sport}")
                f.write(f"\n  Maturity: {market.maturity_date}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_carver_integration()