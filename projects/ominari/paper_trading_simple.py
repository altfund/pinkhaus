#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Paper Trading System
A working implementation that demonstrates paper trading functionality.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePaperTradingEngine:
    """Simplified paper trading engine."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
    def place_bet(self, market_id: str, team: str, amount: float, odds: float, confidence: float):
        """Place a paper bet."""
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': datetime.now(timezone.utc),
            'market_id': market_id,
            'team': team,
            'amount': amount,
            'odds': odds,
            'confidence': confidence,
            'status': 'open',
            'pnl': 0.0
        }
        
        # Check if we have enough capital
        if amount <= self.capital:
            self.capital -= amount
            self.trades.append(trade)
            self.performance['total_trades'] += 1
            logger.info(f"📝 Placed bet: {team} for ${amount:.2f} at {odds:.2f} odds (confidence: {confidence:.2%})")
            return trade
        else:
            logger.warning(f"Insufficient capital: ${self.capital:.2f} < ${amount:.2f}")
            return None
            
    def simulate_results(self):
        """Simulate bet results based on confidence."""
        for trade in self.trades:
            if trade['status'] == 'open':
                # Simulate win/loss based on confidence
                win_probability = trade['confidence']
                won = np.random.random() < win_probability
                
                if won:
                    pnl = trade['amount'] * (trade['odds'] - 1)
                    self.capital += trade['amount'] + pnl
                    trade['status'] = 'won'
                    trade['pnl'] = pnl
                    self.performance['winning_trades'] += 1
                    logger.info(f"✅ Won bet #{trade['id']}: +${pnl:.2f}")
                else:
                    pnl = -trade['amount']
                    trade['status'] = 'lost'
                    trade['pnl'] = pnl
                    self.performance['losing_trades'] += 1
                    logger.info(f"❌ Lost bet #{trade['id']}: ${pnl:.2f}")
                    
                self.performance['total_pnl'] += pnl
                
        # Update win rate
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = self.performance['winning_trades'] / self.performance['total_trades']
            
    def get_summary(self):
        """Get performance summary."""
        return {
            'capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            **self.performance
        }


class DataSimulator:
    """Simulates market data for testing."""
    
    @staticmethod
    def generate_markets():
        """Generate sample market data."""
        sports = ['NFL', 'NBA', 'EPL', 'MLB']
        markets = []
        
        for i in range(10):
            sport = np.random.choice(sports)
            markets.append({
                'id': f'market_{i}',
                'sport': sport,
                'home_team': f'{sport}_Home_{i}',
                'away_team': f'{sport}_Away_{i}',
                'home_odds': np.random.uniform(1.5, 3.0),
                'away_odds': np.random.uniform(1.5, 3.0),
                'timestamp': datetime.now(timezone.utc)
            })
            
        return markets
        
    @staticmethod
    def generate_signals(markets):
        """Generate trading signals."""
        signals = {}
        
        for market in markets:
            # Simple signal based on odds
            home_prob = 1 / market['home_odds']
            away_prob = 1 / market['away_odds']
            
            # Normalize
            total = home_prob + away_prob
            home_prob /= total
            away_prob /= total
            
            # Add some noise
            home_prob += np.random.normal(0, 0.1)
            home_prob = max(0.1, min(0.9, home_prob))
            
            signals[market['id']] = {
                'home_confidence': home_prob,
                'away_confidence': 1 - home_prob
            }
            
        return signals


async def run_paper_trading_demo():
    """Run a paper trading demonstration."""
    logger.info("🚀 Starting Paper Trading Demo")
    logger.info("="*60)
    
    # Initialize engine
    engine = SimplePaperTradingEngine(initial_capital=10000.0)
    data_sim = DataSimulator()
    
    # Run for 5 cycles
    for cycle in range(5):
        logger.info(f"\n📊 Cycle {cycle + 1}/5")
        logger.info("-"*40)
        
        # Generate market data
        markets = data_sim.generate_markets()
        logger.info(f"Generated {len(markets)} markets")
        
        # Generate signals
        signals = data_sim.generate_signals(markets)
        
        # Make trading decisions
        trades_placed = 0
        for market in markets[:3]:  # Trade on first 3 markets
            signal = signals[market['id']]
            
            # Trade if high confidence
            if signal['home_confidence'] > 0.65:
                engine.place_bet(
                    market['id'],
                    market['home_team'],
                    100.0,  # $100 bets
                    market['home_odds'],
                    signal['home_confidence']
                )
                trades_placed += 1
            elif signal['away_confidence'] > 0.65:
                engine.place_bet(
                    market['id'],
                    market['away_team'],
                    100.0,
                    market['away_odds'],
                    signal['away_confidence']
                )
                trades_placed += 1
                
        logger.info(f"Placed {trades_placed} trades")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Simulate results
        engine.simulate_results()
        
        # Show current status
        summary = engine.get_summary()
        logger.info("\n💰 Current Status:")
        logger.info(f"  Capital: ${summary['capital']:,.2f}")
        logger.info(f"  Total Return: {summary['total_return']:.2%}")
        logger.info(f"  Win Rate: {summary['win_rate']:.2%}")
        logger.info(f"  Total P&L: ${summary['total_pnl']:,.2f}")
        
        await asyncio.sleep(3)
    
    # Final summary
    logger.info("\n"+"="*60)
    logger.info("📈 Final Results")
    logger.info("="*60)
    
    final = engine.get_summary()
    logger.info(f"Starting Capital: ${engine.initial_capital:,.2f}")
    logger.info(f"Ending Capital: ${final['capital']:,.2f}")
    logger.info(f"Total Return: {final['total_return']:.2%}")
    logger.info(f"Total Trades: {final['total_trades']}")
    logger.info(f"Winning Trades: {final['winning_trades']}")
    logger.info(f"Losing Trades: {final['losing_trades']}")
    logger.info(f"Win Rate: {final['win_rate']:.2%}")
    logger.info(f"Total P&L: ${final['total_pnl']:,.2f}")
    
    # Save results
    results_file = Path("paper_trading_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary': final,
            'trades': [
                {
                    'id': t['id'],
                    'timestamp': t['timestamp'].isoformat(),
                    'market_id': t['market_id'],
                    'team': t['team'],
                    'amount': t['amount'],
                    'odds': t['odds'],
                    'confidence': t['confidence'],
                    'status': t['status'],
                    'pnl': t['pnl']
                }
                for t in engine.trades
            ]
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(run_paper_trading_demo())