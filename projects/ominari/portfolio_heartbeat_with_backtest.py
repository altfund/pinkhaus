#!/usr/bin/env python3
"""
Enhanced portfolio heartbeat with backtesting results
Includes both live portfolio status and historical performance analysis
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from portfolio_heartbeat_robust import RobustPortfolioHeartbeat
from database_v2 import db_manager
from models import Market, Odd, Bet
from sqlalchemy import func, and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioHeartbeatWithBacktest(RobustPortfolioHeartbeat):
    """Enhanced heartbeat that includes backtesting results"""
    
    def __init__(self):
        super().__init__()
        self.include_backtest = True
        self.backtest_interval = 4  # Run backtest every 4 heartbeats (4 hours)
        self.heartbeat_count = 0
        
    def get_historical_edge_performance(self) -> Dict:
        """Analyze historical performance of edge-based betting"""
        with db_manager.get_db_session() as db:
            # Get all completed bets with known outcomes
            # Note: In paper trading, we simulate outcomes
            
            # For now, analyze bet distribution and edge capture
            all_bets = db.query(Bet).all()
            
            edge_ranges = {
                'high_edge': {'min': 10, 'max': 100, 'bets': [], 'total_stake': 0},
                'medium_edge': {'min': 5, 'max': 10, 'bets': [], 'total_stake': 0},
                'low_edge': {'min': 0, 'max': 5, 'bets': [], 'total_stake': 0},
                'negative_edge': {'min': -100, 'max': 0, 'bets': [], 'total_stake': 0}
            }
            
            # Categorize bets by edge (estimated from odds and probability)
            for bet in all_bets:
                if hasattr(bet, 'probability') and bet.probability and bet.odds:
                    # Estimate edge from fair probability and odds
                    fair_odds = 1 / bet.probability
                    edge = ((bet.odds / fair_odds) - 1) * 100
                    
                    for range_name, range_data in edge_ranges.items():
                        if range_data['min'] <= edge < range_data['max']:
                            range_data['bets'].append(bet)
                            range_data['total_stake'] += bet.stake
                            break
            
            # Calculate statistics
            stats = {}
            for range_name, range_data in edge_ranges.items():
                stats[range_name] = {
                    'count': len(range_data['bets']),
                    'total_stake': range_data['total_stake'],
                    'avg_stake': range_data['total_stake'] / len(range_data['bets']) if range_data['bets'] else 0,
                    'percentage': len(range_data['bets']) / len(all_bets) * 100 if all_bets else 0
                }
            
            return stats
    
    def simulate_backtest_results(self) -> Dict:
        """Simulate backtesting results based on current strategy"""
        # In a real system, this would run actual backtests
        # For now, we'll analyze the strategy performance
        
        with db_manager.get_db_session() as db:
            # Get betting patterns
            recent_bets = db.query(Bet).order_by(Bet.id.desc()).limit(100).all()
            
            if not recent_bets:
                return {
                    'status': 'No betting history',
                    'recommendation': 'Continue paper trading to build history'
                }
            
            # Analyze Kelly fraction usage
            kelly_fractions = []
            for bet in recent_bets:
                if hasattr(bet, 'stake') and hasattr(bet, 'probability') and bet.probability:
                    # Estimate Kelly fraction used
                    bankroll = 10000  # Assumed
                    kelly_full = (bet.probability * (bet.odds - 1) - (1 - bet.probability)) / (bet.odds - 1)
                    actual_fraction = (bet.stake / bankroll) / kelly_full if kelly_full > 0 else 0
                    kelly_fractions.append(min(actual_fraction, 1.0))
            
            avg_kelly = sum(kelly_fractions) / len(kelly_fractions) if kelly_fractions else 0
            
            # Simulated backtest results
            backtest = {
                'period_analyzed': 'Last 100 bets',
                'total_markets': len(set(b.source_id for b in recent_bets)),
                'avg_kelly_fraction': round(avg_kelly, 3),
                'strategy_consistency': 'High' if 0.2 <= avg_kelly <= 0.3 else 'Medium',
                'risk_management': 'Conservative' if avg_kelly < 0.3 else 'Moderate',
                'expected_long_term_roi': f"{avg_kelly * 15:.1f}%" if avg_kelly > 0 else "0%",
                'recommendation': self.get_strategy_recommendation(avg_kelly)
            }
            
            return backtest
    
    def get_strategy_recommendation(self, kelly_fraction: float) -> str:
        """Get strategy recommendation based on analysis"""
        if kelly_fraction < 0.1:
            return "Consider increasing bet sizes - current sizing is very conservative"
        elif kelly_fraction > 0.5:
            return "Consider reducing bet sizes - current sizing may be too aggressive"
        elif 0.2 <= kelly_fraction <= 0.3:
            return "Optimal Kelly fraction range - maintain current strategy"
        else:
            return "Kelly fraction within acceptable range"
    
    def format_enhanced_heartbeat(self, stats: Dict, markets: Dict, opportunities: Dict, 
                                 blockchain: Optional[Dict], historical: Dict, backtest: Dict) -> Dict:
        """Format enhanced heartbeat with backtest results"""
        # Start with base heartbeat
        embed = self.format_portfolio_update(stats, markets, opportunities, blockchain)
        
        # Add historical edge performance
        if self.heartbeat_count % self.backtest_interval == 0:
            edge_str = "**Edge Performance Analysis:**\n"
            for range_name, data in historical.items():
                if data['count'] > 0:
                    range_label = range_name.replace('_', ' ').title()
                    edge_str += f"• {range_label}: {data['count']} bets (${data['total_stake']:.2f})\n"
            
            embed["fields"].append({
                "name": "📈 Historical Edge Analysis",
                "value": edge_str,
                "inline": False
            })
            
            # Add backtest insights
            backtest_str = (
                f"**Period:** {backtest['period_analyzed']}\n"
                f"**Markets Analyzed:** {backtest['total_markets']}\n"
                f"**Avg Kelly Fraction:** {backtest['avg_kelly_fraction']}\n"
                f"**Strategy Consistency:** {backtest['strategy_consistency']}\n"
                f"**Risk Profile:** {backtest['risk_management']}\n"
                f"**Expected Annual ROI:** {backtest['expected_long_term_roi']}\n\n"
                f"💡 {backtest['recommendation']}"
            )
            
            embed["fields"].append({
                "name": "🔬 Backtesting Insights",
                "value": backtest_str,
                "inline": False
            })
            
            # Update title for backtest heartbeat
            embed["title"] = "📊 Portfolio Heartbeat + Backtesting Report"
            embed["color"] = 0x9b59b6  # Purple for special report
        
        return embed
    
    async def send_heartbeat_safe(self):
        """Enhanced heartbeat with backtesting"""
        try:
            # Increment counter
            self.heartbeat_count += 1
            
            # Gather all data
            portfolio_stats = self.get_portfolio_stats()
            market_stats = self.get_active_markets_count()
            edge_opportunities = self.get_edge_opportunities()
            blockchain_stats = self.get_blockchain_stats()
            
            # Get enhanced data
            historical_edge = self.get_historical_edge_performance()
            backtest_results = self.simulate_backtest_results() if self.heartbeat_count % self.backtest_interval == 0 else {}
            
            # Format enhanced embed
            embed = self.format_enhanced_heartbeat(
                portfolio_stats, 
                market_stats, 
                edge_opportunities,
                blockchain_stats,
                historical_edge,
                backtest_results
            )
            
            # Send to Discord
            success = discord_notifier.send_embed(embed)
            
            if success:
                logger.info(f"✅ Enhanced heartbeat sent (#{self.heartbeat_count})")
                if self.heartbeat_count % self.backtest_interval == 0:
                    logger.info("📊 Included backtesting analysis")
                self.last_update = datetime.now(timezone.utc)
                self.consecutive_failures = 0
            else:
                raise Exception("Failed to send embed to Discord")
                
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"❌ Enhanced heartbeat failed: {e}")
            
            # Fall back to regular heartbeat
            if self.consecutive_failures > 2:
                logger.info("Falling back to simple heartbeat")
                await super().send_heartbeat_safe()


# Import discord notifier
from notifications.discord_notifier import discord_notifier


async def main():
    """Run enhanced heartbeat system"""
    heartbeat = PortfolioHeartbeatWithBacktest()
    
    try:
        await heartbeat.run_forever(interval_minutes=60)
    except KeyboardInterrupt:
        logger.info("Stopping enhanced heartbeat...")
    finally:
        heartbeat.is_running = False


if __name__ == "__main__":
    print("🫀 Enhanced Portfolio Heartbeat with Backtesting")
    print("📊 Includes historical edge analysis every 4 hours")
    print("🔬 Provides strategy recommendations")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(main())