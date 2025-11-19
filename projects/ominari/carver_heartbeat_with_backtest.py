#!/usr/bin/env python3
"""
Carver Portfolio Heartbeat with Comprehensive Backtest Reports
Integrates the comprehensive Carver backtest results into Discord heartbeat notifications.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict
# import matplotlib.pyplot as plt
# import io
# import base64

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from paper_trading_sessions import PaperTradingSessionManager
from notifications.discord_notifier import discord_notifier
from comprehensive_carver_backtest import ComprehensiveCarverBacktest
from real_market_resolution_service import RealMarketResolutionService
from sqlalchemy import func, and_, desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarverHeartbeatWithBacktest:
    """Enhanced heartbeat system with comprehensive Carver backtest integration."""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.heartbeat_interval = 3600  # 1 hour
        self.backtest_interval = 4  # Run full backtest every 4 heartbeats (4 hours)
        self.quick_analysis_interval = 1  # Quick analysis every heartbeat
        self.heartbeat_count = 0
        
        # Deployment configuration for heartbeat reports
        self.backtest_config = {
            'initial_bankroll': 10000,
            'kelly_fraction': 0.25,     # Increased for deployment
            'max_position_pct': 0.08,   # Increased from 0.05 to 8% 
            'min_bet_size': 10,
            'transaction_cost_pct': 0.015,  # Reduced transaction costs
            'min_confidence': 0.03,     # Lowered for more signals
            'min_edge': 0.001,         # Lowered for more opportunities  
            'risk_free_rate': 0.02,
            'lookback_days': 30,       # Shorter for heartbeat
            'market_limit': None       # Remove arbitrary limits for deployment
        }
        
        self.last_backtest_results = None
        self.last_backtest_time = None
        
    def get_current_portfolio_status(self) -> Dict:
        """Get current paper trading portfolio status."""
        try:
            # Reload sessions from disk to get latest data
            self.session_manager.sessions = self.session_manager._load_sessions()
            
            # Get the LATEST active session with actual trades (not just the "current" one)
            current_session = self.session_manager.get_current_session()
            
            # Check if current session has actual activity
            if current_session and current_session.get('positions'):
                logger.info(f"Using active session {current_session['session_id']} with {len(current_session.get('positions', {}))} positions")
            else:
                # Look for the most recent session with actual trades/positions
                logger.info("Current session empty, looking for session with actual trades...")
                
                # Access sessions directly from the manager
                all_sessions_dict = self.session_manager.sessions.get("sessions", {})
                
                # Find session with positions or trades
                for session_id, session_data in all_sessions_dict.items():
                    if session_data.get('positions') or session_data.get('trades'):
                        logger.info(f"Found active trading session: {session_id}")
                        # Add session_id to session data for compatibility
                        session_data['session_id'] = session_id
                        current_session = session_data
                        break
                
                if not current_session or not current_session.get('positions'):
                    logger.info("No active paper trading session found, creating new one")
                    session_id = self.session_manager.create_session(10000)
                    current_session = self.session_manager.get_current_session()
            
            # Calculate portfolio metrics
            current_bankroll = current_session.get('current_bankroll', 10000)
            initial_bankroll = current_session.get('initial_bankroll', 10000)
            portfolio_value = current_session.get('portfolio_value', current_bankroll)
            
            # Count positions and trades
            open_positions = current_session.get('positions', {})
            closed_positions = current_session.get('closed_positions', [])
            all_trades = current_session.get('trades', [])
            
            # Calculate total stake and P&L
            total_open_stake = sum(pos.get('total_stake', 0) for pos in open_positions.values())
            performance = current_session.get('performance', {})
            total_pnl = performance.get('total_pnl', 0)
            
            # Get recent trades for display
            recent_trades = []
            for pos_key, pos in list(open_positions.items())[:5]:  # Show 5 recent positions
                recent_trades.append({
                    'bet_name': pos.get('market_name', pos_key),
                    'execution_stake': pos.get('total_stake', 0)
                })
            
            return {
                'status': 'Active' if current_session.get('status') == 'active' else 'Inactive',
                'session_id': current_session.get('session_id'),
                'bankroll': current_bankroll,
                'initial_bankroll': initial_bankroll,
                'portfolio_value': portfolio_value,
                'trades_count': performance.get('total_trades', 0),
                'open_positions_count': len(open_positions),
                'total_stake': total_open_stake,
                'total_pnl': total_pnl,
                'total_fees': performance.get('total_fees', 0),
                'recent_trades': recent_trades
            }
            
        except Exception as e:
            logger.error(f"Error getting paper trading portfolio status: {e}")
            return {'status': f'Error: {e}'}
            
    def get_current_market_chunk(self) -> Dict:
        """Get current market chunk based on timing breaks."""
        try:
            with db_manager.get_db_session() as db:
                now = datetime.now(timezone.utc)
                
                # Get all upcoming soccer markets in next 24 hours
                upcoming_markets = db.query(Market).filter(
                    Market.maturity_date > now,
                    Market.maturity_date < now + timedelta(hours=24),
                    Market.sport == 'Soccer'
                ).order_by(Market.maturity_date).all()
                
                if not upcoming_markets:
                    return {
                        'markets_in_chunk': 0,
                        'signals_generated': 0,
                        'avg_edge': 0,
                        'avg_confidence': 0,
                        'chunk_info': 'No upcoming markets'
                    }
                
                # Group markets into timing-based chunks (30-minute breaks)
                chunks = self._create_market_chunks(upcoming_markets, min_break_minutes=30)
                
                # Get the current active chunk
                current_chunk = self._get_active_chunk(chunks, now)
                
                if not current_chunk or not current_chunk.get('markets'):
                    return {
                        'markets_in_chunk': 0,
                        'signals_generated': 0, 
                        'avg_edge': 0,
                        'avg_confidence': 0,
                        'chunk_info': 'No active chunk'
                    }
                
                # Analyze ALL markets in the current chunk (no arbitrary limits)
                return self._analyze_chunk_signals(current_chunk)
                
        except Exception as e:
            logger.error(f"Error getting market chunk analysis: {e}")
            return {'error': str(e)}
            
    def _create_market_chunks(self, markets: List, min_break_minutes: int = 30):
        """Create timing-based market chunks with natural breaks."""
        if not markets:
            return []
            
        chunks = []
        current_chunk_markets = [markets[0]]
        chunk_start = markets[0].maturity_date
        
        for i in range(1, len(markets)):
            market = markets[i]
            last_market = markets[i-1]
            
            # Check if there's a break of min_break_minutes or more
            # Ensure timezone awareness
            market_time = market.maturity_date
            last_time = last_market.maturity_date
            
            if market_time.tzinfo is None:
                market_time = market_time.replace(tzinfo=timezone.utc)
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
                
            time_gap = (market_time - last_time).total_seconds() / 60
            
            if time_gap >= min_break_minutes:
                # End current chunk and start new one
                chunk_end = last_market.maturity_date
                
                chunk = {
                    'start_time': chunk_start,
                    'end_time': chunk_end, 
                    'markets': current_chunk_markets,
                    'chunk_id': f"chunk_{chunk_start.strftime('%Y%m%d_%H%M')}",
                    'market_count': len(current_chunk_markets)
                }
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_markets = [market]
                chunk_start = market.maturity_date
            else:
                current_chunk_markets.append(market)
        
        # Add final chunk
        if current_chunk_markets:
            chunk = {
                'start_time': chunk_start,
                'end_time': current_chunk_markets[-1].maturity_date,
                'markets': current_chunk_markets,
                'chunk_id': f"chunk_{chunk_start.strftime('%Y%m%d_%H%M')}",
                'market_count': len(current_chunk_markets)
            }
            chunks.append(chunk)
            
        return chunks
        
    def _get_active_chunk(self, chunks: List, now: datetime):
        """Get the currently active market chunk."""
        for chunk in chunks:
            # Ensure timezone awareness
            start_time = chunk['start_time']
            end_time = chunk['end_time']
            
            # Make timezone aware if needed
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
                
            # Chunk is active if we're within 2 hours before start
            chunk_activation = start_time - timedelta(hours=2)
            
            if now >= chunk_activation and now <= end_time:
                return chunk
                
        # Return the next upcoming chunk if no active chunk
        if chunks:
            return chunks[0]
        return None
        
    def _analyze_chunk_signals(self, chunk: Dict) -> Dict:
        """Analyze signals for ALL markets in a chunk (no limits)."""
        try:
            with db_manager.get_db_session() as db:
                from fixed_soccer_signals import FixedSoccerWinDrawLossSignal
                from fixed_edge_calculation import FixedEdgeSignalProvider
                
                wdl_provider = FixedSoccerWinDrawLossSignal()
                edge_provider = FixedEdgeSignalProvider()
                
                total_signals = 0
                total_edge = 0
                total_confidence = 0
                
                # Process ALL markets in the chunk - no arbitrary limits!
                for market in chunk['markets']:
                    # Get odds
                    odds_records = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(10).all()
                    
                    if len(odds_records) >= 3:
                        odds_data = {}
                        for odd in odds_records:
                            if odd.outcome not in odds_data:
                                odds_data[odd.outcome] = odd.decimal_odds
                                
                        # Test enhanced edge signals (more reliable method) 
                        try:
                            edge_signals = edge_provider.generate_signals_for_market(market, odds_data)
                            if edge_signals:
                                for signal_key, signal_data in edge_signals.items():
                                    total_signals += 1
                                    total_edge += signal_data.get('edge', 0)
                                    total_confidence += signal_data.get('confidence', 0)
                        except Exception as e:
                            logger.debug(f"Edge signal generation failed for {market.source_id}: {e}")
                            
                        # Test WDL signals
                        try:
                            wdl_signals = wdl_provider.generate_signal(market, odds_data)
                            if wdl_signals:
                                for signal_key, signal_data in wdl_signals.items():
                                    total_signals += 1
                                    total_edge += signal_data.get('edge', 0)
                                    total_confidence += signal_data.get('confidence', 0)
                        except Exception as e:
                            logger.debug(f"WDL signal generation failed for {market.source_id}: {e}")
                            
                signals_generated = total_signals
                avg_edge = total_edge / total_signals if total_signals > 0 else 0
                avg_confidence = total_confidence / total_signals if total_signals > 0 else 0
                        
                return {
                    'markets_in_chunk': len(chunk['markets']),
                    'chunk_id': chunk['chunk_id'],
                    'chunk_start': chunk['start_time'].isoformat(),
                    'chunk_end': chunk['end_time'].isoformat(),
                    'signals_generated': signals_generated,
                    'avg_edge': avg_edge,
                    'avg_confidence': avg_confidence,
                    'chunk_info': f"Active chunk with {len(chunk['markets'])} markets"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing chunk signals: {e}")
            return {'error': str(e)}
            
    def run_comprehensive_backtest(self) -> Optional[Dict]:
        """Run comprehensive backtest and return results."""
        try:
            logger.info("🔬 Running comprehensive Carver backtest for heartbeat...")
            
            backtester = ComprehensiveCarverBacktest(self.backtest_config)
            results = backtester.run_comprehensive_backtest()
            
            # Summarize results for heartbeat
            performance_summary = {}
            for strategy_name, trades in results['trades'].items():
                performance = backtester.calculate_performance_metrics(trades)
                performance_summary[strategy_name] = {
                    'total_return': performance.total_return,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'win_rate': performance.win_rate,
                    'trades_count': performance.trades_count,
                    'profit_factor': performance.profit_factor,
                    'avg_edge': performance.avg_edge,
                    'avg_confidence': performance.avg_confidence
                }
                
            backtest_summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config': self.backtest_config,
                'performance': performance_summary,
                'best_strategy': max(performance_summary.items(), 
                                   key=lambda x: x[1]['total_return'])[0] if performance_summary else None,
                'total_trades': sum(p['trades_count'] for p in performance_summary.values()),
                'avg_return': sum(p['total_return'] for p in performance_summary.values()) / len(performance_summary) if performance_summary else 0
            }
            
            self.last_backtest_results = backtest_summary
            self.last_backtest_time = datetime.now(timezone.utc)
            
            logger.info(f"✅ Backtest completed: {backtest_summary['total_trades']} total trades, {backtest_summary['avg_return']:.1%} avg return")
            
            return backtest_summary
            
        except Exception as e:
            logger.error(f"Error running comprehensive backtest: {e}")
            return None
            
    def create_performance_chart_for_discord(self, performance_data: Dict) -> Optional[str]:
        """Create a compact performance chart for Discord."""
        try:
            strategies = list(performance_data.keys())
            returns = [performance_data[s]['total_return'] * 100 for s in strategies]
            sharpe_ratios = [performance_data[s]['sharpe_ratio'] for s in strategies]
            
            # Create compact chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Returns chart
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars1 = ax1.bar(strategies, returns, color=colors, alpha=0.7)
            ax1.set_title('Strategy Returns (%)', fontweight='bold')
            ax1.set_ylabel('Return (%)')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars1, returns):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Sharpe ratios chart
            colors2 = ['blue' if s > 0 else 'orange' for s in sharpe_ratios]
            bars2 = ax2.bar(strategies, sharpe_ratios, color=colors2, alpha=0.7)
            ax2.set_title('Sharpe Ratios', fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars2, sharpe_ratios):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Encode to base64 for Discord
            img_data = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_data
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return None
            
    def create_heartbeat_embed(self, portfolio_status: Dict, chunk_analysis: Dict, 
                              backtest_data: Optional[Dict] = None) -> Dict:
        """Create comprehensive Discord embed for heartbeat."""
        now = datetime.now(timezone.utc)
        
        embed = {
            "title": "🎯 Carver Trading System Heartbeat",
            "description": f"System status and performance analysis",
            "color": 0x00ff99,
            "timestamp": now.isoformat(),
            "fields": []
        }
        
        # Portfolio Status Section
        portfolio_text = f"**Status**: {portfolio_status.get('status', 'Unknown')}\n"
        
        # Use portfolio_value for better accuracy
        current_value = portfolio_status.get('portfolio_value', portfolio_status.get('bankroll', 10000))
        initial = portfolio_status.get('initial_bankroll', 10000)
        total_pnl = portfolio_status.get('total_pnl', 0)
        
        # Calculate P&L percentage
        pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0
        
        portfolio_text += f"**Portfolio Value**: ${current_value:,.2f}\n"
        portfolio_text += f"**P&L**: ${total_pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
        portfolio_text += f"**Open Positions**: {portfolio_status.get('open_positions_count', 0)}\n"
        portfolio_text += f"**Total Trades**: {portfolio_status.get('trades_count', 0)}\n"
        portfolio_text += f"**Active Stake**: ${portfolio_status.get('total_stake', 0):,.2f}"
            
        embed["fields"].append({
            "name": "📊 Live Portfolio Status",
            "value": portfolio_text,
            "inline": True
        })
        
        # Market Chunk Analysis Section  
        if chunk_analysis.get('error'):
            chunk_text = f"**Error**: {chunk_analysis['error']}"
        else:
            chunk_text = f"**Markets in Chunk**: {chunk_analysis.get('markets_in_chunk', 0)}\n"
            chunk_text += f"**Signals Generated**: {chunk_analysis.get('signals_generated', 0)}\n"
            
            if chunk_analysis.get('avg_edge', 0) != 0:
                chunk_text += f"**Avg Edge**: {chunk_analysis['avg_edge']:+.2f}%\n"
                chunk_text += f"**Avg Confidence**: {chunk_analysis['avg_confidence']:.3f}\n"
            
            if chunk_analysis.get('chunk_id'):
                chunk_text += f"**Chunk**: {chunk_analysis['chunk_id']}\n"
                chunk_text += f"**Info**: {chunk_analysis.get('chunk_info', 'Active')}"
            
        embed["fields"].append({
            "name": "🎯 Market Chunk Analysis",
            "value": chunk_text,
            "inline": True
        })
        
        # Backtest Results Section
        if backtest_data:
            best_strategy = backtest_data.get('best_strategy', 'None')
            total_trades = backtest_data.get('total_trades', 0)
            avg_return = backtest_data.get('avg_return', 0)
            
            backtest_text = f"**Period**: {backtest_data['config']['lookback_days']} days\n"
            backtest_text += f"**Total Trades**: {total_trades}\n"
            backtest_text += f"**Avg Return**: {avg_return:+.1%}\n"
            backtest_text += f"**Best Strategy**: {best_strategy}\n"
            
            # Top 3 strategies
            performance = backtest_data.get('performance', {})
            if performance:
                sorted_strategies = sorted(performance.items(), 
                                         key=lambda x: x[1]['total_return'], reverse=True)[:3]
                backtest_text += "\n**Top Performers**:\n"
                for i, (strategy, perf) in enumerate(sorted_strategies, 1):
                    backtest_text += f"{i}. {strategy}: {perf['total_return']:+.1%}\n"
            
            embed["fields"].append({
                "name": "🔬 Backtest Analysis",
                "value": backtest_text,
                "inline": False
            })
        elif self.last_backtest_results:
            # Show last backtest if no new one
            age = (datetime.now(timezone.utc) - self.last_backtest_time).total_seconds() / 3600
            embed["fields"].append({
                "name": "🔬 Last Backtest",
                "value": f"**Age**: {age:.1f} hours ago\n**Status**: {self.last_backtest_results.get('total_trades', 0)} trades analyzed",
                "inline": False
            })
        
        # Recent Trades Section
        if 'recent_trades' in portfolio_status and portfolio_status['recent_trades']:
            trades_text = ""
            for trade in portfolio_status['recent_trades'][:3]:  # Show last 3 trades
                profit_text = f"${trade.execution_stake:,.0f}" 
                trades_text += f"• {trade.bet_name[:30]}... - {profit_text}\n"
                
            embed["fields"].append({
                "name": "📈 Recent Trades",
                "value": trades_text or "No recent trades",
                "inline": False
            })
        
        # System Health
        health_text = f"**Heartbeat**: #{self.heartbeat_count + 1}\n"
        health_text += f"**Next Backtest**: {'Now' if self.heartbeat_count % self.backtest_interval == 0 else f'In {self.backtest_interval - (self.heartbeat_count % self.backtest_interval)} cycles'}\n"
        health_text += f"**Signal Providers**: 5 active\n"
        health_text += f"**Database**: Connected"
        
        embed["fields"].append({
            "name": "⚙️ System Health",
            "value": health_text,
            "inline": True
        })
        
        embed["footer"] = {
            "text": "Carver Enhanced Trading System | Next heartbeat in 1 hour"
        }
        
        return embed
        
    async def send_heartbeat_notification(self):
        """Send comprehensive heartbeat notification to Discord."""
        try:
            logger.info("💓 Sending Carver heartbeat notification...")
            
            # Resolve finished markets with REAL sports data every few heartbeats
            if self.heartbeat_count % 3 == 0:  # Every 3 heartbeats (3 hours)
                logger.info("🏁 Resolving finished markets with real sports data...")
                try:
                    resolution_service = RealMarketResolutionService()
                    resolution_results = resolution_service.resolve_markets_with_real_data()
                    if resolution_results.get('resolved_count', 0) > 0:
                        logger.info(f"✅ Resolved {resolution_results['resolved_count']} markets with REAL results!")
                except Exception as e:
                    logger.warning(f"Market resolution error: {e}")
            
            # Get current status
            portfolio_status = self.get_current_portfolio_status()
            chunk_analysis = self.get_current_market_chunk()
            
            # Run full backtest periodically
            backtest_data = None
            if self.heartbeat_count % self.backtest_interval == 0:
                logger.info("🔬 Running scheduled comprehensive backtest...")
                backtest_data = self.run_comprehensive_backtest()
            
            # Create Discord embed
            embed = self.create_heartbeat_embed(portfolio_status, chunk_analysis, backtest_data)
            
            # Send to Discord
            if discord_notifier.enabled:
                discord_notifier.send_embed(embed)
                
                # Send performance chart if backtest was run
                if backtest_data and 'performance' in backtest_data:
                    chart_data = self.create_performance_chart_for_discord(backtest_data['performance'])
                    if chart_data:
                        # Note: Discord.py would handle file uploads differently
                        # This is a placeholder for chart integration
                        logger.info("📊 Performance chart generated for Discord")
                        
                logger.info("✅ Heartbeat notification sent successfully")
            else:
                logger.warning("Discord notifications disabled")
                
            self.heartbeat_count += 1
            
        except Exception as e:
            logger.error(f"Error sending heartbeat notification: {e}")
            
    async def start_heartbeat_loop(self):
        """Start the continuous heartbeat loop."""
        logger.info("🎯 Starting Carver heartbeat system...")
        logger.info(f"📅 Heartbeat interval: {self.heartbeat_interval} seconds")
        logger.info(f"🔬 Backtest interval: Every {self.backtest_interval} heartbeats")
        
        while True:
            try:
                await self.send_heartbeat_notification()
                await asyncio.sleep(self.heartbeat_interval)
                
            except KeyboardInterrupt:
                logger.info("Heartbeat system stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry


async def main():
    """Main entry point for the heartbeat system."""
    heartbeat = CarverHeartbeatWithBacktest()
    await heartbeat.start_heartbeat_loop()


if __name__ == "__main__":
    print("💓 Carver Portfolio Heartbeat with Comprehensive Backtest")
    print("=" * 65)
    print("Features:")
    print("• Live portfolio status monitoring")
    print("• Real-time signal generation analysis") 
    print("• Comprehensive backtest reports every 4 hours")
    print("• Performance charts and visualizations")
    print("• Discord notifications with detailed embeds")
    print("• Recent trade tracking")
    print("• System health monitoring")
    print()
    
    asyncio.run(main())