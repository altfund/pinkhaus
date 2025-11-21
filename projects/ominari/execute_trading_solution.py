#!/usr/bin/env python3
"""
Trading Execution Solution
Integrates the fixed edge calculation with actual trade execution
"""

import os
import sys
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from paper_trading_sessions import PaperTradingSessionManager
from fixed_edge_calculation import FixedEdgeSignalProvider
from database_v2 import db_manager
from models import Market, Odd
from notifications.discord_notifier import discord_notifier
from trading_costs import RealisticTradingCostCalculator
from conservative_edge_calculator import ConservativeEdgeCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedEdgeTradingExecutor:
    """Trading executor using fixed edge calculation."""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.edge_provider = FixedEdgeSignalProvider()
        self.cost_calculator = RealisticTradingCostCalculator()
        self.conservative_calculator = ConservativeEdgeCalculator()
        self.trading_cycle_minutes = 15  # Trade every 15 minutes
        self.max_trades_per_cycle = 3
        self.running = False
        self.min_conservative_edge = 2.0  # Minimum 2% conservative edge required
        
    def get_or_create_session(self) -> str:
        """Get active trading session - find or create session with trades."""
        current_session = self.session_manager.get_current_session()
        
        # Use existing session with trades if available
        if current_session and current_session.get('positions'):
            logger.info(f"Using existing session {current_session['session_id']} with {len(current_session.get('positions', {}))} positions")
            return current_session['session_id']
        
        # Look for most recent session with actual trades
        all_sessions_dict = self.session_manager.sessions.get("sessions", {})
        
        # Find session with positions or trades, sorted by created_at
        sessions_with_trades = []
        for session_id, session_data in all_sessions_dict.items():
            if session_data.get('positions') or session_data.get('trades'):
                sessions_with_trades.append((session_id, session_data))
        
        # Sort by created_at and get most recent
        if sessions_with_trades:
            sessions_with_trades.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)
            session_id = sessions_with_trades[0][0]
            logger.info(f"Found existing trading session: {session_id}")
            return session_id
        
        # Create new session only if no active trading session exists
        logger.info("No active trading session found, creating new one")
        return self.session_manager.create_session(10000)
    
    def find_tradeable_markets(self):
        """Find markets suitable for trading."""
        try:
            with db_manager.get_db_session() as db:
                now = datetime.now(timezone.utc)
                # Convert to naive datetime for database comparison
                now_naive = now.replace(tzinfo=None)
                
                # Get upcoming markets in next 24 hours
                markets = db.query(Market).filter(
                    Market.maturity_date > now_naive + timedelta(minutes=30),  # At least 30 min future
                    Market.maturity_date < now_naive + timedelta(hours=24),   # Within 24 hours
                    Market.sport == 'Soccer',
                    Market.is_finished == False
                ).limit(20).all()
                
                tradeable_markets = []
                
                for market in markets:
                    # Get odds
                    odds_records = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(10).all()
                    
                    if len(odds_records) >= 3:
                        odds_data = {}
                        for odd in odds_records:
                            if odd.outcome not in odds_data:
                                odds_data[odd.outcome] = odd.decimal_odds
                        
                        if len(odds_data) >= 3:  # Home, away, draw
                            tradeable_markets.append({
                                'market': market,
                                'odds_data': odds_data
                            })
                
                return tradeable_markets
                
        except Exception as e:
            logger.error(f"Error finding tradeable markets: {e}")
            return []
    
    def filter_signals_with_conservative_edge(self, signals: dict, market, odds_data) -> dict:
        """Filter signals using conservative edge calculation."""
        # Get conservative edge metrics
        conservative_edges = self.conservative_calculator.calculate_conservative_edge(market, odds_data)
        
        filtered_signals = {}
        
        for signal_key, signal_data in signals.items():
            outcome = signal_data.get('outcome')
            raw_edge = signal_data.get('edge', 0)
            
            # Find matching conservative edge
            conservative_edge_data = None
            for edge_outcome, edge_metrics in conservative_edges.items():
                if edge_outcome.lower() == outcome.lower():
                    conservative_edge_data = edge_metrics
                    break
            
            if conservative_edge_data:
                conservative_edge = conservative_edge_data.final_conservative_edge
                
                # Only trade if conservative edge meets minimum threshold
                if conservative_edge >= self.min_conservative_edge:
                    # Update signal with conservative metrics
                    filtered_signal = signal_data.copy()
                    filtered_signal.update({
                        'raw_edge': raw_edge,
                        'conservative_edge': conservative_edge,
                        'edge': conservative_edge,  # Use conservative edge
                        'market_efficiency': conservative_edge_data.market_efficiency_factor,
                        'uncertainty_discount': conservative_edge_data.uncertainty_discount,
                        'competition_factor': conservative_edge_data.competition_factor,
                        'confidence_score': conservative_edge_data.confidence_score
                    })
                    
                    filtered_signals[signal_key] = filtered_signal
                    
                    logger.info(f"✅ Conservative Signal: {market.home_team} vs {market.away_team} - {outcome} @ {signal_data.get('market_odds', 0):.2f} = {conservative_edge:+.2f}% edge (was {raw_edge:+.2f}%)")
                else:
                    logger.info(f"❌ Signal filtered: {market.home_team} vs {market.away_team} - {outcome} conservative edge {conservative_edge:+.2f}% < {self.min_conservative_edge}% threshold")
        
        return filtered_signals
    
    def execute_trades_for_signals(self, signals: dict, market, odds_data):
        """Execute actual trades for qualifying signals with dynamic position management."""
        trades = []
        
        # Get current session to check existing positions
        session = self.session_manager.get_current_session()
        existing_positions = session.get('positions', {}) if session else {}
        
        for signal_key, signal_data in signals.items():
            edge = signal_data.get('edge', 0)
            confidence = signal_data.get('confidence', 0)
            outcome = signal_data.get('outcome')
            market_odds = signal_data.get('market_odds', 3.0)
            forecast = signal_data.get('forecast', 0.33)
            
            # Calculate optimal position size using Kelly
            edge_decimal = edge / 100
            kelly_fraction = edge_decimal / (market_odds - 1) if market_odds > 1 else 0
            safe_kelly = kelly_fraction * 0.25  # 25% Kelly
            
            # Get current bankroll
            current_bankroll = session.get('current_bankroll', 10000) if session else 10000
            
            # Calculate optimal stake
            theoretical_stake = current_bankroll * safe_kelly
            max_stake = min(200, current_bankroll * 0.02)  # 2% max position
            optimal_stake = min(theoretical_stake, max_stake)
            
            # Check if we already have a position for this market + outcome
            position_key = f"{market.source_id}_{outcome}"
            existing_position = existing_positions.get(position_key)
            
            if existing_position:
                current_stake = existing_position.get('total_stake', 0)
                current_odds = existing_position.get('avg_odds', market_odds)
                
                # Calculate position difference
                stake_difference = optimal_stake - current_stake
                rebalance_threshold = current_stake * 0.15  # 15% rebalance threshold
                
                # Check if we should rebalance (adjust position)
                if abs(stake_difference) > max(rebalance_threshold, 10):  # Min $10 change
                    if stake_difference > 0:
                        # Increase position size
                        additional_stake = stake_difference
                        base_trade = {
                            'market_id': market.source_id,
                            'market_name': f"{market.home_team} vs {market.away_team}",
                            'outcome': outcome,
                            'stake': additional_stake,
                            'odds': market_odds,
                            'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                            'signal_type': 'position_increase',
                            'edge': edge,
                            'confidence': confidence,
                            'forecast': forecast,
                            'kelly_fraction': kelly_fraction,
                            'position_action': 'increase',
                            'existing_stake': current_stake,
                            'new_total_stake': current_stake + additional_stake
                        }
                        
                        # Apply realistic trading costs to increase (with sport context)
                        trade = self.cost_calculator.apply_costs_to_trade(
                            base_trade, {'sport': 'soccer'}
                        )
                        
                        effective_odds = trade['effective_odds']
                        cost_impact = trade['fee_info']['total_fee']
                        effective_stake = trade['fee_info']['execution_stake']
                        
                        trades.append(trade)
                        logger.info(f"📈 INCREASING POSITION (with costs):")
                        logger.info(f"  Market: {trade['market_name']}")
                        logger.info(f"  Bet: {outcome.upper()} @ {market_odds:.3f} → {effective_odds:.3f}")
                        logger.info(f"  Edge: {edge:+.2f}%")
                        logger.info(f"  Current Stake: ${current_stake:.2f}")
                        logger.info(f"  Additional: ${additional_stake:.2f} + ${cost_impact:.2f} fees = ${effective_stake:.2f}")
                        logger.info(f"  New Total: ${current_stake + additional_stake:.2f}")
                        
                    # Note: We could implement position reduction here too, but for simplicity
                    # in paper trading, we'll only increase positions when edge improves
                else:
                    logger.info(f"⚖️ Position within rebalance threshold: {market.home_team} vs {market.away_team} - {outcome.upper()}")
                    logger.info(f"  Current: ${current_stake:.2f}, Optimal: ${optimal_stake:.2f}, Diff: ${stake_difference:.2f}")
            else:
                # No existing position - create new one
                if optimal_stake >= 10:  # Min bet $10
                    # Create base trade
                    base_trade = {
                        'market_id': market.source_id,
                        'market_name': f"{market.home_team} vs {market.away_team}",
                        'outcome': outcome,
                        'stake': optimal_stake,
                        'odds': market_odds,
                        'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                        'signal_type': 'new_position',
                        'edge': edge,
                        'confidence': confidence,
                        'forecast': forecast,
                        'kelly_fraction': kelly_fraction,
                        'position_action': 'new'
                    }
                    
                    # Apply realistic trading costs (with sport context)
                    trade = self.cost_calculator.apply_costs_to_trade(
                        base_trade, {'sport': 'soccer'}
                    )
                    
                    # Recalculate edge after costs (important for Kelly sizing)
                    effective_odds = trade['effective_odds']
                    cost_impact = trade['fee_info']['total_fee']
                    effective_stake = trade['fee_info']['execution_stake']
                    
                    # Adjusted Kelly edge with costs
                    if effective_odds > 1.0:
                        adjusted_edge = ((forecast * effective_odds - 1) / (effective_odds - 1)) * 100
                        trade['adjusted_edge'] = adjusted_edge
                    else:
                        trade['adjusted_edge'] = edge
                    
                    trades.append(trade)
                    logger.info(f"💰 NEW POSITION (with costs):")
                    logger.info(f"  Market: {trade['market_name']}")
                    logger.info(f"  Bet: {outcome.upper()} @ {market_odds:.3f} → {effective_odds:.3f} (after slippage)")
                    logger.info(f"  Original Edge: {edge:+.2f}% → Adjusted: {trade.get('adjusted_edge', edge):+.2f}%")
                    logger.info(f"  Stake: ${optimal_stake:.2f} + ${cost_impact:.2f} fees = ${effective_stake:.2f}")
                    logger.info(f"  Fee Breakdown: {trade['fee_info']['fee_breakdown']}")
                    logger.info(f"  Kelly: {kelly_fraction:.4f}")
        
        return trades
    
    async def run_trading_cycle(self):
        """Run one trading cycle."""
        try:
            logger.info(f"🔄 Running Trading Cycle - {datetime.now().strftime('%H:%M:%S')}")
            
            # Reload sessions from disk to sync with heartbeat system
            self.session_manager.sessions = self.session_manager._load_sessions()
            
            # Get session
            session_id = self.get_or_create_session()
            
            # Find tradeable markets
            markets = self.find_tradeable_markets()
            logger.info(f"📊 Found {len(markets)} tradeable markets")
            
            if not markets:
                logger.info("No tradeable markets found")
                return
            
            total_trades = 0
            trades_executed = []
            
            for market_data in markets:
                if total_trades >= self.max_trades_per_cycle:
                    break
                    
                market = market_data['market']
                odds_data = market_data['odds_data']
                
                # Generate signals
                signals = self.edge_provider.generate_signals_for_market(market, odds_data)
                
                if signals:
                    # Apply conservative edge filtering
                    conservative_signals = self.filter_signals_with_conservative_edge(signals, market, odds_data)
                    
                    if conservative_signals:
                        # Execute trades
                        trades = self.execute_trades_for_signals(conservative_signals, market, odds_data)
                        
                        if trades:
                            # Record trades in session
                            success = self.session_manager.record_trades(session_id, trades)
                            
                            if success:
                                trades_executed.extend(trades)
                                total_trades += len(trades)
                                logger.info(f"✅ Recorded {len(trades)} trades for {market.home_team} vs {market.away_team}")
                            else:
                                logger.error(f"❌ Failed to record trades for {market.home_team} vs {market.away_team}")
            
            # Report results with costs
            if trades_executed:
                total_stake = sum(t['stake'] for t in trades_executed)
                total_fees = sum(t.get('fee_info', {}).get('total_fee', 0) for t in trades_executed)
                total_execution_cost = sum(t.get('fee_info', {}).get('execution_stake', t['stake']) for t in trades_executed)
                avg_edge = sum(t['edge'] for t in trades_executed) / len(trades_executed)
                
                # Calculate adjusted edge after costs
                adjusted_edges = [t.get('adjusted_edge', t['edge']) for t in trades_executed if t.get('adjusted_edge')]
                avg_adjusted_edge = sum(adjusted_edges) / len(adjusted_edges) if adjusted_edges else avg_edge
                
                logger.info(f"🎯 CYCLE COMPLETE:")
                logger.info(f"  Trades Executed: {len(trades_executed)}")
                logger.info(f"  Total Stake: ${total_stake:.2f}")
                logger.info(f"  Total Fees: ${total_fees:.2f} ({total_fees/total_stake*100:.1f}%)")
                logger.info(f"  Total Cost: ${total_execution_cost:.2f}")
                logger.info(f"  Average Edge: {avg_edge:+.2f}% → {avg_adjusted_edge:+.2f}% (after costs)")
                
                # Send Discord notification
                embed = {
                    "title": "💰 Trades Executed",
                    "description": f"Fixed edge trading system executed {len(trades_executed)} trades",
                    "color": 0x00ff00,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fields": [
                        {
                            "name": "Execution Summary",
                            "value": f"**Trades**: {len(trades_executed)}\n"
                                   f"**Total Stake**: ${total_stake:.2f}\n"
                                   f"**Total Fees**: ${total_fees:.2f} ({total_fees/total_stake*100:.1f}%)\n"
                                   f"**Total Cost**: ${total_execution_cost:.2f}\n"
                                   f"**Avg Edge**: {avg_edge:+.2f}% → {avg_adjusted_edge:+.2f}%",
                            "inline": False
                        }
                    ]
                }
                
                try:
                    discord_notifier.send_embed(embed)
                except:
                    pass
                    
            else:
                logger.info("No qualifying trades found this cycle")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def start_trading_loop(self):
        """Start the continuous trading loop."""
        self.running = True
        logger.info("🚀 Starting Fixed Edge Trading Executor")
        logger.info(f"⏱️ Trading cycle: Every {self.trading_cycle_minutes} minutes")
        logger.info(f"🎯 Max trades per cycle: {self.max_trades_per_cycle}")
        
        # Send startup notification
        embed = {
            "title": "🚀 Trading Executor Started",
            "description": "Fixed edge trading system is now executing trades",
            "color": 0x00ff00,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            discord_notifier.send_embed(embed)
        except:
            pass
        
        while self.running:
            try:
                await self.run_trading_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.trading_cycle_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

async def main():
    """Main entry point."""
    executor = FixedEdgeTradingExecutor()
    await executor.start_trading_loop()

if __name__ == "__main__":
    print("💰 Fixed Edge Trading Executor")
    print("=" * 40)
    print("Features:")
    print("• Uses fixed intrinsic probability models")
    print("• Executes actual paper trades")
    print("• 15-minute trading cycles") 
    print("• Position sizing with Kelly + Thorp limits")
    print("• Discord trade notifications")
    print()
    
    asyncio.run(main())