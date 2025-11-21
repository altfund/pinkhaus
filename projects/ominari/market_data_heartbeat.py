#!/usr/bin/env python3
"""
Market Data Heartbeat - Real-time Mark-to-Market Updates
Part of the robust Ominari heartbeat ecosystem
Updates portfolio values based on live odds changes every 5 minutes
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from paper_trading_sessions import PaperTradingSessionManager
from unified_portfolio_calculator import UnifiedPortfolioCalculator
from notifications.discord_notifier import discord_notifier
from sqlalchemy import func, and_, desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataHeartbeat:
    """Fast heartbeat for real-time mark-to-market portfolio updates"""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.portfolio_calculator = UnifiedPortfolioCalculator()
        self.update_interval = 300  # 5 minutes for mark-to-market
        self.notification_interval = 6  # Notify every 6 updates (30 minutes)
        self.update_count = 0
        self.last_portfolio_value = None
        self.significant_change_threshold = 0.005  # 0.5% change triggers notification
        
    def get_live_position_values(self) -> Dict:
        """Get current mark-to-market values for all open positions"""
        try:
            current_session = self.session_manager.get_current_session()
            if not current_session or not current_session.get('positions'):
                return {'total_mtm_value': 0, 'position_updates': []}
            
            position_updates = []
            total_mtm_value = 0
            
            with db_manager.get_db_session() as db:
                for pos_key, position in current_session.get('positions', {}).items():
                    if position.get('status') != 'open':
                        continue
                    
                    market_name = position.get('market_name', '')
                    outcome = position.get('outcome', 'home')
                    stake = position.get('total_stake', 0)  # Fixed: use total_stake instead of stake
                    entry_odds = position.get('avg_odds', 1.0)  # Fixed: use avg_odds instead of odds
                    
                    try:
                        # Find market by team name (approximate match)
                        team_name = market_name.split(' vs ')[0][:10]
                        market = db.query(Market).filter(
                            Market.home_team.ilike(f"%{team_name}%")
                        ).order_by(Market.maturity_date.desc()).first()
                        
                        if market:
                            # Get latest odds for this outcome
                            latest_odd = db.query(Odd).filter(
                                Odd.source_id == market.source_id,
                                Odd.outcome == outcome
                            ).order_by(Odd.updated_at.desc()).first()
                            
                            if latest_odd:
                                current_odds = latest_odd.decimal_odds
                                
                                # Calculate mark-to-market value
                                # Current position value = stake * current_odds (what we'd get if we won now)
                                current_position_value = stake * current_odds
                                entry_position_value = stake * entry_odds
                                mtm_change = current_position_value - entry_position_value
                                
                                position_updates.append({
                                    'market_name': market_name,
                                    'outcome': outcome,
                                    'stake': stake,
                                    'entry_odds': entry_odds,
                                    'current_odds': current_odds,
                                    'mtm_value': current_position_value,
                                    'mtm_change': mtm_change,
                                    'odds_movement': ((current_odds - entry_odds) / entry_odds) * 100
                                })
                                
                                total_mtm_value += current_position_value
                                
                            else:
                                logger.warning(f"No current odds found for {market_name} {outcome}")
                        else:
                            logger.warning(f"Market not found for position: {market_name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing position {pos_key}: {e}")
                        continue
            
            return {
                'total_mtm_value': total_mtm_value,
                'position_updates': position_updates,
                'positions_count': len(position_updates)
            }
            
        except Exception as e:
            logger.error(f"Error getting live position values: {e}")
            return {'total_mtm_value': 0, 'position_updates': [], 'error': str(e)}
    
    def calculate_updated_portfolio_value(self, mtm_data: Dict) -> Dict:
        """Calculate updated portfolio value with mark-to-market adjustments"""
        try:
            # Get base portfolio metrics
            metrics = self.portfolio_calculator.get_current_portfolio_metrics(force_reload=True)
            
            # Get current cash (bankroll not at risk)
            current_cash = metrics.current_bankroll
            
            # Add mark-to-market value of open positions
            mtm_value = mtm_data.get('total_mtm_value', 0)
            
            # Updated portfolio value = cash + mark-to-market value of positions
            updated_portfolio_value = current_cash + mtm_value
            
            # For consistency with main heartbeat, use the unified calculator's portfolio_value
            # when no positions are open (should equal current_cash)
            if mtm_data.get('positions_count', 0) == 0:
                # No positions: use the main heartbeat's calculation for consistency
                base_portfolio_value = metrics.portfolio_value
                updated_portfolio_value = metrics.portfolio_value  # Use same as main heartbeat
                mtm_adjustment = 0  # No adjustment when no positions
            else:
                # Active positions: use mark-to-market calculation
                base_portfolio_value = current_cash
                mtm_adjustment = mtm_value
            
            return {
                'base_portfolio_value': base_portfolio_value,
                'updated_portfolio_value': updated_portfolio_value,
                'current_cash': current_cash,
                'mtm_value': mtm_value,
                'mtm_adjustment': mtm_adjustment,
                'total_pnl': updated_portfolio_value - metrics.initial_bankroll,
                'roi_percentage': ((updated_portfolio_value - metrics.initial_bankroll) / metrics.initial_bankroll) * 100,
                'positions_count': mtm_data.get('positions_count', 0),
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating updated portfolio value: {e}")
            return {'error': str(e)}
    
    def check_significant_change(self, new_value: float) -> bool:
        """Check if portfolio value change is significant enough for notification"""
        if self.last_portfolio_value is None:
            self.last_portfolio_value = new_value
            return True
        
        change_pct = abs((new_value - self.last_portfolio_value) / self.last_portfolio_value)
        if change_pct >= self.significant_change_threshold:
            self.last_portfolio_value = new_value
            return True
        
        return False
    
    def create_mtm_notification(self, portfolio_data: Dict, mtm_data: Dict) -> Dict:
        """Create Discord notification for significant mark-to-market changes"""
        now = datetime.now(timezone.utc)
        
        updated_value = portfolio_data.get('updated_portfolio_value', 0)
        mtm_adjustment = portfolio_data.get('mtm_adjustment', 0)
        positions_count = portfolio_data.get('positions_count', 0)
        roi_pct = portfolio_data.get('roi_percentage', 0)
        
        # Determine color based on change
        color = 0x4B9CD3  # Default blue
        if mtm_adjustment > 0:
            color = 0x28a745  # Green for gains
        elif mtm_adjustment < 0:
            color = 0xffc107  # Yellow for losses
        
        embed = {
            "title": "📊 Mark-to-Market Update",
            "description": f"Live portfolio valuation with current odds",
            "color": color,
            "timestamp": now.isoformat(),
            "fields": []
        }
        
        # Portfolio summary
        main_text = f"**Portfolio**: ${updated_value:,.2f} ({roi_pct:+.1f}%)\n"
        if mtm_adjustment != 0:
            main_text += f"**Change**: {mtm_adjustment:+.2f} (mark-to-market)\n"
        main_text += f"**Positions**: {positions_count} active"
        
        embed["fields"].append({
            "name": "💰 Current Valuation",
            "value": main_text,
            "inline": False
        })
        
        # Position details if any significant moves
        if mtm_data.get('position_updates'):
            significant_moves = [pos for pos in mtm_data['position_updates'] 
                               if abs(pos.get('odds_movement', 0)) > 5]  # >5% odds movement
            
            if significant_moves:
                moves_text = ""
                for pos in significant_moves[:3]:  # Show top 3 moves
                    odds_change = pos.get('odds_movement', 0)
                    move_emoji = "📈" if odds_change > 0 else "📉"
                    moves_text += f"{move_emoji} {pos.get('market_name', 'Unknown')[:20]}...\n"
                    moves_text += f"   {pos.get('entry_odds', 0):.2f} → {pos.get('current_odds', 0):.2f} ({odds_change:+.1f}%)\n"
                
                if moves_text:
                    embed["fields"].append({
                        "name": "📈 Significant Moves",
                        "value": moves_text,
                        "inline": True
                    })
        
        embed["footer"] = {
            "text": f"Mark-to-Market Update • Next check in 5 minutes"
        }
        
        return embed
    
    async def send_mtm_update(self):
        """Send mark-to-market update"""
        try:
            logger.info("📊 Running mark-to-market update...")
            
            # Get live position values
            mtm_data = self.get_live_position_values()
            
            # Calculate updated portfolio value
            portfolio_data = self.calculate_updated_portfolio_value(mtm_data)
            
            if 'error' in portfolio_data:
                logger.error(f"Portfolio calculation error: {portfolio_data['error']}")
                return
            
            updated_value = portfolio_data.get('updated_portfolio_value', 0)
            mtm_adjustment = portfolio_data.get('mtm_adjustment', 0)
            positions_count = portfolio_data.get('positions_count', 0)
            
            logger.info(f"📊 Portfolio: ${updated_value:,.2f} | MTM Adj: {mtm_adjustment:+.2f} | Positions: {positions_count}")
            
            # Check if we should send notification
            should_notify = (
                self.update_count % self.notification_interval == 0 or  # Regular interval
                self.check_significant_change(updated_value)  # Significant change
            )
            
            # For zero positions, only notify once per hour to avoid spam
            if positions_count == 0 and self.update_count % 12 == 0:  # Every 12 updates (1 hour) when no positions
                should_notify = True
            
            if should_notify and discord_notifier.enabled:
                embed = self.create_mtm_notification(portfolio_data, mtm_data)
                discord_notifier.send_embed(embed)
                logger.info("📨 Mark-to-market notification sent")
            
            self.update_count += 1
            
        except Exception as e:
            logger.error(f"Error in mark-to-market update: {e}")
    
    async def start_mtm_loop(self):
        """Start the mark-to-market monitoring loop"""
        logger.info("📊 Starting mark-to-market heartbeat system...")
        logger.info(f"🕐 Update interval: {self.update_interval} seconds")
        logger.info(f"📢 Notification interval: Every {self.notification_interval} updates")
        
        while True:
            try:
                await self.send_mtm_update()
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Mark-to-market heartbeat stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in mark-to-market loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

async def main():
    """Main entry point for the mark-to-market heartbeat system"""
    mtm_heartbeat = MarketDataHeartbeat()
    await mtm_heartbeat.start_mtm_loop()

if __name__ == "__main__":
    print("📊 Market Data Heartbeat - Mark-to-Market Monitor")
    print("=" * 55)
    print("Features:")
    print("• Real-time mark-to-market portfolio valuation")
    print("• Live odds tracking for open positions") 
    print("• 5-minute update intervals")
    print("• Smart notifications for significant changes")
    print("• Integration with existing heartbeat ecosystem")
    print()
    
    asyncio.run(main())