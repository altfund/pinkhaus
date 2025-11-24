#!/usr/bin/env python3
"""
Market Data Heartbeat - Portfolio & Odds Tracking
Part of the robust Ominari heartbeat ecosystem
Tracks live odds movements for open positions every 5 minutes
Uses correct betting accounting: Portfolio = Cash + Stakes
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

# Load environment variables first
from load_env import load_dotenv
load_dotenv()

# Use PostgreSQL like other services
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
    """Fast heartbeat for portfolio tracking and odds monitoring

    Uses correct betting accounting:
    - Portfolio Value = Cash + Stakes (not Cash + Potential Payouts)
    - Tracks odds movements for informational purposes only
    - No unrealized P&L until positions settle
    """

    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.portfolio_calculator = UnifiedPortfolioCalculator()
        self.update_interval = 300  # 5 minutes for odds updates
        self.notification_interval = 6  # Notify every 6 updates (30 minutes)
        self.update_count = 0
        self.last_portfolio_value = None
        self.significant_change_threshold = 0.005  # 0.5% change triggers notification
        
    def get_live_position_values(self) -> Dict:
        """Get current odds for all open positions and track movements"""
        try:
            current_session = self.session_manager.get_current_session()
            if not current_session or not current_session.get('positions'):
                return {'total_stakes': 0, 'position_updates': [], 'total_odds_shift': 0}

            position_updates = []
            total_stakes = 0
            total_odds_shift = 0

            with db_manager.get_db_session() as db:
                for pos_key, position in current_session.get('positions', {}).items():
                    if position.get('status') != 'open':
                        continue

                    market_name = position.get('market_name', '')
                    outcome = position.get('outcome', 'home')
                    stake = position.get('total_stake', 0)
                    entry_odds = position.get('avg_odds', 1.0)

                    # ALWAYS count this position's stake regardless of odds availability
                    total_stakes += stake

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

                                # Track odds movement for informational purposes
                                # Note: Position value = stake (not stake × odds)
                                # Odds movement shows how probability has shifted
                                entry_potential = stake * entry_odds
                                current_potential = stake * current_odds
                                potential_shift = current_potential - entry_potential

                                position_updates.append({
                                    'market_name': market_name,
                                    'outcome': outcome,
                                    'stake': stake,
                                    'entry_odds': entry_odds,
                                    'current_odds': current_odds,
                                    'entry_potential': entry_potential,
                                    'current_potential': current_potential,
                                    'potential_shift': potential_shift,
                                    'odds_movement': ((current_odds - entry_odds) / entry_odds) * 100
                                })

                                total_odds_shift += potential_shift

                            else:
                                logger.warning(f"No current odds found for {market_name} {outcome}")
                        else:
                            logger.warning(f"Market not found for position: {market_name}")

                    except Exception as e:
                        logger.error(f"Error processing position {pos_key}: {e}")
                        continue

            return {
                'total_stakes': total_stakes,
                'position_updates': position_updates,
                'positions_count': len(position_updates),
                'total_odds_shift': total_odds_shift
            }
            
        except Exception as e:
            logger.error(f"Error getting live position values: {e}")
            return {'total_stakes': 0, 'position_updates': [], 'total_odds_shift': 0, 'error': str(e)}
    
    def calculate_updated_portfolio_value(self, position_data: Dict) -> Dict:
        """Calculate correct portfolio value using proper betting accounting"""
        try:
            # Get base portfolio metrics
            metrics = self.portfolio_calculator.get_current_portfolio_metrics(force_reload=True)

            # Get current cash (bankroll not deployed)
            current_cash = metrics.current_bankroll

            # Get total stakes deployed
            total_stakes = position_data.get('total_stakes', 0)

            # Portfolio value = cash + stakes (NOT cash + potential payouts)
            # Stakes are assets until the bet settles
            portfolio_value = current_cash + total_stakes

            # Calculate realized P&L (only from settled trades)
            realized_pnl = portfolio_value - metrics.initial_bankroll

            # Get odds shift for informational purposes
            odds_shift = position_data.get('total_odds_shift', 0)
            positions_count = position_data.get('positions_count', 0)

            return {
                'portfolio_value': portfolio_value,
                'current_cash': current_cash,
                'total_stakes': total_stakes,
                'initial_bankroll': metrics.initial_bankroll,
                'realized_pnl': realized_pnl,
                'roi_percentage': (realized_pnl / metrics.initial_bankroll) * 100,
                'positions_count': positions_count,
                'odds_shift': odds_shift,  # Informational only
                'last_update': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
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
    
    def create_mtm_notification(self, portfolio_data: Dict, position_data: Dict) -> Dict:
        """Create Discord notification with odds tracking and correct portfolio value"""
        now = datetime.now(timezone.utc)

        portfolio_value = portfolio_data.get('portfolio_value', 0)
        current_cash = portfolio_data.get('current_cash', 0)
        total_stakes = portfolio_data.get('total_stakes', 0)
        positions_count = portfolio_data.get('positions_count', 0)
        roi_pct = portfolio_data.get('roi_percentage', 0)
        odds_shift = portfolio_data.get('odds_shift', 0)

        # Determine color based on odds shift (informational)
        color = 0x4B9CD3  # Default blue
        if odds_shift > 100:
            color = 0x28a745  # Green for favorable odds moves
        elif odds_shift < -100:
            color = 0xffc107  # Yellow for unfavorable odds moves

        embed = {
            "title": "📊 Portfolio & Odds Update",
            "description": f"Live odds tracking for open positions",
            "color": color,
            "timestamp": now.isoformat(),
            "fields": []
        }

        # Portfolio summary with correct accounting
        main_text = f"**Portfolio**: ${portfolio_value:,.2f} ({roi_pct:+.1f}%)\n"
        main_text += f"**Cash**: ${current_cash:,.2f}\n"
        main_text += f"**Stakes**: ${total_stakes:,.2f}\n"
        main_text += f"**Positions**: {positions_count} open"

        embed["fields"].append({
            "name": "💰 Portfolio Value",
            "value": main_text,
            "inline": False
        })

        # Odds movements (informational)
        if position_data.get('position_updates'):
            significant_moves = [pos for pos in position_data['position_updates']
                               if abs(pos.get('odds_movement', 0)) > 5]  # >5% odds movement

            if significant_moves:
                moves_text = ""
                for pos in significant_moves[:3]:  # Show top 3 moves
                    odds_change = pos.get('odds_movement', 0)
                    potential_shift = pos.get('potential_shift', 0)
                    move_emoji = "📈" if odds_change > 0 else "📉"
                    moves_text += f"{move_emoji} {pos.get('market_name', 'Unknown')[:25]}\n"
                    moves_text += f"   Odds: {pos.get('entry_odds', 0):.2f} → {pos.get('current_odds', 0):.2f} ({odds_change:+.1f}%)\n"

                if moves_text:
                    embed["fields"].append({
                        "name": "📊 Odds Movements",
                        "value": moves_text,
                        "inline": False
                    })

                    # Add note about odds movements
                    note_text = f"Total odds shift: ${odds_shift:+.2f}\n"
                    note_text += "*(Informational only - not added to portfolio)*"
                    embed["fields"].append({
                        "name": "ℹ️ Note",
                        "value": note_text,
                        "inline": False
                    })

        embed["footer"] = {
            "text": f"Odds Update • Portfolio value = cash + stakes"
        }

        return embed
    
    async def send_mtm_update(self):
        """Send portfolio and odds update"""
        try:
            logger.info("📊 Running portfolio and odds update...")

            # Get live position data and odds movements
            position_data = self.get_live_position_values()

            # Calculate correct portfolio value
            portfolio_data = self.calculate_updated_portfolio_value(position_data)

            if 'error' in portfolio_data:
                logger.error(f"Portfolio calculation error: {portfolio_data['error']}")
                return

            portfolio_value = portfolio_data.get('portfolio_value', 0)
            current_cash = portfolio_data.get('current_cash', 0)
            total_stakes = portfolio_data.get('total_stakes', 0)
            positions_count = portfolio_data.get('positions_count', 0)
            odds_shift = portfolio_data.get('odds_shift', 0)

            logger.info(f"📊 Portfolio: ${portfolio_value:,.2f} | Cash: ${current_cash:,.2f} | Stakes: ${total_stakes:,.2f} | Positions: {positions_count}")
            if odds_shift != 0:
                logger.info(f"   Odds shift: ${odds_shift:+.2f} (informational)")

            # Check if we should send notification
            should_notify = (
                self.update_count % self.notification_interval == 0 or  # Regular interval
                self.check_significant_change(portfolio_value)  # Significant change
            )

            # For zero positions, only notify once per hour to avoid spam
            if positions_count == 0 and self.update_count % 12 == 0:  # Every 12 updates (1 hour) when no positions
                should_notify = True

            if should_notify and discord_notifier.enabled:
                embed = self.create_mtm_notification(portfolio_data, position_data)
                discord_notifier.send_embed(embed)
                logger.info("📨 Portfolio & odds notification sent")

            self.update_count += 1

        except Exception as e:
            logger.error(f"Error in portfolio/odds update: {e}")
    
    async def start_mtm_loop(self):
        """Start the portfolio and odds monitoring loop"""
        logger.info("📊 Starting portfolio & odds tracking system...")
        logger.info(f"🕐 Update interval: {self.update_interval} seconds")
        logger.info(f"📢 Notification interval: Every {self.notification_interval} updates")
        logger.info(f"💰 Portfolio accounting: Value = Cash + Stakes")
        
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
    """Main entry point for the portfolio & odds tracking system"""
    mtm_heartbeat = MarketDataHeartbeat()
    await mtm_heartbeat.start_mtm_loop()

if __name__ == "__main__":
    print("📊 Market Data Heartbeat - Portfolio & Odds Tracker")
    print("=" * 55)
    print("Features:")
    print("• Correct portfolio accounting (Value = Cash + Stakes)")
    print("• Live odds tracking for open positions")
    print("• 5-minute update intervals")
    print("• Informational odds movement tracking")
    print("• Integration with existing heartbeat ecosystem")
    print()

    asyncio.run(main())