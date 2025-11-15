#!/usr/bin/env python3
"""
Discord notification system for Ominari trading bot
Sends trade alerts, backtest results, and daily summaries to Discord
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Handles Discord webhook notifications for trading events"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize Discord notifier with webhook URL"""
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.info("Discord notifications disabled - no webhook URL configured")
        else:
            logger.info("Discord notifications enabled")
            
    def _send_embed(self, embed: Dict) -> bool:
        """Send an embed message to Discord"""
        if not self.enabled:
            return False
            
        try:
            payload = {"embeds": [embed]}
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.debug("Discord notification sent successfully")
                return True
            else:
                logger.error(f"Discord notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False
            
    def send_startup_message(self):
        """Send a startup notification"""
        embed = {
            "title": "🚀 Ominari Trading Bot Started",
            "description": "Paper trading system is now online",
            "color": 0x00ff00,  # Green
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "Mode",
                    "value": "Paper Trading",
                    "inline": True
                },
                {
                    "name": "Status",
                    "value": "✅ Active",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Ominari Trading System"
            }
        }
        
        return self._send_embed(embed)
        
    def send_trade_alert(self, trade: Dict):
        """Send a trade alert"""
        # Determine color based on trade type/status
        if trade.get('type') == 'CLOSE':
            color = 0x00ff00 if trade.get('won') else 0xff0000  # Green if won, red if lost
            title = f"💰 Trade Closed: {'WON' if trade.get('won') else 'LOST'}"
        else:
            color = 0x0099ff  # Blue for new trades
            title = "📊 New Trade Placed"
            
        # Format PnL with proper sign
        pnl = trade.get('pnl', 0)
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        
        embed = {
            "title": title,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "Market",
                    "value": trade.get('market', 'Unknown'),
                    "inline": True
                },
                {
                    "name": "Outcome",
                    "value": trade.get('outcome', 'Unknown'),
                    "inline": True
                },
                {
                    "name": "Amount",
                    "value": f"${trade.get('amount', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Odds",
                    "value": f"{trade.get('odds', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Edge",
                    "value": f"{trade.get('edge', 0):.2f}%",
                    "inline": True
                },
                {
                    "name": "Bankroll",
                    "value": f"${trade.get('bankroll', 0):.2f}",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Ominari Paper Trading"
            }
        }
        
        # Add liquidity info if available
        if 'liquidity_info' in trade:
            embed["fields"].insert(3, {
                "name": "Liquidity",
                "value": trade['liquidity_info'],
                "inline": False
            })
        
        # Add PnL field for closed trades
        if trade.get('type') == 'CLOSE':
            embed["fields"].insert(3, {
                "name": "P&L",
                "value": pnl_str,
                "inline": True
            })
            
        return self._send_embed(embed)
        
    def send_backtest_results(self, results: Dict):
        """Send backtest results summary"""
        # Determine color based on ROI
        roi = results.get('roi', 0)
        if roi > 5:
            color = 0x00ff00  # Green
        elif roi > 0:
            color = 0xffff00  # Yellow
        else:
            color = 0xff0000  # Red
            
        embed = {
            "title": "📈 Backtest Results",
            "description": f"Strategy: {results.get('strategy', 'Unknown')}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "Period",
                    "value": f"{results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')}",
                    "inline": False
                },
                {
                    "name": "Initial Bankroll",
                    "value": f"${results.get('initial_bankroll', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Final Bankroll",
                    "value": f"${results.get('final_bankroll', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "ROI",
                    "value": f"{roi:.2f}%",
                    "inline": True
                },
                {
                    "name": "Total Bets",
                    "value": str(results.get('total_bets', 0)),
                    "inline": True
                },
                {
                    "name": "Wins",
                    "value": str(results.get('wins', 0)),
                    "inline": True
                },
                {
                    "name": "Win Rate",
                    "value": f"{results.get('win_rate', 0):.1f}%",
                    "inline": True
                },
                {
                    "name": "Sharpe Ratio",
                    "value": f"{results.get('sharpe_ratio', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Max Drawdown",
                    "value": f"{results.get('max_drawdown', 0):.1f}%",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Ominari Backtesting Engine"
            }
        }
        
        return self._send_embed(embed)
        
    def send_daily_summary(self, summary: Dict):
        """Send daily trading summary"""
        # Determine color based on daily PnL
        daily_pnl = summary.get('daily_pnl', 0)
        if daily_pnl > 0:
            color = 0x00ff00  # Green
        elif daily_pnl == 0:
            color = 0xffff00  # Yellow
        else:
            color = 0xff0000  # Red
            
        embed = {
            "title": "📊 Daily Trading Summary",
            "description": f"Date: {summary.get('date', datetime.now().strftime('%Y-%m-%d'))}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "Starting Bankroll",
                    "value": f"${summary.get('start_bankroll', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Ending Bankroll",
                    "value": f"${summary.get('end_bankroll', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Daily P&L",
                    "value": f"{'+' if daily_pnl >= 0 else ''}{daily_pnl:.2f}",
                    "inline": True
                },
                {
                    "name": "Total Trades",
                    "value": str(summary.get('total_trades', 0)),
                    "inline": True
                },
                {
                    "name": "Winning Trades",
                    "value": str(summary.get('winning_trades', 0)),
                    "inline": True
                },
                {
                    "name": "Daily Win Rate",
                    "value": f"{summary.get('win_rate', 0):.1f}%",
                    "inline": True
                },
                {
                    "name": "Total Exposure",
                    "value": f"${summary.get('total_exposure', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Active Positions",
                    "value": str(summary.get('active_positions', 0)),
                    "inline": True
                },
                {
                    "name": "Overall ROI",
                    "value": f"{summary.get('total_roi', 0):.2f}%",
                    "inline": True
                }
            ],
            "footer": {
                "text": "Ominari Trading System"
            }
        }
        
        # Add top trades if available
        if 'top_trades' in summary and summary['top_trades']:
            trades_text = "\n".join([
                f"• {t['market']}: {'+' if t['pnl'] >= 0 else ''}{t['pnl']:.2f} ({t['outcome']} @ {t['odds']:.2f})"
                for t in summary['top_trades'][:3]
            ])
            embed["fields"].append({
                "name": "Top Trades",
                "value": trades_text,
                "inline": False
            })
            
        return self._send_embed(embed)
        
    def send_error_alert(self, error: str, context: str = "System"):
        """Send error alert"""
        embed = {
            "title": "⚠️ Error Alert",
            "description": f"Error in: {context}",
            "color": 0xff0000,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "Error",
                    "value": str(error)[:1024],  # Discord has field value limit
                    "inline": False
                }
            ],
            "footer": {
                "text": "Ominari Trading System"
            }
        }
        
        return self._send_embed(embed)
        
    def send_market_alert(self, alert_type: str, details: Dict):
        """Send market condition alerts"""
        colors = {
            'low_liquidity': 0xff9900,  # Orange
            'high_opportunity': 0x00ff00,  # Green
            'market_closed': 0x808080,  # Gray
            'unusual_odds': 0xffff00     # Yellow
        }
        
        titles = {
            'low_liquidity': '💧 Low Liquidity Alert',
            'high_opportunity': '🎯 High Edge Opportunity',
            'market_closed': '🏁 Market Closed',
            'unusual_odds': '📊 Unusual Odds Movement'
        }
        
        embed = {
            "title": titles.get(alert_type, "Market Alert"),
            "color": colors.get(alert_type, 0x0099ff),
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [],
            "footer": {
                "text": "Ominari Market Monitor"
            }
        }
        
        # Add fields based on alert type
        if alert_type == 'low_liquidity':
            embed["description"] = "Markets with insufficient liquidity detected"
            embed["fields"] = [
                {
                    "name": "Market",
                    "value": details.get('market', 'Unknown'),
                    "inline": True
                },
                {
                    "name": "Available Liquidity",
                    "value": f"${details.get('liquidity', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Required",
                    "value": f"${details.get('required', 0):.2f}",
                    "inline": True
                }
            ]
        elif alert_type == 'high_opportunity':
            embed["description"] = "High edge opportunity detected"
            embed["fields"] = [
                {
                    "name": "Market",
                    "value": details.get('market', 'Unknown'),
                    "inline": True
                },
                {
                    "name": "Edge",
                    "value": f"{details.get('edge', 0):.2f}%",
                    "inline": True
                },
                {
                    "name": "Liquidity",
                    "value": f"${details.get('liquidity', 0):.2f}",
                    "inline": True
                },
                {
                    "name": "Max Bet",
                    "value": f"${details.get('max_bet', 0):.2f}",
                    "inline": True
                }
            ]
        
        return self._send_embed(embed)
    
    def send_embed(self, embed: Dict) -> bool:
        """Send a custom embed message to Discord"""
        return self._send_embed(embed)


# Global instance
discord_notifier = DiscordNotifier()