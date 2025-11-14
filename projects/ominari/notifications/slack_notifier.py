"""
Slack Notification System for Ominari
Sends backtest results and trading alerts to Slack
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Handles sending notifications to Slack"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.info("Slack notifications disabled - no webhook URL configured")
            
    def send_message(self, text: str, attachments: Optional[List[Dict]] = None) -> bool:
        """Send a message to Slack"""
        if not self.enabled:
            return False
            
        payload = {"text": text}
        
        if attachments:
            payload["attachments"] = attachments
            
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
            
    def send_backtest_results(self, results: Dict):
        """Send backtest results to Slack"""
        # Format the message
        text = f"🎯 *Backtest Completed*"
        
        # Create attachment with results
        color = "good" if results.get("roi", 0) > 0 else "danger"
        
        fields = [
            {"title": "Period", "value": f"{results.get('start_date')} to {results.get('end_date')}", "short": True},
            {"title": "Total Trades", "value": str(results.get('total_trades', 0)), "short": True},
            {"title": "Win Rate", "value": f"{results.get('win_rate', 0):.1f}%", "short": True},
            {"title": "ROI", "value": f"{results.get('roi', 0):.2f}%", "short": True},
            {"title": "Final Bankroll", "value": f"${results.get('final_bankroll', 0):,.2f}", "short": True},
            {"title": "Total PnL", "value": f"${results.get('total_pnl', 0):,.2f}", "short": True},
            {"title": "Max Drawdown", "value": f"{results.get('max_drawdown', 0):.1f}%", "short": True},
            {"title": "Sharpe Ratio", "value": f"{results.get('sharpe_ratio', 0):.2f}", "short": True}
        ]
        
        attachment = {
            "color": color,
            "fields": fields,
            "footer": "Ominari Trading System",
            "ts": int(datetime.now().timestamp())
        }
        
        # Add top performers if available
        if results.get('top_strategies'):
            top_text = "\n*Top Performing Strategies:*\n"
            for i, strategy in enumerate(results['top_strategies'][:3], 1):
                top_text += f"{i}. {strategy['name']}: ROI {strategy['roi']:.2f}%\n"
            text += "\n" + top_text
            
        self.send_message(text, [attachment])
        
    def send_trade_alert(self, trade: Dict):
        """Send paper trading alert"""
        emoji = "📈" if trade.get('type') == 'BUY' else "📉"
        
        text = f"{emoji} *Paper Trade Executed*"
        
        color = "good" if trade.get('edge', 0) > 5 else "warning"
        
        attachment = {
            "color": color,
            "fields": [
                {"title": "Market", "value": trade.get('market', 'Unknown'), "short": True},
                {"title": "Outcome", "value": trade.get('outcome', ''), "short": True},
                {"title": "Amount", "value": f"${trade.get('amount', 0):.2f}", "short": True},
                {"title": "Odds", "value": f"{trade.get('odds', 0):.2f}", "short": True},
                {"title": "Edge", "value": f"{trade.get('edge', 0):.2f}%", "short": True},
                {"title": "Kelly %", "value": f"{trade.get('kelly_pct', 0):.1f}%", "short": True}
            ],
            "footer": f"Bankroll: ${trade.get('bankroll', 0):,.2f}",
            "ts": int(datetime.now().timestamp())
        }
        
        self.send_message(text, [attachment])
        
    def send_daily_summary(self, summary: Dict):
        """Send daily trading summary"""
        text = "📊 *Daily Trading Summary*"
        
        color = "good" if summary.get('daily_pnl', 0) > 0 else "danger"
        
        fields = [
            {"title": "Date", "value": summary.get('date', datetime.now().strftime('%Y-%m-%d')), "short": True},
            {"title": "Trades Today", "value": str(summary.get('trades_today', 0)), "short": True},
            {"title": "Daily PnL", "value": f"${summary.get('daily_pnl', 0):,.2f}", "short": True},
            {"title": "Daily ROI", "value": f"{summary.get('daily_roi', 0):.2f}%", "short": True},
            {"title": "Win Rate", "value": f"{summary.get('win_rate', 0):.1f}%", "short": True},
            {"title": "Current Bankroll", "value": f"${summary.get('current_bankroll', 0):,.2f}", "short": True},
            {"title": "Active Positions", "value": str(summary.get('active_positions', 0)), "short": True},
            {"title": "Total Exposure", "value": f"${summary.get('total_exposure', 0):,.2f}", "short": True}
        ]
        
        attachment = {
            "color": color,
            "fields": fields,
            "footer": "Ominari Trading System",
            "ts": int(datetime.now().timestamp())
        }
        
        # Add performance chart if available
        if summary.get('performance_chart_url'):
            attachment['image_url'] = summary['performance_chart_url']
            
        self.send_message(text, [attachment])
        
    def send_error_alert(self, error: str, context: Optional[Dict] = None):
        """Send error alert"""
        text = "🚨 *Trading System Error*"
        
        fields = [
            {"title": "Error", "value": error, "short": False},
            {"title": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "short": True}
        ]
        
        if context:
            for key, value in context.items():
                fields.append({"title": key, "value": str(value), "short": True})
                
        attachment = {
            "color": "danger",
            "fields": fields,
            "footer": "Ominari Trading System",
            "ts": int(datetime.now().timestamp())
        }
        
        self.send_message(text, [attachment])
        
    def send_startup_message(self):
        """Send system startup notification"""
        text = "🚀 *Ominari Trading System Started*"
        
        attachment = {
            "color": "good",
            "fields": [
                {"title": "Status", "value": "System Online", "short": True},
                {"title": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                {"title": "Dashboard", "value": "http://localhost:8888", "short": True},
                {"title": "Performance", "value": "http://localhost:8889", "short": True}
            ],
            "footer": "Ominari Trading System",
            "ts": int(datetime.now().timestamp())
        }
        
        self.send_message(text, [attachment])


# Create singleton instance
slack_notifier = SlackNotifier()


def setup_slack_webhook():
    """Interactive setup for Slack webhook"""
    print("\n🔔 Slack Notification Setup")
    print("===========================")
    print("\nTo get a Slack webhook URL:")
    print("1. Go to https://api.slack.com/apps")
    print("2. Create a new app or select existing")
    print("3. Add 'Incoming Webhooks' feature")
    print("4. Create a webhook for your channel")
    print("5. Copy the webhook URL")
    print("\nExample: https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
    
    webhook_url = input("\nEnter your Slack webhook URL (or press Enter to skip): ").strip()
    
    if webhook_url:
        # Test the webhook
        test_notifier = SlackNotifier(webhook_url)
        if test_notifier.send_message("✅ Ominari Slack integration test successful!"):
            print("✅ Webhook test successful!")
            
            # Save to .env file
            env_file = ".env"
            env_content = []
            
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    env_content = f.readlines()
                    
            # Update or add webhook URL
            updated = False
            for i, line in enumerate(env_content):
                if line.startswith('SLACK_WEBHOOK_URL='):
                    env_content[i] = f'SLACK_WEBHOOK_URL={webhook_url}\n'
                    updated = True
                    break
                    
            if not updated:
                env_content.append(f'\n# Slack Notifications\nSLACK_WEBHOOK_URL={webhook_url}\n')
                
            with open(env_file, 'w') as f:
                f.writelines(env_content)
                
            print("✅ Webhook URL saved to .env file")
            print("\n📱 You'll receive notifications for:")
            print("  • Backtest results")
            print("  • Paper trades")
            print("  • Daily summaries")
            print("  • System errors")
            return True
        else:
            print("❌ Webhook test failed - please check the URL")
            return False
    else:
        print("\n⏭️  Skipping Slack setup - you can set it up later")
        return False


if __name__ == "__main__":
    # Run setup if called directly
    setup_slack_webhook()