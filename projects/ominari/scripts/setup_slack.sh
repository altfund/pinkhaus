#!/bin/bash

# Setup Slack notifications for Ominari

echo "🔔 Ominari Slack Notification Setup"
echo "=================================="
echo

# Run the Python setup script
cd "$(dirname "$0")/.."
.venv/bin/python -m notifications.slack_notifier

echo
echo "✅ Setup complete!"
echo
echo "If you configured a webhook URL, you'll receive:"
echo "• Backtest results after each run"
echo "• Paper trading alerts for significant trades"
echo "• Daily trading summaries at midnight"
echo "• Error alerts if something goes wrong"
echo
echo "To test notifications manually:"
echo "python -c \"from notifications.slack_notifier import slack_notifier; slack_notifier.send_message('🎯 Test message from Ominari')\""