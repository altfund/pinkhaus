#!/bin/bash

echo "🎮 Discord Webhook Setup for Ominari Trading Bot"
echo "=============================================="
echo ""
echo "To set up Discord notifications:"
echo ""
echo "1. Create a Discord webhook:"
echo "   a. Right-click on a Discord channel where you want notifications"
echo "   b. Select 'Edit Channel' → 'Integrations' → 'Webhooks'"
echo "   c. Click 'New Webhook' or select an existing one"
echo "   d. Give it a name like 'Ominari Trading Bot'"
echo "   e. Copy the Webhook URL"
echo ""
echo "2. Enter your Discord webhook URL:"
read -p "Discord Webhook URL: " webhook_url

if [ -z "$webhook_url" ]; then
    echo "❌ No webhook URL provided. Exiting."
    exit 1
fi

# Validate webhook URL format
if [[ ! "$webhook_url" =~ ^https://discord\.com/api/webhooks/ ]]; then
    echo "❌ Invalid Discord webhook URL format"
    echo "   URL should start with: https://discord.com/api/webhooks/"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "# Ominari Trading Bot Environment Variables" > .env
fi

# Check if DISCORD_WEBHOOK_URL already exists
if grep -q "DISCORD_WEBHOOK_URL=" .env 2>/dev/null; then
    echo ""
    echo "⚠️  DISCORD_WEBHOOK_URL already exists in .env"
    read -p "Do you want to update it? (y/n): " update_existing
    
    if [[ "$update_existing" =~ ^[Yy]$ ]]; then
        # Update existing webhook URL
        sed -i.bak "s|DISCORD_WEBHOOK_URL=.*|DISCORD_WEBHOOK_URL=$webhook_url|" .env
        echo "✅ Updated Discord webhook URL"
    else
        echo "❌ Keeping existing webhook URL"
        exit 0
    fi
else
    # Add new webhook URL
    echo "" >> .env
    echo "# Discord Notifications" >> .env
    echo "DISCORD_WEBHOOK_URL=$webhook_url" >> .env
    echo "✅ Added Discord webhook URL to .env"
fi

# Test the webhook
echo ""
echo "Testing Discord webhook..."

python3 -c "
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('$0'))))

# Set the webhook URL environment variable
os.environ['DISCORD_WEBHOOK_URL'] = '$webhook_url'

from notifications.discord_notifier import DiscordNotifier

notifier = DiscordNotifier()
if notifier.send_startup_message():
    print('✅ Test message sent successfully! Check your Discord channel.')
else:
    print('❌ Failed to send test message. Please check your webhook URL.')
"

echo ""
echo "Setup complete! Discord notifications are configured."
echo ""
echo "To use in your code:"
echo "  from notifications.discord_notifier import discord_notifier"
echo "  discord_notifier.send_trade_alert(trade_data)"
echo ""
echo "Or set the environment variable:"
echo "  export DISCORD_WEBHOOK_URL='$webhook_url'"