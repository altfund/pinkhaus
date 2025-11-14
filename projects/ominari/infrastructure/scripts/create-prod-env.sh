#!/bin/bash
# Script to create production environment file from example

set -e

echo "🔧 Creating Production Environment Configuration"
echo "============================================="

# Check if example file exists
if [ ! -f ".env.production.example" ]; then
    echo "❌ Error: .env.production.example not found"
    exit 1
fi

# Copy example to production
cp .env.production.example .env.production

# Generate secure keys
API_SECRET_KEY=$(openssl rand -hex 32)
SECRET_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Update the file with generated keys
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/GENERATE_SECURE_KEY/$API_SECRET_KEY/g" .env.production
    sed -i '' "s/GENERATE_ANOTHER_SECURE_32_CHAR_KEY_HERE/$SECRET_KEY/g" .env.production
    sed -i '' "s/CHANGE_ME/$DB_PASSWORD/g" .env.production
else
    # Linux
    sed -i "s/GENERATE_SECURE_KEY/$API_SECRET_KEY/g" .env.production
    sed -i "s/GENERATE_ANOTHER_SECURE_32_CHAR_KEY_HERE/$SECRET_KEY/g" .env.production
    sed -i "s/CHANGE_ME/$DB_PASSWORD/g" .env.production
fi

echo "✅ Generated secure keys"
echo ""
echo "⚠️  IMPORTANT: You still need to update these values in .env.production:"
echo "   - OVERTIME_API_KEY: Your actual Overtime API key"
echo "   - DATABASE_URL: Your production database URL"
echo "   - BLOCKCHAIN_RPC_URL: Your RPC endpoint"
echo "   - AWS_ACCOUNT_ID: Your AWS account ID (if using AWS)"
echo "   - ECR_REGISTRY: Your ECR registry URL (if using AWS)"
echo "   - SENTRY_DSN: Your Sentry DSN (if using Sentry)"
echo "   - SLACK_WEBHOOK: Your Slack webhook (if using notifications)"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env.production and update the values above"
echo "2. Ensure .env.production is in .gitignore (never commit secrets!)"
echo "3. Store a backup of .env.production securely"
echo "4. Consider using AWS Secrets Manager for production"

# Set restrictive permissions
chmod 600 .env.production
echo ""
echo "🔒 Set file permissions to 600 (read/write for owner only)"