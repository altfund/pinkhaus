#!/bin/bash
# Production deployment script for Ominari

set -e

echo "🚀 Ominari Production Deployment"
echo "================================"

# Check if running from main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Error: Production deployment must be run from main branch"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Confirm production deployment
read -p "⚠️  Deploy to PRODUCTION? This will affect real users. (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# Load production environment
if [ ! -f ".env.production" ]; then
    echo "❌ Error: .env.production not found"
    echo "Copy .env.production.example and configure it first"
    exit 1
fi

export $(cat .env.production | grep -v '^#' | xargs)

# Run pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check tests
echo "Running tests..."
.venv/bin/python -m pytest tests/ -v --tb=short || {
    echo "❌ Tests failed. Fix before deploying."
    exit 1
}

# Check dashboard
echo "Checking dashboard..."
.venv/bin/python -c "from web_dashboard_real_odds import app; print('✓ Dashboard OK')" || {
    echo "❌ Dashboard check failed"
    exit 1
}

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t ominari:production .

# Tag for ECR
docker tag ominari:production $ECR_REGISTRY/ominari:production
docker tag ominari:production $ECR_REGISTRY/ominari:latest

# Login to ECR
echo "🔐 Logging into AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Push to ECR
echo "⬆️  Pushing to ECR..."
docker push $ECR_REGISTRY/ominari:production
docker push $ECR_REGISTRY/ominari:latest

# Create database backup
echo "💾 Creating database backup..."
BACKUP_FILE="backup_production_$(date +%Y%m%d_%H%M%S).sql"
pg_dump $DATABASE_URL > $BACKUP_FILE
echo "Backup saved to: $BACKUP_FILE"

# Deploy to ECS
echo "🚢 Deploying to ECS..."
aws ecs update-service \
    --cluster ominari-cluster \
    --service ominari-service \
    --force-new-deployment \
    --desired-count 2

# Wait for deployment
echo "⏳ Waiting for deployment to stabilize..."
aws ecs wait services-stable \
    --cluster ominari-cluster \
    --services ominari-service

# Run smoke tests
echo "🧪 Running smoke tests..."
sleep 30  # Give services time to start

# Check health endpoint
curl -f https://api.ominari.trading/health || {
    echo "❌ Health check failed!"
    echo "Rolling back..."
    aws ecs update-service \
        --cluster ominari-cluster \
        --service ominari-service \
        --task-definition ominari-task:$((CURRENT_VERSION-1))
    exit 1
}

# Check dashboard
curl -f https://ominari.trading || {
    echo "❌ Dashboard check failed!"
    exit 1
}

# Update monitoring
echo "📊 Updating monitoring..."
curl -X POST https://monitoring.ominari.trading/api/v1/alerts \
    -H "Content-Type: application/json" \
    -d "{\"deployment\": \"production\", \"version\": \"$(git rev-parse HEAD)\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

echo "✅ Production deployment successful!"
echo ""
echo "📊 Deployment Summary:"
echo "- Version: $(git rev-parse HEAD)"
echo "- Time: $(date)"
echo "- Dashboard: https://ominari.trading"
echo "- API: https://api.ominari.trading"
echo ""
echo "📋 Next steps:"
echo "1. Monitor CloudWatch logs"
echo "2. Check Grafana dashboards"
echo "3. Verify paper trading is working"
echo "4. Monitor for 24 hours"

# Send notification
if [ ! -z "$SLACK_WEBHOOK" ]; then
    curl -X POST $SLACK_WEBHOOK \
        -H 'Content-type: application/json' \
        -d "{\"text\":\"✅ Ominari deployed to production successfully!\"}"
fi