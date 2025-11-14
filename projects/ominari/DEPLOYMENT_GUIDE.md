# Ominari Production Deployment Guide

This guide walks through deploying Ominari to production step by step.

## Prerequisites

- Domain name (ominari.trading) with DNS access
- AWS account (for cloud deployment) or dedicated server
- PostgreSQL 15+ database
- Redis (optional but recommended)
- Docker & Docker Compose
- SSL certificate (Let's Encrypt)

## Step 1: Environment Setup

### 1.1 Create Production Environment File

```bash
# Generate production environment configuration
./infrastructure/scripts/create-prod-env.sh

# Edit the generated file
nano .env.production
```

**Important values to update:**
- `OVERTIME_API_KEY` - Your actual Overtime API key
- `DATABASE_URL` - Production database connection string
- `BLOCKCHAIN_RPC_URL` - Your RPC endpoint (Alchemy, Infura, etc.)
- `AWS_ACCOUNT_ID` - If deploying to AWS
- `SENTRY_DSN` - For error tracking (optional)

### 1.2 Verify Configuration

```bash
# Test environment file
source .env.production
echo $DATABASE_URL  # Should show your database URL
```

## Step 2: Database Setup

### 2.1 Create Production Database

```bash
# If using the existing PostgreSQL instance
# Database is already created: database_v2 on port 5999

# For a new production database:
createdb -h your-db-host -U postgres ominari_production
```

### 2.2 Run Migrations

```bash
# Backup existing data first
./infrastructure/scripts/database-migration.sh backup

# Run migrations
./infrastructure/scripts/database-migration.sh migrate

# Verify database
./infrastructure/scripts/database-migration.sh verify
```

## Step 3: Build & Test Locally

### 3.1 Build Production Docker Image

```bash
# Build production image
docker build -f infrastructure/docker/Dockerfile.production -t ominari:production .

# Test locally
docker run -it --rm \
  --env-file .env.production \
  -p 8888:8888 \
  ominari:production
```

### 3.2 Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Test dashboard import
python -c "from web_dashboard_real_odds import app; print('✓ Dashboard OK')"
```

## Step 4: Domain & SSL Setup

### 4.1 Configure DNS

Point these domains to your server IP:
- `ominari.trading` → your-server-ip
- `www.ominari.trading` → your-server-ip  
- `api.ominari.trading` → your-server-ip

### 4.2 Obtain SSL Certificate

```bash
# Update email in setup-ssl.sh first!
nano infrastructure/scripts/setup-ssl.sh

# Run SSL setup
sudo ./infrastructure/scripts/setup-ssl.sh all

# Test SSL
./infrastructure/scripts/setup-ssl.sh test
```

## Step 5: Deploy to Production

### Option A: Direct Server Deployment

```bash
# Copy files to server
scp -r . user@your-server:/opt/ominari

# SSH to server
ssh user@your-server

# Start services
cd /opt/ominari
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f
```

### Option B: AWS Deployment

#### 5.1 Setup AWS Infrastructure

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Review plan
terraform plan

# Apply infrastructure
terraform apply
```

#### 5.2 Push to ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

# Tag and push
docker tag ominari:production $ECR_REGISTRY/ominari:latest
docker push $ECR_REGISTRY/ominari:latest
```

#### 5.3 Deploy to ECS/EKS

```bash
# Using deployment script
./deploy-production.sh

# Or manually with kubectl
kubectl apply -f infrastructure/kubernetes/deployment.yaml
```

## Step 6: Post-Deployment

### 6.1 Verify Deployment

```bash
# Check health endpoint
curl https://ominari.trading/health

# Check dashboard
curl -I https://ominari.trading

# Check API
curl https://api.ominari.trading/api/v1/markets
```

### 6.2 Setup Monitoring

1. Access Grafana at http://your-server:3000
2. Import dashboard from `monitoring/grafana-dashboards/`
3. Configure alerts in Prometheus

### 6.3 Setup Backups

```bash
# Test backup
./infrastructure/scripts/backup-restore.sh backup-all

# Setup cron for automated backups
crontab -e
# Add: 0 3 * * * /opt/ominari/infrastructure/scripts/backup-restore.sh auto
```

## Step 7: Production Checklist

- [ ] Environment variables configured
- [ ] Database migrated and backed up
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backups automated
- [ ] Health checks passing
- [ ] Paper trading active
- [ ] Error tracking setup
- [ ] Performance acceptable
- [ ] Security headers verified

## Troubleshooting

### Database Connection Issues

```bash
# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check logs
docker-compose logs database
```

### SSL Certificate Issues

```bash
# Renew certificate
sudo certbot renew --force-renewal

# Check certificate
openssl s_client -connect ominari.trading:443
```

### Application Errors

```bash
# Check logs
docker-compose logs -f ominari-dashboard

# Connect to container
docker exec -it ominari-dashboard bash
```

## Rollback Procedure

If issues occur:

```bash
# Quick rollback using script
./deploy-production.sh rollback

# Or manual Docker rollback
docker-compose down
docker-compose up -d --scale ominari-dashboard=0
docker run -d --name ominari-old [previous-image-tag]
```

## Security Notes

1. **Never commit `.env.production`** - Add to .gitignore
2. **Rotate credentials regularly** - Database passwords, API keys
3. **Use AWS Secrets Manager** for production secrets
4. **Enable CloudTrail** for audit logging
5. **Configure WAF** for DDoS protection
6. **Regular security scans** with tools like Trivy

## Support

For production issues:
1. Check logs first
2. Review monitoring dashboards
3. Check recent deployments
4. Contact on-call engineer

---

Remember: **Always test in staging before production!**