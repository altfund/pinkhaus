# Production Deployment Checklist

## Pre-Deployment Verification

### 1. Code Quality ✓
- [x] Pre-commit hooks installed
- [x] All tests passing
- [x] Linting and formatting clean
- [x] Type checking passing

### 2. Security
- [ ] Environment variables secured
  ```bash
  # Create .env.production with:
  DATABASE_URL=postgresql://ominari_user:SECURE_PASSWORD@db.ominari.trading:5432/ominari_production
  BLOCKCHAIN_RPC_URL=https://YOUR_RPC_ENDPOINT
  API_KEY=YOUR_SECURE_API_KEY
  SECRET_KEY=YOUR_SECRET_KEY
  ```
- [ ] Secrets removed from code
- [ ] SSL certificates configured
- [ ] Database passwords rotated
- [ ] API keys secured in AWS Secrets Manager

### 3. Infrastructure Setup
- [ ] Production database provisioned (AWS RDS PostgreSQL)
- [ ] Redis cache configured (AWS ElastiCache)
- [ ] Container registry setup (AWS ECR)
- [ ] Load balancer configured (AWS ALB)
- [ ] Auto-scaling groups configured

### 4. Monitoring & Logging
- [ ] CloudWatch logs configured
- [ ] Prometheus metrics exported
- [ ] Grafana dashboards deployed
- [ ] PagerDuty alerts configured
- [ ] Health check endpoints verified

### 5. Database
- [ ] Production migrations tested
- [ ] Backup strategy implemented
- [ ] Point-in-time recovery enabled
- [ ] Read replicas configured (if needed)

### 6. Application Configuration
- [ ] Rate limiting configured appropriately
- [ ] Cache TTLs optimized
- [ ] WebSocket scaling configured
- [ ] CORS settings secured

## Deployment Steps

### 1. Final Testing
```bash
# Run full test suite
pytest tests/ -v

# Test production build
docker build -t ominari:prod .
docker run -it --rm ominari:prod python -m pytest
```

### 2. Database Migration
```bash
# Backup current database
pg_dump -h localhost -p 5999 -U ominari_user ominari_production > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migrations
alembic upgrade head
```

### 3. Deploy Application
```bash
# Use the deployment script
./deploy.sh production deploy

# Or manual deployment
docker build -t ominari:latest .
docker tag ominari:latest $ECR_REGISTRY/ominari:latest
docker push $ECR_REGISTRY/ominari:latest
aws ecs update-service --cluster ominari-cluster --service ominari-service --force-new-deployment
```

### 4. Verify Deployment
- [ ] Health checks passing
- [ ] Dashboard accessible
- [ ] Paper trading functional
- [ ] Real-time data flowing
- [ ] Edge calculations working

### 5. Monitor Initial Operation
- [ ] Watch logs for errors
- [ ] Monitor resource usage
- [ ] Check response times
- [ ] Verify data accuracy

## Production URLs

- Main Dashboard: https://ominari.trading
- API Endpoint: https://api.ominari.trading
- Health Check: https://api.ominari.trading/health
- Metrics: https://metrics.ominari.trading
- Grafana: https://monitoring.ominari.trading

## Rollback Plan

If issues occur:
```bash
# Quick rollback
./deploy.sh production rollback

# Manual rollback
aws ecs update-service --cluster ominari-cluster --service ominari-service --task-definition ominari-task:PREVIOUS_VERSION
```

## Post-Deployment

1. Monitor for 24 hours
2. Check all integrations
3. Verify paper trading accuracy
4. Review performance metrics
5. Update documentation

## Emergency Contacts

- On-call: [Your Phone]
- Escalation: [Manager Phone]
- AWS Support: [Case Number]

---

**Remember**: Deploy during low-traffic hours and have rollback ready!