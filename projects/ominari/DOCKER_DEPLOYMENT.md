# Ominari DApp - Docker Deployment Guide

## Quick Start

### 1. Build and Start All Services
```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Access Services
- **Dashboard**: http://localhost:8888
- **Health Check**: http://localhost:8888/health
- **Cache Stats**: http://localhost:8888/cache-stats
- **Metrics**: http://localhost:8888/metrics
- **Grafana**: http://localhost:3000 (admin/ominari_admin_2025)
- **Prometheus**: http://localhost:9091

## Individual Services

### Dashboard Only
```bash
docker-compose up -d ominari-dashboard
```

### Trading Engine Only
```bash
docker-compose up -d ominari-trading
```

### Paper Trading Only
```bash
docker-compose up -d ominari-paper
```

## Database Management

### Connect to PostgreSQL
```bash
docker exec -it ominari-postgres psql -U ominari_user -d ominari_live
```

### Create Backup
```bash
docker-compose --profile backup up ominari-backup
```

### Restore from Backup
```bash
docker exec -i ominari-postgres psql -U ominari_user -d ominari_live < backup.sql
```

## Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ominari-dashboard

# Last 100 lines
docker-compose logs --tail=100 ominari-dashboard
```

### Check Health
```bash
# Dashboard health
curl http://localhost:8888/health

# All services health
docker-compose exec ominari-dashboard python -c "
import requests
services = [
    ('Dashboard', 'http://localhost:8888/health'),
    ('PostgreSQL', 'postgres:5432'),
    ('Redis', 'redis:6379')
]
for name, url in services:
    try:
        if ':' in url and not url.startswith('http'):
            # TCP check
            import socket
            host, port = url.split(':')
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            print(f'{name}: {'UP' if result == 0 else 'DOWN'}')
        else:
            r = requests.get(url, timeout=5)
            print(f'{name}: {'UP' if r.status_code == 200 else f'DOWN ({r.status_code})'}')
    except Exception as e:
        print(f'{name}: DOWN ({str(e)})')
"
```

## Performance Features

### Rate Limiting
- **General**: 60 requests/minute
- **API**: 30 requests/minute  
- **WebSocket**: 120 requests/minute

### Caching
- **Market Data**: 30 second TTL
- **In-Memory**: Automatic cleanup
- **Redis**: For distributed caching

## Scaling

### Horizontal Scaling
```bash
# Scale dashboard to 3 instances
docker-compose up -d --scale ominari-dashboard=3
```

### Resource Limits
Edit docker-compose.yml:
```yaml
ominari-dashboard:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs ominari-dashboard

# Rebuild image
docker-compose build --no-cache ominari-dashboard

# Reset volumes
docker-compose down -v
```

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
docker exec -it ominari-postgres pg_isready -U ominari_user

# Check environment variables
docker-compose exec ominari-dashboard env | grep PG_
```

### Performance Issues
```bash
# Check resource usage
docker stats

# Check cache hit rate
curl http://localhost:8888/cache-stats

# Monitor rate limiting
docker-compose logs ominari-dashboard | grep "Rate limit"
```

## Production Deployment

### 1. Use Production Compose File
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 2. Enable HTTPS (Nginx)
```bash
# Start with production profile
docker-compose --profile production up -d
```

### 3. Environment Variables
Create `.env` file:
```env
PG_PASSWORD=secure_password_here
REDIS_PASSWORD=secure_redis_password
SECRET_KEY=your_secret_key_here
```

### 4. Security Checklist
- [ ] Change default passwords
- [ ] Enable Redis password authentication
- [ ] Use HTTPS for all endpoints
- [ ] Set up firewall rules
- [ ] Enable log rotation
- [ ] Set up automated backups
- [ ] Monitor disk space

## Maintenance

### Update Images
```bash
# Pull latest images
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate
```

### Clean Up
```bash
# Remove stopped containers
docker-compose rm -f

# Remove unused volumes
docker volume prune

# Full cleanup (CAUTION: removes data)
docker-compose down -v
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          ssh user@server "cd /app && docker-compose pull && docker-compose up -d"
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Check health: `curl http://localhost:8888/health`
3. Review this guide
4. Open an issue on GitHub