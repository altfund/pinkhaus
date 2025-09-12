#!/bin/bash
# Setup script for Ominari monitoring

echo "=== Ominari Monitoring Setup ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not installed. Please install docker-compose first."
    exit 1
fi

# Start monitoring stack
echo "Starting monitoring stack..."
cd monitoring
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 10

# Check service status
echo ""
echo "=== Service Status ==="
docker-compose ps

echo ""
echo "=== Access URLs ==="
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/ominari123)"
echo "Alertmanager: http://localhost:9093"

echo ""
echo "=== Next Steps ==="
echo "1. Update web_monitor_auth.py to expose /metrics endpoint"
echo "2. Import dashboard in Grafana"
echo "3. Configure alert notifications in Alertmanager"

echo ""
echo "✅ Monitoring setup complete!"
