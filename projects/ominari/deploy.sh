#!/bin/bash
#
# Ominari Blockchain Trading System - Automated Deployment Script
#
# Usage:
#   ./deploy.sh [environment] [action]
#
# Environments:
#   - testnet: Deploy to testnet environment
#   - staging: Deploy to staging environment
#   - production: Deploy to production environment
#
# Actions:
#   - deploy: Full deployment
#   - update: Update existing deployment
#   - rollback: Rollback to previous version
#   - status: Check deployment status

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENVIRONMENT="${1:-testnet}"
ACTION="${2:-deploy}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="deployment_${ENVIRONMENT}_${TIMESTAMP}.log"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Environment validation
validate_environment() {
    case $ENVIRONMENT in
        testnet|staging|production)
            log_info "Environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: testnet, staging, production"
            exit 1
            ;;
    esac
}

# Load environment variables
load_env_vars() {
    local env_file=".env.${ENVIRONMENT}"
    
    if [ -f "$env_file" ]; then
        log_info "Loading environment variables from $env_file"
        export $(cat "$env_file" | grep -v '^#' | xargs)
    else
        log_warning "Environment file $env_file not found"
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        log_success "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        exit 1
    fi
}

# Deploy testnet
deploy_testnet() {
    log_info "Deploying to testnet..."
    
    # Create testnet configuration if not exists
    if [ ! -f "docker-compose.testnet.yml" ]; then
        log_info "Creating testnet configuration..."
        python testnet_config.py
    fi
    
    # Start testnet services
    docker-compose -f docker-compose.testnet.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check service status
    docker-compose -f docker-compose.testnet.yml ps
    
    log_success "Testnet deployment complete"
}

# Deploy staging
deploy_staging() {
    log_info "Deploying to staging..."
    
    # Use main docker-compose with staging overrides
    docker-compose up -d
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose exec -T ominari-trading python -c "
from alembic.config import Config
from alembic import command
alembic_cfg = Config('alembic.ini')
command.upgrade(alembic_cfg, 'head')
"
    
    log_success "Staging deployment complete"
}

# Deploy production
deploy_production() {
    log_warning "Production deployment - requires confirmation"
    read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        log_error "Production deployment cancelled"
        exit 1
    fi
    
    log_info "Deploying to production..."
    
    # Create backup before deployment
    log_info "Creating backup..."
    docker-compose exec postgres pg_dump -U ominari_user ominari_live > "backup_pre_deploy_${TIMESTAMP}.sql"
    
    # Deploy with zero downtime
    docker-compose up -d --no-deps --scale ominari-trading=2 ominari-trading
    
    # Wait for new instance to be healthy
    sleep 30
    
    # Remove old instance
    docker-compose up -d --no-deps --scale ominari-trading=1 ominari-trading
    
    log_success "Production deployment complete"
}

# Update deployment
update_deployment() {
    log_info "Updating $ENVIRONMENT deployment..."
    
    # Pull latest changes
    git pull origin main
    
    # Rebuild images
    build_images
    
    # Update services
    case $ENVIRONMENT in
        testnet)
            docker-compose -f docker-compose.testnet.yml up -d
            ;;
        *)
            docker-compose up -d
            ;;
    esac
    
    log_success "Update complete"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back $ENVIRONMENT deployment..."
    
    # Get previous image
    local previous_image=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep ominari | head -2 | tail -1)
    
    if [ -z "$previous_image" ]; then
        log_error "No previous image found for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $previous_image"
    
    # Update docker-compose to use previous image
    # This would need to be implemented based on your versioning strategy
    
    log_warning "Rollback functionality needs to be configured for your versioning strategy"
}

# Check deployment status
check_status() {
    log_info "Checking $ENVIRONMENT deployment status..."
    
    case $ENVIRONMENT in
        testnet)
            docker-compose -f docker-compose.testnet.yml ps
            ;;
        *)
            docker-compose ps
            ;;
    esac
    
    # Check service health
    log_info "Checking service health..."
    
    # Trading API
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "Trading API: Healthy"
    else
        log_error "Trading API: Not responding"
    fi
    
    # PostgreSQL
    if docker-compose exec postgres pg_isready -U ominari_user > /dev/null 2>&1; then
        log_success "PostgreSQL: Ready"
    else
        log_error "PostgreSQL: Not ready"
    fi
    
    # Redis
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis: Ready"
    else
        log_error "Redis: Not ready"
    fi
}

# Post-deployment tests
run_post_deployment_tests() {
    log_info "Running post-deployment tests..."
    
    # Run integration tests
    if [ "$ENVIRONMENT" = "testnet" ]; then
        docker-compose -f docker-compose.testnet.yml exec -T ominari-testnet python blockchain_integration_tests.py
    fi
    
    # Check API endpoints
    local endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8000/api/v1/status"
        "http://localhost:9090"  # Web monitor
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s "$endpoint" > /dev/null; then
            log_success "Endpoint $endpoint: OK"
        else
            log_warning "Endpoint $endpoint: Not available"
        fi
    done
}

# Main deployment logic
main() {
    echo "🚀 Ominari Blockchain Trading System Deployment"
    echo "=============================================="
    
    # Validate inputs
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Load environment variables
    load_env_vars
    
    # Execute action
    case $ACTION in
        deploy)
            build_images
            case $ENVIRONMENT in
                testnet)
                    deploy_testnet
                    ;;
                staging)
                    deploy_staging
                    ;;
                production)
                    deploy_production
                    ;;
            esac
            run_post_deployment_tests
            ;;
        update)
            update_deployment
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            check_status
            ;;
        *)
            log_error "Invalid action: $ACTION"
            log_error "Valid actions: deploy, update, rollback, status"
            exit 1
            ;;
    esac
    
    log_success "Deployment script completed"
    
    # Show summary
    echo ""
    echo "📊 Deployment Summary"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Action: $ACTION"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Show access URLs
    case $ENVIRONMENT in
        testnet)
            echo "🌐 Access URLs:"
            echo "  Trading API: http://localhost:8001"
            echo "  Web Monitor: http://localhost:9090"
            echo "  Grafana: http://localhost:3001"
            echo "  PostgreSQL: localhost:5433"
            echo ""
            echo "⚠️  Remember: This is TESTNET - no real money!"
            ;;
        *)
            echo "🌐 Access URLs:"
            echo "  Trading API: http://localhost:8000"
            echo "  Web Monitor: http://localhost:9090"
            echo "  Grafana: http://localhost:3000"
            echo "  Prometheus: http://localhost:9091"
            ;;
    esac
}

# Run main function
main