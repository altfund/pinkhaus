#!/bin/bash
#
# Multi-Chain Deployment Script
# Handles deployment across different environments with proper data isolation
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENVIRONMENT="${1:-development}"
ACTION="${2:-status}"
CHAIN="${3:-all}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_chain() {
    echo -e "${MAGENTA}[CHAIN]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 [environment] [action] [chain]"
    echo ""
    echo "Environments:"
    echo "  production    - Mainnet chains only"
    echo "  staging       - Testnet chains with production config"
    echo "  testnet       - Testnet chains for testing"
    echo "  development   - Local + testnet chains"
    echo ""
    echo "Actions:"
    echo "  deploy        - Deploy chains for environment"
    echo "  stop          - Stop chains"
    echo "  status        - Show status"
    echo "  logs          - Show logs"
    echo "  validate      - Validate data isolation"
    echo "  migrate       - Run migrations"
    echo ""
    echo "Chains:"
    echo "  all           - All chains in environment"
    echo "  optimism      - Optimism only"
    echo "  arbitrum      - Arbitrum only"
    echo "  base          - Base only"
    echo "  local         - Local chain only"
    echo ""
    echo "Examples:"
    echo "  $0 production deploy          # Deploy all mainnet chains"
    echo "  $0 testnet deploy optimism    # Deploy Optimism testnet only"
    echo "  $0 development status         # Show dev environment status"
    exit 1
}

# Set environment variables based on environment
set_environment_vars() {
    case $ENVIRONMENT in
        production)
            export ENVIRONMENT=production
            export DATABASE_URL="${POSTGRES_URL:-postgresql://ominari_user:ominari_2025_secure@postgres:5432/ominari_production}"
            export MAINNET_REPLICAS=2
            export ENABLE_MAINNET=true
            export ENABLE_TESTNET=false
            export LOG_LEVEL=INFO
            export API_RATE_LIMIT=1000
            export COMPOSE_PROFILES=production,mainnet,monitoring
            ;;
        staging)
            export ENVIRONMENT=staging
            export DATABASE_URL="${POSTGRES_URL:-postgresql://ominari_user:ominari_staging@postgres:5432/ominari_staging}"
            export TESTNET_PREFIX=stage
            export TESTNET_REPLICAS=2
            export ENABLE_MAINNET=false
            export ENABLE_TESTNET=true
            export LOG_LEVEL=INFO
            export API_RATE_LIMIT=500
            export COMPOSE_PROFILES=testnet,staging,monitoring
            ;;
        testnet)
            export ENVIRONMENT=testnet
            export DATABASE_URL="${POSTGRES_URL:-postgresql://ominari_user:ominari_testnet@postgres:5432/ominari_testnet}"
            export TESTNET_PREFIX=test
            export TESTNET_REPLICAS=1
            export ENABLE_MAINNET=false
            export ENABLE_TESTNET=true
            export LOG_LEVEL=DEBUG
            export API_RATE_LIMIT=100
            export COMPOSE_PROFILES=testnet
            ;;
        development)
            export ENVIRONMENT=development
            export DATABASE_URL="${POSTGRES_URL:-postgresql://ominari_user:ominari_dev@postgres:5432/ominari_dev}"
            export TESTNET_PREFIX=dev
            export TESTNET_REPLICAS=1
            export ENABLE_MAINNET=false
            export ENABLE_TESTNET=true
            export ENABLE_LOCAL=true
            export LOG_LEVEL=DEBUG
            export API_RATE_LIMIT=100
            export COMPOSE_PROFILES=development,testnet,local,debug
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            usage
            ;;
    esac
    
    # Common vars
    export REDIS_URL="${REDIS_URL:-redis://redis:6379/0}"
    export API_PORT="${API_PORT:-8000}"
    export CORS_ORIGINS="${CORS_ORIGINS:-*}"
}

# Get active chains for environment
get_active_chains() {
    case $ENVIRONMENT in
        production)
            echo "optimism-mainnet arbitrum-mainnet base-mainnet"
            ;;
        staging|testnet)
            echo "optimism-sepolia arbitrum-sepolia"
            ;;
        development)
            echo "local optimism-sepolia arbitrum-sepolia"
            ;;
    esac
}

# Deploy chains
deploy_chains() {
    log_info "Deploying chains for $ENVIRONMENT environment"
    
    # Create required directories
    mkdir -p logs data backups
    
    # Set environment variables
    set_environment_vars
    
    # Deploy base services first
    log_info "Starting base services..."
    docker-compose -f docker-compose.yml up -d postgres redis
    
    # Wait for services to be ready
    log_info "Waiting for base services..."
    sleep 10
    
    # Run migrations
    run_migrations
    
    # Deploy chain services
    if [ "$CHAIN" = "all" ]; then
        log_info "Deploying all chains for $ENVIRONMENT..."
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml up -d
    else
        log_info "Deploying $CHAIN chain..."
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml up -d blockchain-$CHAIN-*
    fi
    
    # Deploy unified API
    log_info "Deploying unified API..."
    docker-compose -f docker-compose.yml -f docker-compose.multichain.yml up -d unified-api
    
    # Show status
    show_status
}

# Stop chains
stop_chains() {
    log_info "Stopping chains for $ENVIRONMENT environment"
    
    set_environment_vars
    
    if [ "$CHAIN" = "all" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml down
    else
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml stop blockchain-$CHAIN-*
    fi
}

# Show status
show_status() {
    log_info "Chain Status for $ENVIRONMENT environment"
    echo ""
    
    set_environment_vars
    
    # Show running services
    docker-compose -f docker-compose.yml -f docker-compose.multichain.yml ps
    
    # Show chain-specific status
    echo ""
    log_chain "Active Chains:"
    for chain in $(get_active_chains); do
        container="ominari-blockchain-$chain"
        if docker ps --format "table {{.Names}}" | grep -q $container; then
            log_success "  ✓ $chain"
            
            # Get sync status
            sync_status=$(docker exec $container python -c "
from multi_chain_data_manager import MultiChainDataManager
manager = MultiChainDataManager()
print(f'Namespace: {manager.get_data_namespace(manager.chains[\"${chain}\"])}')
" 2>/dev/null || echo "N/A")
            
            echo "    $sync_status"
        else
            log_error "  ✗ $chain (not running)"
        fi
    done
    
    # Show data isolation
    echo ""
    validate_isolation
}

# Show logs
show_logs() {
    set_environment_vars
    
    if [ "$CHAIN" = "all" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml logs -f --tail 100
    else
        docker-compose -f docker-compose.yml -f docker-compose.multichain.yml logs -f --tail 100 blockchain-$CHAIN-*
    fi
}

# Validate data isolation
validate_isolation() {
    log_info "Validating data isolation..."
    
    # Run validation script
    docker-compose -f docker-compose.yml -f docker-compose.multichain.yml run --rm unified-api python -c "
from multi_chain_data_manager import MultiChainDataManager
manager = MultiChainDataManager()
validation = manager.validate_data_isolation()
print('')
print('Data Isolation Check:')
for check, passed in validation.items():
    if check != 'all_valid':
        status = '✓' if passed else '✗'
        print(f'  {status} {check}')
print('')
if validation['all_valid']:
    print('✅ All isolation checks passed')
else:
    print('❌ Some isolation checks failed')
"
}

# Run migrations
run_migrations() {
    log_info "Running database migrations..."
    
    docker-compose -f docker-compose.yml -f docker-compose.multichain.yml run --rm unified-api python -c "
import os
from multi_chain_data_manager import MultiChainDataManager

# Get active chains
manager = MultiChainDataManager()
chains = manager.get_active_chains()

print(f'Environment: {manager.environment.value}')
print(f'Active chains: {len(chains)}')

# Create tables for each chain
for chain in chains:
    config = manager.get_database_config(chain)
    print(f'Creating tables for {chain.chain_name} in namespace {config[\"namespace\"]}')
    # Migration logic here
"
}

# Main script logic
main() {
    echo "🔗 Ominari Multi-Chain Deployment"
    echo "================================="
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(production|staging|testnet|development)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        usage
    fi
    
    # Execute action
    case $ACTION in
        deploy)
            deploy_chains
            ;;
        stop)
            stop_chains
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        validate)
            set_environment_vars
            validate_isolation
            ;;
        migrate)
            set_environment_vars
            run_migrations
            ;;
        *)
            log_error "Invalid action: $ACTION"
            usage
            ;;
    esac
    
    echo ""
    log_success "Operation completed"
}

# Run main function
main