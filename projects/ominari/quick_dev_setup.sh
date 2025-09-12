#!/bin/bash
# Quick Developer Setup - Get Ominari running in under 2 minutes

set -e

echo "🚀 Ominari Quick Developer Setup"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Step 1: Check prerequisites
print_step "Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    print_warning "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

print_success "Prerequisites check passed"

# Step 2: Install dependencies
print_step "Installing Python dependencies..."
uv sync
print_success "Dependencies installed"

# Step 3: Environment setup
print_step "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.template .env
    print_success "Created .env file from template"
else
    print_warning ".env file already exists"
fi

# Step 4: Create logs directory
mkdir -p logs
print_success "Log directory ready"

# Step 5: Check database
if [ -f "sport_odds.db" ]; then
    DB_SIZE=$(du -h sport_odds.db | cut -f1)
    print_success "Found existing database ($DB_SIZE)"
else
    print_warning "No existing database found - will use hybrid setup"
    HYBRID_MODE=true
fi

# Step 6: Start services based on what's available
if [ "$HYBRID_MODE" = true ]; then
    print_step "Starting hybrid architecture..."
    uv run python hybrid_sync_deployment.py
else
    print_step "Starting traditional system..."
    ./start_system.sh
fi

# Step 7: Health check
print_step "Performing health check..."
sleep 10

if curl -s http://localhost:8888/api/status > /dev/null 2>&1; then
    print_success "Web dashboard is running!"
else
    print_warning "Dashboard might still be starting..."
fi

# Final output
echo ""
echo "================================"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo -e "${BLUE}📊 Dashboard:${NC} http://localhost:8888"
echo -e "${BLUE}🔍 Health Check:${NC} ./health_check.sh"
echo -e "${BLUE}📝 View Logs:${NC} ./ominari_logs.sh"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Open http://localhost:8888 in your browser"
echo "2. Check the Markets tab for live data"
echo "3. Use Trading tab to see paper trading activity"
echo "4. Monitor with: ./ominari_logs.sh --all"
echo ""
echo -e "${YELLOW}Need Help?${NC} Check FUTURE_DEVELOPER_GUIDE.md"
echo ""