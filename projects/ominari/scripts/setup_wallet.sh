#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔐 Ominari Wallet Setup${NC}"
echo -e "${BLUE}=========================${NC}"
echo

# Function to validate Ethereum private key
validate_private_key() {
    local key=$1
    # Check if it's a hex string of correct length
    if [[ $key =~ ^0x[a-fA-F0-9]{64}$ ]] || [[ $key =~ ^[a-fA-F0-9]{64}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to test wallet setup
test_wallet() {
    echo -e "\n${BLUE}Testing wallet configuration...${NC}"
    
    cd "$(dirname "$0")/.."
    
    python3 - <<EOF
import sys
sys.path.insert(0, '.')

try:
    from real_trading_config import RealTradingConfig
    from real_trading_engine import RealTradingEngine
    
    config = RealTradingConfig()
    
    if not config.is_configured():
        print("❌ Wallet not configured")
        sys.exit(1)
        
    print(f"✅ Wallet configured: {config.get_wallet_address()}")
    print(f"✅ Mode: {config.get_mode()}")
    print(f"✅ Default network: {config.config['default_network']}")
    
    # Check balances
    print("\n🔍 Checking balances...")
    engine = RealTradingEngine(config)
    
    total_balance = 0.0
    for network in ['arbitrum', 'optimism', 'base']:
        balance = engine.check_collateral_balance(network)
        print(f"  {network}: \${balance}")
        total_balance += float(balance)
    
    if total_balance == 0:
        print("\n⚠️  Warning: No collateral balance found")
        print("   Please fund your wallet with USDC/sUSD to start trading")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF
}

# Main setup flow
echo -e "${YELLOW}⚠️  WARNING: This will configure your wallet for real trading${NC}"
echo -e "${YELLOW}Make sure to:${NC}"
echo -e "  1. Use a dedicated trading wallet (not your main wallet)"
echo -e "  2. Only add funds you're willing to risk"
echo -e "  3. Start with testnet mode first"
echo
read -p "Continue with wallet setup? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 0
fi

# Get trading mode
echo -e "\n${BLUE}Select trading mode:${NC}"
echo "1) Testnet (recommended for testing)"
echo "2) Mainnet (real money)"
read -p "Enter choice (1-2): " mode_choice

case $mode_choice in
    1)
        MODE="testnet"
        echo -e "${GREEN}✓ Testnet mode selected${NC}"
        ;;
    2)
        MODE="mainnet"
        echo -e "${RED}⚠️  MAINNET mode selected - real money at risk!${NC}"
        read -p "Are you SURE you want mainnet? (type 'yes' to confirm): " confirm
        if [[ $confirm != "yes" ]]; then
            echo "Switching to testnet mode for safety"
            MODE="testnet"
        fi
        ;;
    *)
        echo "Invalid choice, defaulting to testnet"
        MODE="testnet"
        ;;
esac

# Get private key
echo -e "\n${BLUE}Enter your wallet private key:${NC}"
echo -e "${YELLOW}Note: The key will be encrypted and stored securely${NC}"
echo -e "Format: 0x... (64 hex characters) or without 0x prefix"
read -s -p "Private key: " PRIVATE_KEY
echo

# Validate private key
if ! validate_private_key "$PRIVATE_KEY"; then
    echo -e "${RED}❌ Invalid private key format${NC}"
    echo "Expected format: 64 hexadecimal characters (with or without 0x prefix)"
    exit 1
fi

# Add 0x prefix if missing
if [[ ! $PRIVATE_KEY =~ ^0x ]]; then
    PRIVATE_KEY="0x$PRIVATE_KEY"
fi

# Configure wallet
echo -e "\n${BLUE}Configuring wallet...${NC}"

cd "$(dirname "$0")/.."

python3 - <<EOF
import sys
sys.path.insert(0, '.')

try:
    from real_trading_config import RealTradingConfig
    
    config = RealTradingConfig()
    
    # Set mode first
    config.switch_mode("$MODE")
    
    # Set wallet
    address = config.set_wallet("$PRIVATE_KEY")
    print(f"✅ Wallet configured successfully!")
    
    # Update default safety limits for testnet
    if "$MODE" == "testnet":
        config.update_safety_limits({
            "max_bet_size_usd": 10.0,
            "max_daily_loss_usd": 50.0,
            "min_edge_required": 1.0
        })
        print("✅ Testnet safety limits applied")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Wallet configuration failed${NC}"
    exit 1
fi

# Test the setup
test_wallet

# Show safety reminder
echo -e "\n${GREEN}✅ Wallet setup complete!${NC}"
echo
echo -e "${BLUE}Safety reminders:${NC}"
echo "• Current mode: $MODE"
echo "• Default network: Arbitrum"
echo "• Max bet size: $100 (mainnet) / $10 (testnet)"
echo "• Max daily loss: $500 (mainnet) / $50 (testnet)"
echo "• Min edge required: 3% (mainnet) / 1% (testnet)"
echo "• Emergency stop: Available via config"
echo
echo -e "${YELLOW}To start trading:${NC}"
echo "1. Fund your wallet with USDC (Arbitrum/Base) or sUSD (Optimism)"
echo "2. Run: python main.py"
echo "3. Monitor trades in the dashboard"
echo
echo -e "${YELLOW}To switch modes:${NC}"
echo "• Testnet: python -c \"from real_trading_config import RealTradingConfig; c=RealTradingConfig(); c.switch_mode('testnet')\""
echo "• Mainnet: python -c \"from real_trading_config import RealTradingConfig; c=RealTradingConfig(); c.switch_mode('mainnet')\""
echo
echo -e "${RED}Emergency stop:${NC}"
echo "python -c \"from real_trading_config import RealTradingConfig; c=RealTradingConfig(); c.set_emergency_stop(True)\""