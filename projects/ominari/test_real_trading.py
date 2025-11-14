#!/usr/bin/env python3
"""
Test real trading configuration and balances
Run this after setting up your wallet to verify everything works
"""

import sys
from real_trading_config import RealTradingConfig
from real_trading_engine import RealTradingEngine

def main():
    print("🧪 Testing Real Trading Configuration")
    print("=" * 50)
    
    # Check configuration
    config = RealTradingConfig()
    
    if not config.is_configured():
        print("❌ No wallet configured!")
        print("\nTo configure your wallet, run:")
        print("  ./scripts/setup_wallet.sh")
        sys.exit(1)
        
    print(f"✅ Wallet configured")
    print(f"   Address: {config.get_wallet_address()}")
    print(f"   Mode: {config.get_mode()}")
    print(f"   Emergency Stop: {config.is_emergency_stopped()}")
    
    # Check safety limits
    limits = config.get_safety_limits()
    print(f"\n📊 Safety Limits:")
    print(f"   Max bet: ${limits['max_bet_size_usd']}")
    print(f"   Max daily loss: ${limits['max_daily_loss_usd']}")
    print(f"   Min edge required: {limits['min_edge_required']}%")
    print(f"   Max exposure: {limits['max_exposure_pct']}%")
    
    # Check balances
    print(f"\n💰 Checking Balances...")
    try:
        engine = RealTradingEngine(config)
        
        total_balance = 0
        for network in ['arbitrum', 'optimism', 'base']:
            balance = engine.check_collateral_balance(network)
            print(f"   {network}: ${balance}")
            total_balance += float(balance)
            
        print(f"   TOTAL: ${total_balance:.2f}")
        
        if total_balance == 0:
            print("\n⚠️  No collateral found in wallet")
            print("   To start trading, send USDC (Arbitrum/Base) or sUSD (Optimism)")
            print("   to your wallet address")
        else:
            print("\n✅ Ready to trade!")
            
    except Exception as e:
        print(f"\n❌ Error checking balances: {e}")
        
    # Show next steps
    print("\n📝 Next Steps:")
    if config.get_mode() == 'testnet':
        print("   1. You're in testnet mode (safe practice)")
        print("   2. Fund your wallet with testnet tokens")
        print("   3. Run: python main.py")
    else:
        print("   1. ⚠️  You're in MAINNET mode (real money!)")
        print("   2. Consider switching to testnet first")
        print("   3. When ready, run: python main.py")
        
    print("\n💡 Useful Commands:")
    print("   Switch to testnet: python -c \"from real_trading_config import RealTradingConfig; RealTradingConfig().switch_mode('testnet')\"")
    print("   Emergency stop: python -c \"from real_trading_config import RealTradingConfig; RealTradingConfig().set_emergency_stop(True)\"")
    print("   Check status: curl http://localhost:8888/api/trading-status")


if __name__ == "__main__":
    main()