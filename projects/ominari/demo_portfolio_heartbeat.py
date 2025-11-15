#!/usr/bin/env python3
"""
Demo the portfolio heartbeat system
Shows what data is collected and sent to Discord
"""

import asyncio
from datetime import datetime, timezone
from portfolio_heartbeat import PortfolioHeartbeat

async def demo_heartbeat():
    """Demo all components of the heartbeat"""
    heartbeat = PortfolioHeartbeat()
    
    print("🫀 PORTFOLIO HEARTBEAT DEMO")
    print("=" * 60)
    
    # 1. Portfolio Stats
    print("\n📊 Portfolio Statistics:")
    stats = heartbeat.get_portfolio_stats()
    print(f"  Bankroll: ${stats['current_bankroll']:,.2f}")
    print(f"  P&L: ${stats['pnl']:+,.2f} ({stats['roi']:+.2f}%)")
    print(f"  Total Bets: {stats['total_bets']}")
    print(f"  Total Staked: ${stats['total_staked']:,.2f}")
    print(f"  Average Bet: ${stats['avg_bet_size']:.2f}")
    print(f"  Session ID: {stats.get('session_id', 'N/A')}")
    
    # 2. Market Coverage
    print("\n🏆 Market Coverage:")
    markets = heartbeat.get_active_markets_count()
    print(f"  Active Markets: {markets['total']}")
    print(f"  Markets with Odds: {markets['with_odds']}")
    print("  By Sport:")
    for sport, count in list(markets['by_sport'].items())[:5]:
        print(f"    • {sport}: {count}")
    
    # 3. Edge Opportunities
    print("\n🎯 Edge Opportunities:")
    opportunities = heartbeat.get_edge_opportunities()
    print(f"  Total Positive Edge Markets: {opportunities['total_count']}")
    print(f"  High Edge (>10%): {len(opportunities['high_edge'])}")
    print(f"  Medium Edge (5-10%): {len(opportunities['medium_edge'])}")
    print(f"  Low Edge (0-5%): {len(opportunities['low_edge'])}")
    
    if opportunities['high_edge']:
        print("\n  Best Opportunity:")
        best = opportunities['high_edge'][0]
        print(f"    {best['teams']}")
        print(f"    Edge: {best['edge']:.2f}%")
        print(f"    Total Probability: {best['total_prob']:.3f}")
    
    # 4. Blockchain Stats
    print("\n🔗 Blockchain Data (from public APIs):")
    blockchain = heartbeat.get_blockchain_stats()
    if blockchain:
        if 'arbitrum_gas_gwei' in blockchain:
            print(f"  Arbitrum Gas: {blockchain['arbitrum_gas_gwei']} Gwei")
        if 'eth_price_usd' in blockchain:
            print(f"  ETH Price: ${blockchain['eth_price_usd']:,.2f}")
    else:
        print("  (No blockchain data available)")
    
    # 5. Send actual heartbeat
    print("\n📢 Sending heartbeat to Discord...")
    await heartbeat.send_heartbeat()
    print("✅ Heartbeat sent!")
    
    print("\n" + "=" * 60)
    print("📝 Summary:")
    print("  • Portfolio tracking with P&L and ROI")
    print("  • Real-time market coverage statistics")
    print("  • Edge opportunity detection and ranking")
    print("  • Blockchain gas prices and ETH price")
    print("  • All data from actual database, not mocked")
    print("\n🔄 In production, this runs every hour automatically")

if __name__ == "__main__":
    asyncio.run(demo_heartbeat())