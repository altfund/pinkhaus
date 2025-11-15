#!/usr/bin/env python3
"""Test portfolio heartbeat once"""

import asyncio
from portfolio_heartbeat import PortfolioHeartbeat

async def test_heartbeat():
    """Test sending one heartbeat"""
    heartbeat = PortfolioHeartbeat()
    
    print("📊 Testing portfolio heartbeat...")
    await heartbeat.send_heartbeat()
    print("✅ Heartbeat test complete - check Discord!")

if __name__ == "__main__":
    asyncio.run(test_heartbeat())