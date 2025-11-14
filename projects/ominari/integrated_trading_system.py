#!/usr/bin/env python3
"""
Integrated trading system - switches between paper and real trading
Based on wallet configuration and safety settings
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Set up environment
os.environ['PG_PORT'] = '5999'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier
from real_trading_config import RealTradingConfig
from real_trading_engine import RealTradingEngine
from liquidity_aware_trading import LiquidityAwareTradingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """Unified system that handles both paper and real trading"""
    
    def __init__(self):
        self.real_config = RealTradingConfig()
        self.is_real_trading = False
        self.real_engine = None
        self.paper_system = LiquidityAwareTradingSystem()
        self.is_running = False
        
        # Check if real trading is configured and enabled
        if self.real_config.is_configured():
            mode = self.real_config.get_mode()
            if mode == 'mainnet':
                logger.warning("🔴 REAL TRADING MODE ENABLED - REAL MONEY AT RISK!")
                self.is_real_trading = True
                self.real_engine = RealTradingEngine(self.real_config)
            else:
                logger.info("🟡 Testnet mode - simulating real trades without execution")
                self.real_engine = RealTradingEngine(self.real_config)
        else:
            logger.info("🟢 Paper trading mode - no wallet configured")
            
    async def start(self):
        """Start the integrated trading system"""
        self.is_running = True
        
        # Send startup notification with mode info
        mode_info = "PAPER TRADING"
        if self.real_config.is_configured():
            mode_info = f"REAL TRADING ({self.real_config.get_mode().upper()})"
            wallet = self.real_config.get_wallet_address()
            mode_info += f" - Wallet: {wallet[:6]}...{wallet[-4:]}"
            
        discord_notifier.send_startup_message(extra_info=mode_info)
        
        # Check balances if real trading
        if self.real_engine:
            logger.info("Checking collateral balances...")
            total_balance = 0
            for network in ['arbitrum', 'optimism', 'base']:
                balance = self.real_engine.check_collateral_balance(network)
                logger.info(f"  {network}: ${balance}")
                total_balance += float(balance)
                
            if total_balance == 0:
                logger.warning("⚠️  No collateral found - continuing in paper mode")
                self.is_real_trading = False
                
        # Start trading loops
        await self._trading_loop()
        
    async def _trading_loop(self):
        """Main trading loop that handles both paper and real trades"""
        logger.info(f"Starting trading loop (Real: {self.is_real_trading})...")
        
        while self.is_running:
            try:
                # Find opportunities using paper system
                opportunities = await self.paper_system.find_tradeable_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} opportunities")
                    
                    # Process opportunities
                    for opp in opportunities[:3]:  # Top 3
                        if self.is_real_trading and self.real_engine:
                            # Attempt real trade
                            await self._place_real_trade(opp)
                        else:
                            # Paper trade
                            await self.paper_system.place_liquidity_aware_bet(opp)
                            
                # Wait before next cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                discord_notifier.send_error_alert(str(e), "Trading Loop")
                await asyncio.sleep(30)
                
    async def _place_real_trade(self, opportunity: Dict):
        """Place a real trade if all conditions are met"""
        try:
            # Double-check emergency stop
            if self.real_config.is_emergency_stopped():
                logger.warning("Emergency stop active - skipping real trade")
                return
                
            # Place the trade
            result = await self.real_engine.place_real_bet(
                opportunity,
                network=self.real_config.config['default_network']
            )
            
            if result:
                logger.info(f"✅ Real trade executed: {result['tx_hash']}")
                
                # Also record as paper trade for tracking
                await self.paper_system.place_liquidity_aware_bet(opportunity)
            else:
                # Fall back to paper trade
                logger.info("Real trade rejected - placing paper trade instead")
                await self.paper_system.place_liquidity_aware_bet(opportunity)
                
        except Exception as e:
            logger.error(f"Error placing real trade: {e}")
            discord_notifier.send_error_alert(str(e), "Real Trade Execution")
            # Fall back to paper trade
            await self.paper_system.place_liquidity_aware_bet(opportunity)
            
    def emergency_stop(self):
        """Activate emergency stop"""
        if self.real_config.is_configured():
            self.real_config.set_emergency_stop(True)
            self.is_real_trading = False
            discord_notifier.send_error_alert(
                "EMERGENCY STOP ACTIVATED - All real trading halted",
                "Emergency Stop"
            )
            logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            
    def get_status(self) -> Dict:
        """Get current system status"""
        status = {
            'mode': 'paper',
            'real_configured': self.real_config.is_configured(),
            'emergency_stopped': False,
            'wallet': None,
            'balances': {}
        }
        
        if self.real_config.is_configured():
            status['mode'] = self.real_config.get_mode()
            status['wallet'] = self.real_config.get_wallet_address()
            status['emergency_stopped'] = self.real_config.is_emergency_stopped()
            
            if self.real_engine:
                for network in ['arbitrum', 'optimism', 'base']:
                    status['balances'][network] = float(
                        self.real_engine.check_collateral_balance(network)
                    )
                    
        return status


async def main():
    """Main entry point"""
    system = IntegratedTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        system.is_running = False
    except Exception as e:
        logger.error(f"Critical error: {e}")
        system.emergency_stop()
    finally:
        logger.info("Integrated trading system stopped")


if __name__ == "__main__":
    asyncio.run(main())