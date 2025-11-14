#!/usr/bin/env python3
"""
Liquidity-aware trading system
Considers actual blockchain liquidity when placing trades
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

# Set up environment
os.environ['PG_PORT'] = '5999'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier
from real_odds_fetcher import RealOddsFetcher
from blockchain_liquidity_fetcher import BlockchainLiquidityFetcher
from paper_trading_live import LivePaperTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiquidityAwareTradingSystem:
    """Trading system that considers liquidity constraints"""
    
    def __init__(self):
        self.odds_fetcher = RealOddsFetcher()
        self.liquidity_fetcher = BlockchainLiquidityFetcher()
        self.paper_trader = LivePaperTrader()
        self.bankroll_config = BankrollConfig()
        self.is_running = False
        
        # Trading parameters with liquidity constraints
        self.min_edge = 2.0  # Minimum 2% edge
        self.max_slippage = 1.0  # Maximum 1% slippage allowed
        self.min_liquidity = 100.0  # Minimum $100 liquidity required
        self.max_bet_pct = 5.0  # Max 5% of bankroll per bet
        self.liquidity_factor = 0.1  # Max 10% of available liquidity
        self.kelly_fraction = 0.15  # More conservative Kelly due to liquidity
        
        # Update paper trader with liquidity-aware settings
        self.paper_trader.min_edge = self.min_edge
        self.paper_trader.kelly_fraction = self.kelly_fraction
        
    def calculate_liquidity_adjusted_bet(self, 
                                       opportunity: Dict,
                                       liquidity_data: Dict,
                                       bankroll: float) -> Tuple[float, str]:
        """
        Calculate bet size considering liquidity constraints
        Returns (bet_amount, reason)
        """
        # Start with Kelly criterion bet
        kelly_bet = self.paper_trader.calculate_kelly_bet(
            opportunity['fair_prob'],
            opportunity['odds'],
            bankroll
        )
        
        # Apply bankroll constraint
        max_bankroll_bet = bankroll * self.max_bet_pct / 100
        bet_size = min(kelly_bet, max_bankroll_bet)
        
        # Apply liquidity constraints
        if liquidity_data:
            # Check max liquidity
            max_liquidity = liquidity_data.get('max_liquidity_usd', 0)
            if max_liquidity < self.min_liquidity:
                return 0, f"Insufficient liquidity: ${max_liquidity:.2f}"
                
            # Apply liquidity factor constraint
            max_liquidity_bet = max_liquidity * self.liquidity_factor
            
            # Get optimal size within slippage tolerance
            outcome_map = {'home': 0, 'away': 1, 'draw': 2}
            outcome_idx = outcome_map.get(opportunity['outcome'], 0)
            
            optimal = self.liquidity_fetcher.get_optimal_bet_size(
                liquidity_data.get('market_address'),
                outcome_idx,
                self.max_slippage,
                liquidity_data.get('network', 'arbitrum')
            )
            
            if optimal['optimal_size'] == 0:
                return 0, optimal['reason']
                
            # Take minimum of all constraints
            bet_size = min(bet_size, max_liquidity_bet, optimal['optimal_size'])
            
            # Final check
            if bet_size < self.paper_trader.min_bet:
                return 0, f"Bet size ${bet_size:.2f} below minimum"
                
            return bet_size, f"Liquidity-adjusted (slippage: {optimal['slippage']:.2f}%)"
        else:
            # No liquidity data, use conservative estimate
            bet_size = min(bet_size, 100)  # Cap at $100 without liquidity data
            return bet_size, "No liquidity data (conservative sizing)"
            
    async def find_tradeable_opportunities(self) -> List[Dict]:
        """Find opportunities that meet edge AND liquidity requirements"""
        # Get base opportunities from paper trader
        opportunities = await self.paper_trader.find_betting_opportunities()
        
        tradeable = []
        
        for opp in opportunities:
            if opp['edge'] < self.min_edge:
                continue
                
            # Get liquidity data if available
            market = opp['market']
            liquidity_data = None
            
            # Check if we have blockchain address
            if market.source_id.startswith('v2_0x') or market.source_id.startswith('0x'):
                market_address = market.source_id[3:] if market.source_id.startswith('v2_') else market.source_id
                
                # Try to get cached liquidity from metadata
                if hasattr(market, 'metadata') and market.metadata:
                    liq_meta = market.metadata.get('liquidity', {})
                    if liq_meta and 'last_updated' in liq_meta:
                        # Check if data is recent (< 5 minutes old)
                        last_updated = datetime.fromisoformat(liq_meta['last_updated'])
                        if (datetime.now(timezone.utc) - last_updated).seconds < 300:
                            liquidity_data = {
                                'market_address': market_address,
                                'network': liq_meta.get('network', 'arbitrum'),
                                'max_liquidity_usd': liq_meta.get('max_usd', 0),
                                'total_pool_size': liq_meta.get('pool_size', 0)
                            }
                            
            # Calculate liquidity-adjusted bet size
            bet_size, reason = self.calculate_liquidity_adjusted_bet(
                opp,
                liquidity_data,
                self.bankroll_config.get_current_bankroll()
            )
            
            if bet_size > 0:
                opp['adjusted_bet_size'] = bet_size
                opp['liquidity_reason'] = reason
                opp['liquidity_data'] = liquidity_data
                tradeable.append(opp)
            else:
                logger.info(f"Skipping {market.home_team} vs {market.away_team}: {reason}")
                
        # Sort by edge, considering liquidity quality
        tradeable.sort(key=lambda x: (
            x['edge'] * (1 if x['liquidity_data'] else 0.5),  # Prefer markets with liquidity data
            -x['adjusted_bet_size']  # Prefer larger bet sizes
        ), reverse=True)
        
        return tradeable
        
    async def place_liquidity_aware_bet(self, opportunity: Dict) -> Optional[Bet]:
        """Place a bet with liquidity considerations"""
        market = opportunity['market']
        bet_size = opportunity['adjusted_bet_size']
        
        logger.info(
            f"Placing liquidity-aware bet: ${bet_size:.2f} on "
            f"{market.home_team} vs {market.away_team} - {opportunity['outcome']} "
            f"@ {opportunity['odds']:.2f} (edge: {opportunity['edge']:.2f}%, "
            f"{opportunity['liquidity_reason']})"
        )
        
        # Create bet through paper trader
        with db_manager.get_db_session() as db:
            bet = Bet(
                session_id=self.paper_trader.session_id,
                source_id=market.source_id,
                sport=market.sport,
                league=market.league,
                home_team=market.home_team,
                away_team=market.away_team,
                outcome=opportunity['outcome'],
                stake=bet_size,
                decimal_odds=opportunity['odds'],
                placed_at=datetime.now(timezone.utc),
                status='pending',
                metadata={
                    'edge': opportunity['edge'],
                    'fair_prob': opportunity['fair_prob'],
                    'liquidity_adjusted': True,
                    'liquidity_reason': opportunity['liquidity_reason']
                }
            )
            db.add(bet)
            db.commit()
            
            # Update bankroll
            self.paper_trader.trades_placed += 1
            self.paper_trader.last_trade_time = datetime.now(timezone.utc)
            
            return bet
            
    async def start(self):
        """Start the liquidity-aware trading system"""
        logger.info("Starting Liquidity-Aware Trading System...")
        self.is_running = True
        
        # Send startup notification
        discord_notifier.send_startup_message()
        
        # Create trading session
        await self._create_session()
        
        # Start concurrent tasks
        await asyncio.gather(
            self._liquidity_update_loop(),
            self._trading_loop(),
            self._monitoring_loop()
        )
        
    async def _create_session(self):
        """Create a new trading session"""
        with db_manager.get_db_session() as db:
            session = BettingSession(
                name=f"Liquidity Trading {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                bankroll=self.bankroll_config.get_current_bankroll(),
                strategy_name="liquidity_aware_kelly",
                session_type='paper',
                created_at=datetime.now(timezone.utc)
            )
            db.add(session)
            db.commit()
            self.paper_trader.session_id = session.id
            logger.info(f"Created liquidity-aware trading session: {session.id}")
            
    async def _liquidity_update_loop(self):
        """Update liquidity data from blockchain"""
        logger.info("Starting liquidity update loop...")
        
        while self.is_running:
            try:
                # Update real odds
                self.odds_fetcher.update_database_with_real_odds()
                
                # Update liquidity data
                self.liquidity_fetcher.update_database_with_liquidity()
                
                # Wait before next update
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in liquidity update: {e}")
                await asyncio.sleep(60)
                
    async def _trading_loop(self):
        """Main trading loop with liquidity awareness"""
        logger.info("Starting liquidity-aware trading loop...")
        
        while self.is_running:
            try:
                # Find tradeable opportunities
                opportunities = await self.find_tradeable_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} tradeable opportunities")
                    
                    # Place bets on best opportunities
                    bets_placed = 0
                    for opp in opportunities[:3]:  # Top 3 opportunities
                        bet = await self.place_liquidity_aware_bet(opp)
                        if bet:
                            bets_placed += 1
                            
                            # Send Discord notification
                            discord_notifier.send_trade_alert({
                                'type': 'NEW',
                                'market': f"{opp['market'].home_team} vs {opp['market'].away_team}",
                                'outcome': opp['outcome'],
                                'amount': bet.stake,
                                'odds': bet.decimal_odds,
                                'edge': opp['edge'],
                                'bankroll': self.bankroll_config.get_current_bankroll(),
                                'liquidity_info': opp['liquidity_reason']
                            })
                            
                    if bets_placed == 0:
                        logger.info("No bets placed due to liquidity constraints")
                else:
                    logger.info("No opportunities meet edge and liquidity requirements")
                    
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
                
    async def _monitoring_loop(self):
        """Monitor performance and liquidity metrics"""
        logger.info("Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Get trading statistics
                stats = self._get_trading_stats()
                
                # Add liquidity metrics
                with db_manager.get_db_session() as db:
                    # Count markets with good liquidity
                    markets_with_liquidity = 0
                    total_liquidity = 0
                    
                    active_markets = db.query(Market).filter(
                        Market.is_active == True
                    ).limit(100).all()
                    
                    for market in active_markets:
                        if hasattr(market, 'metadata') and market.metadata:
                            liq = market.metadata.get('liquidity', {})
                            if liq.get('max_usd', 0) > self.min_liquidity:
                                markets_with_liquidity += 1
                                total_liquidity += liq.get('max_usd', 0)
                                
                stats['liquidity_metrics'] = {
                    'markets_with_liquidity': markets_with_liquidity,
                    'total_available_liquidity': total_liquidity,
                    'avg_liquidity': total_liquidity / markets_with_liquidity if markets_with_liquidity > 0 else 0
                }
                
                logger.info(
                    f"Performance - Bankroll: ${stats['bankroll']:.2f} | "
                    f"P&L: ${stats['pnl']:.2f} | ROI: {stats['roi']:.2f}% | "
                    f"Markets with liquidity: {markets_with_liquidity} | "
                    f"Avg liquidity: ${stats['liquidity_metrics']['avg_liquidity']:.2f}"
                )
                
                # Wait before next check
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                await asyncio.sleep(60)
                
    def _get_trading_stats(self) -> Dict:
        """Get current trading statistics"""
        with db_manager.get_db_session() as db:
            # Get all bets from current session
            all_bets = db.query(Bet).filter(
                Bet.session_id == self.paper_trader.session_id
            ).all()
            
            # Calculate statistics
            total_bets = len(all_bets)
            liquidity_adjusted_bets = sum(
                1 for b in all_bets 
                if b.metadata and b.metadata.get('liquidity_adjusted')
            )
            
            current_bankroll = self.bankroll_config.get_current_bankroll()
            initial_bankroll = self.bankroll_config.initial_bankroll
            
            return {
                'bankroll': current_bankroll,
                'pnl': current_bankroll - initial_bankroll,
                'roi': ((current_bankroll - initial_bankroll) / initial_bankroll * 100),
                'total_bets': total_bets,
                'liquidity_adjusted_bets': liquidity_adjusted_bets,
                'liquidity_adjustment_rate': (liquidity_adjusted_bets / total_bets * 100) if total_bets > 0 else 0
            }
            

async def main():
    """Main entry point"""
    system = LiquidityAwareTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    finally:
        system.is_running = False
        logger.info("Liquidity-aware trading system stopped")


if __name__ == "__main__":
    asyncio.run(main())