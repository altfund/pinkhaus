#!/usr/bin/env python3
"""
Integrated Trading System with Real Odds
Combines real odds fetching with paper trading
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
from real_odds_fetcher import RealOddsFetcher
from paper_trading_live import LivePaperTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """Combines real odds fetching with paper trading"""
    
    def __init__(self):
        self.odds_fetcher = RealOddsFetcher()
        self.paper_trader = LivePaperTrader()
        self.bankroll_config = BankrollConfig()
        self.is_running = False
        
        # Update paper trader settings for realistic trading
        self.paper_trader.min_edge = 2.0  # Require 2% minimum edge
        self.paper_trader.max_exposure_pct = 20.0  # Max 20% exposure
        self.paper_trader.kelly_fraction = 0.2  # Conservative Kelly
        
    async def start(self):
        """Start the integrated trading system"""
        logger.info("Starting Integrated Trading System...")
        self.is_running = True
        
        # Send startup notification
        discord_notifier.send_startup_message()
        
        # Create trading session
        await self._create_session()
        
        # Start concurrent tasks
        await asyncio.gather(
            self._odds_update_loop(),
            self._trading_loop(),
            self._monitoring_loop()
        )
        
    async def _create_session(self):
        """Create a new trading session"""
        with db_manager.get_db_session() as db:
            session = BettingSession(
                name=f"Integrated Trading {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                bankroll=self.bankroll_config.get_current_bankroll(),
                strategy_name="integrated_real_odds",
                session_type='paper',
                created_at=datetime.now(timezone.utc)
            )
            db.add(session)
            db.commit()
            self.paper_trader.session_id = session.id
            logger.info(f"Created trading session: {session.id}")
            
    async def _odds_update_loop(self):
        """Continuously update odds from real sources"""
        logger.info("Starting odds update loop...")
        
        while self.is_running:
            try:
                # Fetch and update real odds
                logger.info("Fetching real odds...")
                self.odds_fetcher.update_database_with_real_odds()
                
                # Report statistics
                with db_manager.get_db_session() as db:
                    total_markets = db.query(Market).filter(Market.is_active == True).count()
                    markets_with_odds = db.query(Market).join(
                        Odd, Market.source_id == Odd.source_market_id
                    ).filter(Market.is_active == True).distinct().count()
                    
                    logger.info(f"Active markets: {total_markets}, with odds: {markets_with_odds}")
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in odds update loop: {e}")
                await asyncio.sleep(30)
                
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        while self.is_running:
            try:
                # Find trading opportunities
                opportunities = await self.paper_trader.find_betting_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} opportunities")
                    
                    # Place bets on best opportunities
                    for opp in opportunities[:3]:  # Top 3 opportunities
                        if opp['edge'] >= self.paper_trader.min_edge:
                            bet = await self.paper_trader.place_bet(opp)
                            if bet:
                                # Send Discord notification
                                discord_notifier.send_trade_alert({
                                    'type': 'NEW',
                                    'market': f"{opp['market'].home_team} vs {opp['market'].away_team}",
                                    'outcome': opp['outcome'],
                                    'amount': bet.stake,
                                    'odds': bet.decimal_odds,
                                    'edge': opp['edge'],
                                    'bankroll': self.bankroll_config.get_current_bankroll()
                                })
                else:
                    logger.info("No opportunities found (all edges below threshold)")
                    
                # Check for completed bets
                await self._check_completed_bets()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
                
    async def _check_completed_bets(self):
        """Check and settle completed bets"""
        with db_manager.get_db_session() as db:
            # Get active bets
            active_bets = db.query(Bet).filter(
                Bet.session_id == self.paper_trader.session_id,
                Bet.status == 'pending'
            ).all()
            
            for bet in active_bets:
                # Check if market is resolved
                market = db.query(Market).filter(
                    Market.source_id == bet.source_id
                ).first()
                
                if market and not market.is_active and market.winning_outcome:
                    # Settle bet
                    won = (bet.outcome == market.winning_outcome)
                    payout = bet.stake * bet.decimal_odds if won else 0
                    
                    bet.status = 'won' if won else 'lost'
                    bet.payout = payout
                    bet.resolved_at = datetime.now(timezone.utc)
                    
                    # Update bankroll
                    if won:
                        self.bankroll_config.record_bet_result(True, payout - bet.stake)
                    else:
                        self.bankroll_config.record_bet_result(False, -bet.stake)
                        
                    # Send notification
                    discord_notifier.send_trade_alert({
                        'type': 'CLOSE',
                        'market': f"{market.home_team} vs {market.away_team}",
                        'outcome': bet.outcome,
                        'amount': bet.stake,
                        'odds': bet.decimal_odds,
                        'edge': 0,  # We don't store edge in bet
                        'won': won,
                        'pnl': payout - bet.stake if won else -bet.stake,
                        'bankroll': self.bankroll_config.get_current_bankroll()
                    })
                    
                    logger.info(f"Settled bet: {'WON' if won else 'LOST'} ${payout:.2f}")
                    
            db.commit()
            
    async def _monitoring_loop(self):
        """Monitor and report system performance"""
        logger.info("Starting monitoring loop...")
        
        last_summary_date = datetime.now(timezone.utc).date()
        
        while self.is_running:
            try:
                # Get current stats
                stats = self._get_trading_stats()
                
                # Log performance
                logger.info(
                    f"Performance - Bankroll: ${stats['bankroll']:.2f} | "
                    f"P&L: ${stats['pnl']:.2f} | ROI: {stats['roi']:.2f}% | "
                    f"Bets: {stats['total_bets']} | Win Rate: {stats['win_rate']:.1f}%"
                )
                
                # Send daily summary at midnight
                current_date = datetime.now(timezone.utc).date()
                if current_date > last_summary_date:
                    discord_notifier.send_daily_summary(stats)
                    last_summary_date = current_date
                    
                # Wait before next check
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
                
    def _get_trading_stats(self) -> Dict:
        """Get current trading statistics"""
        with db_manager.get_db_session() as db:
            # Get all bets from current session
            all_bets = db.query(Bet).filter(
                Bet.session_id == self.paper_trader.session_id
            ).all()
            
            total_bets = len(all_bets)
            pending_bets = sum(1 for b in all_bets if b.status == 'pending')
            won_bets = sum(1 for b in all_bets if b.status == 'won')
            lost_bets = sum(1 for b in all_bets if b.status == 'lost')
            
            # Calculate P&L
            total_staked = sum(b.stake for b in all_bets)
            total_payout = sum(b.payout or 0 for b in all_bets if b.status in ['won', 'lost'])
            pnl = total_payout - sum(b.stake for b in all_bets if b.status in ['won', 'lost'])
            
            # Get current bankroll
            current_bankroll = self.bankroll_config.get_current_bankroll()
            initial_bankroll = self.bankroll_config.initial_bankroll
            
            # Calculate win rate
            completed_bets = won_bets + lost_bets
            win_rate = (won_bets / completed_bets * 100) if completed_bets > 0 else 0
            
            # Get top trades
            top_trades = []
            for bet in sorted(all_bets, key=lambda x: abs(x.payout - x.stake) if x.payout else 0, reverse=True)[:3]:
                if bet.status in ['won', 'lost']:
                    market = db.query(Market).filter(Market.source_id == bet.source_id).first()
                    if market:
                        top_trades.append({
                            'market': f"{market.home_team} vs {market.away_team}",
                            'outcome': bet.outcome,
                            'odds': bet.decimal_odds,
                            'pnl': bet.payout - bet.stake if bet.status == 'won' else -bet.stake
                        })
                        
            return {
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'bankroll': current_bankroll,
                'start_bankroll': initial_bankroll,
                'end_bankroll': current_bankroll,
                'pnl': current_bankroll - initial_bankroll,
                'daily_pnl': pnl,  # For today only
                'roi': ((current_bankroll - initial_bankroll) / initial_bankroll * 100),
                'total_bets': total_bets,
                'active_positions': pending_bets,
                'winning_trades': won_bets,
                'total_trades': completed_bets,
                'win_rate': win_rate,
                'wins': won_bets,
                'total_exposure': sum(b.stake for b in all_bets if b.status == 'pending'),
                'total_roi': ((current_bankroll - initial_bankroll) / initial_bankroll * 100),
                'top_trades': top_trades
            }
            
    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping Integrated Trading System...")
        self.is_running = False
        
        # Send final summary
        stats = self._get_trading_stats()
        discord_notifier.send_daily_summary(stats)
        
        logger.info("Trading system stopped")


async def main():
    """Main entry point"""
    system = IntegratedTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())