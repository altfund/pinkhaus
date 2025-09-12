#!/usr/bin/env python3
"""
Blockchain Paper Trading System

A comprehensive paper trading system that uses:
- Direct blockchain data collection
- Enhanced signal providers with metadata
- Team metadata and venue information
- Real-time odds from multiple chains
"""

import logging
import json
import pandas as pd
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from database_v2 import db_manager
from models import Market, Odd
from signals import get_signal_providers, SIGNAL_WEIGHTS
from market_enrichment import MarketEnrichmentService
from blockchain_reader import BlockchainReader
from enhanced_tag_mappings import tag_mapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a paper trade."""
    trade_id: str
    market_id: str
    blockchain_address: str
    network: str
    
    # Market info
    home_team: str
    away_team: str
    sport: str
    league: str
    
    # Trade details
    position: str  # 'home', 'away', 'draw'
    odds: float
    stake: float
    expected_payout: float
    
    # Signal info
    signal_probability: float
    signal_providers_used: List[str]
    kelly_fraction: float
    
    # Timing
    placed_at: datetime
    market_starts_at: datetime
    expires_at: datetime
    
    # Settlement
    is_settled: bool = False
    result: Optional[str] = None  # 'win', 'lose', 'push'
    pnl: float = 0.0
    settled_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'market_id': self.market_id,
            'blockchain_address': self.blockchain_address,
            'network': self.network,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'sport': self.sport,
            'league': self.league,
            'position': self.position,
            'odds': self.odds,
            'stake': self.stake,
            'expected_payout': self.expected_payout,
            'signal_probability': self.signal_probability,
            'signal_providers_used': self.signal_providers_used,
            'kelly_fraction': self.kelly_fraction,
            'placed_at': self.placed_at.isoformat(),
            'market_starts_at': self.market_starts_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'is_settled': self.is_settled,
            'result': self.result,
            'pnl': self.pnl,
            'settled_at': self.settled_at.isoformat() if self.settled_at else None
        }


class BlockchainPaperTradingEngine:
    """Paper trading engine using blockchain data."""
    
    def __init__(self, 
                 starting_bankroll: float = 10000.0,
                 max_bet_percentage: float = 0.05,  # Max 5% of bankroll per bet
                 kelly_fraction: float = 0.25):     # Quarter Kelly
        
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.max_bet_percentage = max_bet_percentage
        self.kelly_fraction = kelly_fraction
        
        # Services
        self.enrichment_service = MarketEnrichmentService()
        self.signal_providers = get_signal_providers()
        self.signal_weights = SIGNAL_WEIGHTS
        
        # Trading state
        self.active_trades: Dict[str, PaperTrade] = {}
        self.settled_trades: List[PaperTrade] = []
        self.trade_counter = 0
        
        logger.info(f"🚀 Initialized Blockchain Paper Trading Engine")
        logger.info(f"   Starting bankroll: ${starting_bankroll:,.2f}")
        logger.info(f"   Max bet percentage: {max_bet_percentage:.1%}")
        logger.info(f"   Kelly fraction: {kelly_fraction:.2f}")
        logger.info(f"   Signal providers: {len(self.signal_providers)}")
    
    def scan_for_trading_opportunities(self, min_edge: float = 0.05) -> List[Dict]:
        """Scan blockchain markets for trading opportunities."""
        logger.info("🔍 Scanning blockchain markets for opportunities...")
        
        opportunities = []
        
        # Get active blockchain markets
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.source.like('blockchain_%'),
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).limit(20).all()  # Limit to avoid overload
            
            logger.info(f"Found {len(markets)} active blockchain markets")
            
            for market in markets:
                try:
                    opportunity = self._analyze_market_opportunity(market, min_edge)
                    if opportunity:
                        opportunities.append(opportunity)
                except Exception as e:
                    logger.error(f"Error analyzing market {market.source_id}: {e}")
        
        logger.info(f"Found {len(opportunities)} trading opportunities")
        return opportunities
    
    def _analyze_market_opportunity(self, market: Market, min_edge: float) -> Optional[Dict]:
        """Analyze a single market for trading opportunities."""
        
        # Get enriched market data
        network = market.source.replace('blockchain_', '')
        market_address = market.source_id.replace('blockchain_', '')
        
        enriched = self.enrichment_service.enrich_market(market_address, network)
        if not enriched:
            return None
        
        # Get current odds
        current_odds = enriched.current_odds
        if not current_odds or not any(current_odds.values()):
            return None
        
        # Prepare data for signal providers
        market_data = pd.DataFrame([{
            'source_id': market.source_id,
            'source': market.source,
            'sport': market.sport,
            'league_name': market.league_name,
            'home_team': market.home_team,
            'away_team': market.away_team,
            'market_type': market.market_type,
            'maturity_date': market.maturity_date,
            'normalized_outcome': 'Home',  # Analyze home position
            'time': datetime.now(timezone.utc),
            'odds': current_odds.get('home', 0)
        }])
        
        # Get signal probabilities
        signal_results = {}
        weighted_probability = 0.0
        total_weight = 0.0
        
        for provider in self.signal_providers:
            try:
                probs = provider.get_probs(market_data)
                if len(probs) > 0:
                    prob = probs.iloc[0]
                    weight = self.signal_weights.get(provider.name, 1.0)
                    
                    signal_results[provider.name] = {
                        'probability': prob,
                        'weight': weight
                    }
                    
                    weighted_probability += prob * weight
                    total_weight += weight
            except Exception as e:
                logger.warning(f"Signal provider {provider.name} failed: {e}")
        
        if total_weight == 0:
            return None
        
        # Calculate weighted average probability
        avg_probability = weighted_probability / total_weight
        
        # Calculate edge for home position
        home_odds = current_odds.get('home', 0)
        if home_odds <= 1.0:
            return None
        
        implied_probability = 1.0 / home_odds
        edge = avg_probability - implied_probability
        
        # Check if edge meets minimum threshold
        if edge >= min_edge:
            # Calculate kelly bet size
            kelly_bet_size = self._calculate_kelly_bet_size(
                avg_probability, home_odds, self.kelly_fraction
            )
            
            # Limit bet size to max percentage of bankroll
            max_bet = self.current_bankroll * self.max_bet_percentage
            recommended_stake = min(kelly_bet_size, max_bet)
            
            if recommended_stake >= 10.0:  # Minimum $10 bet
                return {
                    'market': market,
                    'enriched': enriched,
                    'position': 'home',
                    'odds': home_odds,
                    'signal_probability': avg_probability,
                    'implied_probability': implied_probability,
                    'edge': edge,
                    'recommended_stake': recommended_stake,
                    'kelly_fraction': kelly_bet_size / self.current_bankroll,
                    'signal_results': signal_results,
                    'expected_value': recommended_stake * edge
                }
        
        return None
    
    def _calculate_kelly_bet_size(self, win_prob: float, odds: float, fraction: float = 1.0) -> float:
        """Calculate Kelly criterion bet size."""
        if win_prob <= 0 or odds <= 1.0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction of capital to wager
        #       b = odds received - 1 (decimal odds - 1)
        #       p = probability of winning
        #       q = probability of losing = 1 - p
        
        b = odds - 1.0
        p = win_prob
        q = 1.0 - p
        
        kelly_fraction_optimal = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly_fraction_used = kelly_fraction_optimal * fraction
        
        # Ensure non-negative
        kelly_fraction_used = max(0.0, kelly_fraction_used)
        
        # Calculate bet size
        bet_size = self.current_bankroll * kelly_fraction_used
        
        return bet_size
    
    def place_paper_trade(self, opportunity: Dict) -> PaperTrade:
        """Place a paper trade based on an opportunity."""
        
        self.trade_counter += 1
        market = opportunity['market']
        enriched = opportunity['enriched']
        
        trade = PaperTrade(
            trade_id=f"BT{self.trade_counter:06d}",
            market_id=market.source_id,
            blockchain_address=enriched.blockchain_address,
            network=enriched.network,
            home_team=enriched.home_team.full_name,
            away_team=enriched.away_team.full_name,
            sport=enriched.sport.name,
            league=enriched.league.name,
            position=opportunity['position'],
            odds=opportunity['odds'],
            stake=opportunity['recommended_stake'],
            expected_payout=opportunity['recommended_stake'] * opportunity['odds'],
            signal_probability=opportunity['signal_probability'],
            signal_providers_used=list(opportunity['signal_results'].keys()),
            kelly_fraction=opportunity['kelly_fraction'],
            placed_at=datetime.now(timezone.utc),
            market_starts_at=market.maturity_date,
            expires_at=market.maturity_date
        )
        
        # Deduct stake from bankroll
        self.current_bankroll -= trade.stake
        
        # Add to active trades
        self.active_trades[trade.trade_id] = trade
        
        logger.info(f"📈 Placed paper trade {trade.trade_id}:")
        logger.info(f"   {trade.home_team} vs {trade.away_team}")
        logger.info(f"   Position: {trade.position} @ {trade.odds:.2f}")
        logger.info(f"   Stake: ${trade.stake:.2f}")
        logger.info(f"   Expected payout: ${trade.expected_payout:.2f}")
        logger.info(f"   Signal probability: {trade.signal_probability:.3f}")
        logger.info(f"   Edge: {opportunity['edge']:.3f}")
        logger.info(f"   Remaining bankroll: ${self.current_bankroll:.2f}")
        
        return trade
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        
        total_exposure = sum(trade.stake for trade in self.active_trades.values())
        unrealized_pnl = sum(
            trade.expected_payout - trade.stake 
            for trade in self.active_trades.values()
        )
        
        realized_pnl = sum(trade.pnl for trade in self.settled_trades)
        
        total_trades = len(self.active_trades) + len(self.settled_trades)
        win_count = len([t for t in self.settled_trades if t.result == 'win'])
        win_rate = win_count / len(self.settled_trades) if self.settled_trades else 0.0
        
        return {
            'starting_bankroll': self.starting_bankroll,
            'current_bankroll': self.current_bankroll,
            'total_exposure': total_exposure,
            'available_cash': self.current_bankroll,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': realized_pnl,
            'total_trades': total_trades,
            'active_trades': len(self.active_trades),
            'settled_trades': len(self.settled_trades),
            'win_rate': win_rate,
            'roi': (realized_pnl / self.starting_bankroll) if self.starting_bankroll > 0 else 0.0
        }
    
    def run_trading_session(self, max_trades: int = 5) -> Dict:
        """Run a trading session."""
        logger.info(f"🎯 Starting trading session (max {max_trades} trades)")
        
        # Scan for opportunities
        opportunities = self.scan_for_trading_opportunities()
        
        # Sort by expected value (descending)
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # Take best opportunities up to max_trades
        placed_trades = []
        for i, opp in enumerate(opportunities[:max_trades]):
            try:
                trade = self.place_paper_trade(opp)
                placed_trades.append(trade)
            except Exception as e:
                logger.error(f"Failed to place trade {i+1}: {e}")
        
        # Get portfolio summary
        portfolio = self.get_portfolio_summary()
        
        session_summary = {
            'session_timestamp': datetime.now(timezone.utc),
            'opportunities_found': len(opportunities),
            'trades_placed': len(placed_trades),
            'trades': [trade.to_dict() for trade in placed_trades],
            'portfolio': portfolio
        }
        
        logger.info(f"✅ Trading session complete:")
        logger.info(f"   Opportunities found: {len(opportunities)}")
        logger.info(f"   Trades placed: {len(placed_trades)}")
        logger.info(f"   Total exposure: ${portfolio['total_exposure']:.2f}")
        logger.info(f"   Available cash: ${portfolio['available_cash']:.2f}")
        
        return session_summary


def demo_blockchain_paper_trading():
    """Demo the blockchain paper trading system."""
    logger.info("🚀 Blockchain Paper Trading System Demo")
    logger.info("=" * 60)
    
    # Create paper trading engine
    engine = BlockchainPaperTradingEngine(
        starting_bankroll=10000.0,
        max_bet_percentage=0.05,  # 5% max per bet
        kelly_fraction=0.25       # Quarter Kelly
    )
    
    # Run trading session
    session = engine.run_trading_session(max_trades=3)
    
    # Display results
    print("\n" + "="*60)
    print("BLOCKCHAIN PAPER TRADING SESSION RESULTS")
    print("="*60)
    
    print(f"📊 Session Summary:")
    print(f"   Opportunities found: {session['opportunities_found']}")
    print(f"   Trades placed: {session['trades_placed']}")
    
    portfolio = session['portfolio']
    print(f"\n💰 Portfolio Status:")
    print(f"   Starting bankroll: ${portfolio['starting_bankroll']:,.2f}")
    print(f"   Current bankroll: ${portfolio['current_bankroll']:,.2f}")
    print(f"   Total exposure: ${portfolio['total_exposure']:,.2f}")
    print(f"   Active trades: {portfolio['active_trades']}")
    
    if session['trades']:
        print(f"\n📈 Trades Placed:")
        for trade in session['trades']:
            print(f"   {trade['trade_id']}: {trade['home_team']} vs {trade['away_team']}")
            print(f"      Position: {trade['position']} @ {trade['odds']:.2f}")
            print(f"      Stake: ${trade['stake']:.2f}")
            print(f"      Signal probability: {trade['signal_probability']:.3f}")
            print(f"      Providers: {', '.join(trade['signal_providers_used'])}")
    
    print(f"\n🎯 System Ready For:")
    print(f"   ✅ Live blockchain market scanning")
    print(f"   ✅ Multi-signal probability analysis")
    print(f"   ✅ Kelly criterion position sizing")
    print(f"   ✅ Real-time portfolio management")
    print(f"   ✅ Multi-chain trading (Optimism + Arbitrum)")


if __name__ == "__main__":
    demo_blockchain_paper_trading()