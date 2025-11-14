#!/usr/bin/env python3
"""
Real odds fetcher that gets actual odds from blockchain contracts
Replaces placeholder odds with real market data
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from web3 import Web3
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealOddsFetcher:
    """Fetches real odds from blockchain and external APIs"""
    
    def __init__(self):
        # Use public Overtime API (no key required)
        self.overtime_api_base = "https://overtimemarketsv2.xyz/overtime-v2"
        
        # Initialize Web3 connections for direct blockchain access
        self.w3_connections = {
            'arbitrum': Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc')),
            'optimism': Web3(Web3.HTTPProvider('https://mainnet.optimism.io')),
        }
        
        # Overtime V2 contract addresses
        self.amm_contracts = {
            'arbitrum': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
            'optimism': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        }
        
    def fetch_live_markets(self) -> List[Dict]:
        """Fetch live markets from Overtime API"""
        try:
            # Get live markets for major leagues
            leagues = ['EPL', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 'UEFA Champions League']
            all_markets = []
            
            for league in leagues:
                url = f"{self.overtime_api_base}/networks/42161/markets/live"
                params = {
                    'sport': 'FOOTBALL',
                    'league': league,
                    'type': '0,1'  # Moneyline and totals
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    markets = data.get('markets', [])
                    logger.info(f"Found {len(markets)} live markets for {league}")
                    all_markets.extend(markets)
                else:
                    logger.warning(f"Failed to fetch markets for {league}: {response.status_code}")
                    
            return all_markets
            
        except Exception as e:
            logger.error(f"Error fetching live markets: {e}")
            return []
            
    def fetch_market_odds(self, market_address: str, network: str = 'arbitrum') -> Dict:
        """Fetch real odds for a specific market from blockchain"""
        try:
            w3 = self.w3_connections.get(network)
            if not w3:
                return {}
                
            # Simple ABI for getMarketDefaultOdds function
            abi = [{
                "inputs": [{"name": "_market", "type": "address"}],
                "name": "getMarketDefaultOdds",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }]
            
            amm_address = self.amm_contracts.get(network)
            contract = w3.eth.contract(address=Web3.to_checksum_address(amm_address), abi=abi)
            
            # Get odds from contract
            odds_raw = contract.functions.getMarketDefaultOdds(
                Web3.to_checksum_address(market_address)
            ).call()
            
            # Convert from contract format (18 decimals) to decimal odds
            odds = {}
            outcomes = ['home', 'away', 'draw'] if len(odds_raw) > 2 else ['home', 'away']
            
            for i, outcome in enumerate(outcomes):
                if i < len(odds_raw) and odds_raw[i] > 0:
                    # Convert from 1e18 format to decimal odds
                    decimal_odds = 1e18 / odds_raw[i]
                    odds[outcome] = round(decimal_odds, 3)
                    
            return odds
            
        except Exception as e:
            logger.error(f"Error fetching odds for market {market_address}: {e}")
            return {}
            
    def update_database_with_real_odds(self):
        """Update database with real odds from blockchain"""
        logger.info("Starting real odds update...")
        
        # Fetch live markets from API
        live_markets = self.fetch_live_markets()
        
        if not live_markets:
            logger.warning("No live markets found from API")
            return
            
        with db_manager.get_db_session() as db:
            updated_count = 0
            
            for api_market in live_markets[:20]:  # Process first 20 markets
                try:
                    # Extract market details
                    home_team = api_market.get('homeTeam', '')
                    away_team = api_market.get('awayTeam', '')
                    market_address = api_market.get('gameId', '')  # This might be the market address
                    maturity = api_market.get('maturity', 0)
                    
                    if not all([home_team, away_team, market_address]):
                        continue
                        
                    # Check if market exists in database
                    db_market = db.query(Market).filter(
                        Market.home_team == home_team,
                        Market.away_team == away_team
                    ).first()
                    
                    if not db_market:
                        # Create new market
                        db_market = Market(
                            source_id=f"v2_{market_address}",
                            sport='Soccer',
                            league=api_market.get('leagueLabel', 'Unknown'),
                            home_team=home_team,
                            away_team=away_team,
                            start_time=datetime.fromtimestamp(maturity, tz=timezone.utc) if maturity else datetime.now(timezone.utc),
                            is_active=True,
                            source='overtime_v2'
                        )
                        db.add(db_market)
                        db.commit()
                        
                    # Get odds from API or blockchain
                    odds_data = {}
                    
                    # Try API first (includes calculated odds)
                    if 'odds' in api_market:
                        api_odds = api_market['odds']
                        if isinstance(api_odds, list) and len(api_odds) >= 2:
                            odds_data = {
                                'home': api_odds[0],
                                'away': api_odds[1],
                                'draw': api_odds[2] if len(api_odds) > 2 else None
                            }
                    
                    # If no API odds, try blockchain
                    if not odds_data and market_address.startswith('0x'):
                        network = api_market.get('network', 'arbitrum')
                        odds_data = self.fetch_market_odds(market_address, network)
                        
                    # Update odds in database
                    if odds_data:
                        for outcome, decimal_odds in odds_data.items():
                            if decimal_odds and decimal_odds > 1:  # Valid odds
                                # Check if odd exists
                                existing_odd = db.query(Odd).filter(
                                    Odd.source_market_id == db_market.source_id,
                                    Odd.outcome == outcome
                                ).order_by(Odd.created_at.desc()).first()
                                
                                # Only create new odd if value changed or doesn't exist
                                if not existing_odd or abs(existing_odd.decimal_odds - decimal_odds) > 0.01:
                                    new_odd = Odd(
                                        source_market_id=db_market.source_id,
                                        outcome=outcome,
                                        decimal_odds=decimal_odds,
                                        source='overtime_v2',
                                        created_at=datetime.now(timezone.utc)
                                    )
                                    db.add(new_odd)
                                    updated_count += 1
                                    
                    db.commit()
                    logger.info(f"Updated market: {home_team} vs {away_team} with odds: {odds_data}")
                    
                except Exception as e:
                    logger.error(f"Error processing market: {e}")
                    db.rollback()
                    continue
                    
        logger.info(f"Real odds update complete. Updated {updated_count} odds records.")
        
    def run_continuous_update(self, interval_seconds: int = 60):
        """Run continuous odds updates"""
        import asyncio
        
        async def update_loop():
            while True:
                try:
                    self.update_database_with_real_odds()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in update loop: {e}")
                    await asyncio.sleep(30)  # Shorter retry on error
                    
        asyncio.run(update_loop())


def main():
    """Test the real odds fetcher"""
    fetcher = RealOddsFetcher()
    
    # Do one update
    fetcher.update_database_with_real_odds()
    
    # Check what we got
    with db_manager.get_db_session() as db:
        recent_odds = db.query(Odd).order_by(Odd.created_at.desc()).limit(10).all()
        
        print("\nRecent odds in database:")
        for odd in recent_odds:
            print(f"  {odd.outcome}: {odd.decimal_odds:.3f} (created: {odd.created_at})")
            
        # Count markets with real odds
        markets_with_odds = db.query(Market).join(
            Odd, Market.source_id == Odd.source_market_id
        ).distinct().count()
        
        print(f"\nMarkets with odds: {markets_with_odds}")


if __name__ == "__main__":
    main()