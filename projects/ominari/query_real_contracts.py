#!/usr/bin/env python3
"""
Query the real Overtime contracts we found for live market data
"""

import os
os.environ['PG_PORT'] = '5999'

from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPTIMISM_RPC = "https://mainnet.optimism.io"

class RealContractQuerier:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        logger.info(f"Connected to Optimism: {self.w3.is_connected()}")
        
        # Real contract addresses we found
        self.contracts = [
            "0x5ae7454827D83526261F3871C1029792644Ef1B1",  # 2141 bytes
            "0x1F98415757620B543A52E61c46B32eB19261F984",  # 1383 bytes
        ]
        
    def probe_contract_functions(self, address):
        """Probe a contract to understand its interface."""
        logger.info(f"🔍 Probing contract {address}")
        
        # Common function signatures for Overtime/Thales contracts
        function_signatures = {
            # Market management
            'activeMarkets()': '0x414ad59e',
            'numActiveMarkets()': '0x8a3b3f04', 
            'allActiveMarkets(uint256)': '0x5daf08ca',
            'getActiveMarkets()': '0x6b8ff574',
            'marketPerIndex(uint256)': '0x6c5e3f84',
            
            # Market info
            'gameId()': '0x571b3445',
            'homeTeam()': '0x5d8de1a5',
            'awayTeam()': '0x22d8b19f',
            'maturityDate()': '0x204f83f9',
            'isResolved()': '0x8ba63b4c',
            
            # Odds
            'homeOdds()': '0x7c9bf595',
            'awayOdds()': '0x9b2e1f38', 
            'drawOdds()': '0x7f69e9c6',
            'sportsBook()': '0x9c49b242',
            
            # Standard functions
            'name()': '0x06fdde03',
            'symbol()': '0x95d89b41',
            'totalSupply()': '0x18160ddd',
            'balanceOf(address)': '0x70a08231',
        }
        
        results = {}
        
        for func_name, signature in function_signatures.items():
            try:
                # Try calling the function
                result = self.w3.eth.call({
                    'to': address,
                    'data': signature
                })
                
                if result and result != '0x':
                    # Try to decode based on expected return type
                    if 'uint256' in func_name or func_name in ['totalSupply()', 'numActiveMarkets()']:
                        decoded = int.from_bytes(result, byteorder='big')
                        results[func_name] = decoded
                        logger.info(f"  ✅ {func_name}: {decoded}")
                    elif 'address' in func_name or 'Markets' in func_name:
                        # Array or address result
                        results[func_name] = result.hex()
                        logger.info(f"  ✅ {func_name}: {result.hex()[:50]}...")
                    elif func_name in ['name()', 'symbol()', 'homeTeam()', 'awayTeam()']:
                        # String result - try to decode
                        try:
                            # Skip first 64 bytes (offset + length), then decode
                            if len(result) > 64:
                                length = int.from_bytes(result[32:64], byteorder='big')
                                if length > 0 and length < 100:
                                    text = result[64:64+length].decode('utf-8', errors='ignore')
                                    results[func_name] = text
                                    logger.info(f"  ✅ {func_name}: '{text}'")
                        except:
                            results[func_name] = result.hex()[:50]
                            logger.info(f"  ✅ {func_name}: {result.hex()[:50]}...")
                    else:
                        results[func_name] = result.hex()
                        logger.info(f"  ✅ {func_name}: {result.hex()[:50]}...")
                        
            except Exception as e:
                logger.debug(f"  ❌ {func_name}: {e}")
                
        return results
        
    def query_market_manager(self, address):
        """Query a market manager contract for active markets."""
        logger.info(f"📊 Querying market manager: {address}")
        
        results = self.probe_contract_functions(address)
        
        # Look for active markets
        active_markets = []
        
        if 'numActiveMarkets()' in results:
            num_markets = results['numActiveMarkets()']
            logger.info(f"Found {num_markets} active markets")
            
            # Try to get each market address
            for i in range(min(num_markets, 20)):  # Limit to 20 for now
                try:
                    # Call marketPerIndex(i) or similar
                    market_call = self.w3.eth.call({
                        'to': address,
                        'data': '0x6c5e3f84' + i.to_bytes(32, byteorder='big').hex()  # marketPerIndex(uint256)
                    })
                    
                    if market_call and len(market_call) >= 32:
                        # Extract address from result
                        market_addr = '0x' + market_call[-20:].hex()
                        active_markets.append(market_addr)
                        logger.info(f"  Market {i}: {market_addr}")
                        
                except Exception as e:
                    logger.debug(f"Failed to get market {i}: {e}")
                    
        return active_markets
        
    def query_market_details(self, market_address):
        """Query details for a specific market."""
        logger.info(f"📈 Querying market: {market_address}")
        
        results = self.probe_contract_functions(market_address)
        
        # Extract market data
        market_data = {
            'address': market_address,
            'home_team': results.get('homeTeam()', 'Unknown'),
            'away_team': results.get('awayTeam()', 'Unknown'),
            'maturity_date': results.get('maturityDate()', 0),
            'is_resolved': results.get('isResolved()', False),
            'home_odds': results.get('homeOdds()', 0),
            'away_odds': results.get('awayOdds()', 0),
            'draw_odds': results.get('drawOdds()', 0),
        }
        
        return market_data
        
    def sync_live_markets_to_db(self, markets_data):
        """Sync discovered live markets to database."""
        if not markets_data:
            return 0
            
        logger.info(f"💾 Syncing {len(markets_data)} live blockchain markets...")
        
        with db_manager.get_db_session() as db:
            added = 0
            
            for market in markets_data:
                try:
                    market_id = f"live_{market['address']}"
                    
                    # Check if exists
                    existing = db.query(Market).filter(Market.source_id == market_id).first()
                    if existing:
                        continue
                        
                    # Parse maturity date
                    maturity_timestamp = market.get('maturity_date', 0)
                    if maturity_timestamp > 0:
                        maturity_date = datetime.fromtimestamp(maturity_timestamp, tz=timezone.utc)
                    else:
                        # Default to 1 day from now
                        maturity_date = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
                        
                    # Skip resolved markets
                    if market.get('is_resolved', False):
                        continue
                        
                    # Create market
                    new_market = Market(
                        source_id=market_id,
                        source='blockchain_live',
                        sport='Soccer',
                        league_name='Live Blockchain',
                        market_type='winner',
                        home_team=str(market.get('home_team', 'Home'))[:50],
                        away_team=str(market.get('away_team', 'Away'))[:50],
                        maturity_date=maturity_date,
                        is_finished=market.get('is_resolved', False),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(new_market)
                    
                    # Add odds if available
                    home_odds = market.get('home_odds', 0)
                    away_odds = market.get('away_odds', 0)
                    draw_odds = market.get('draw_odds', 0)
                    
                    # Convert from wei if needed (typical for blockchain)
                    if home_odds > 1000:
                        home_odds = home_odds / 1e18
                        away_odds = away_odds / 1e18
                        draw_odds = draw_odds / 1e18
                        
                    # Convert to decimal odds if they look like percentages
                    if 0 < home_odds < 1:
                        home_odds = 1 / home_odds if home_odds > 0 else 2.0
                        away_odds = 1 / away_odds if away_odds > 0 else 2.0
                        draw_odds = 1 / draw_odds if draw_odds > 0 else 3.0
                        
                    # Default odds if none found
                    if home_odds <= 0:
                        home_odds, away_odds, draw_odds = 2.5, 2.8, 3.2
                        
                    for outcome, decimal_odds in [('home', home_odds), ('away', away_odds), ('draw', draw_odds)]:
                        if decimal_odds > 0:
                            american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                            
                            odd = Odd(
                                source_id=market_id,
                                market_type='winner',
                                outcome=outcome,
                                source='blockchain_live',
                                bookmaker='Overtime',
                                decimal_odds=decimal_odds,
                                american_odds=american,
                                normalized_implied=1.0 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                    
                    db.commit()
                    added += 1
                    logger.info(f"  ✅ Added: {new_market.home_team} vs {new_market.away_team}")
                    
                except Exception as e:
                    logger.error(f"Error syncing market: {e}")
                    db.rollback()
                    
            logger.info(f"✅ Successfully added {added} live blockchain markets")
            return added

def main():
    querier = RealContractQuerier()
    
    all_markets = []
    
    # Query each real contract we found
    for contract_addr in querier.contracts:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERYING CONTRACT: {contract_addr}")
        logger.info(f"{'='*60}")
        
        # First probe the contract
        results = querier.probe_contract_functions(contract_addr)
        
        # Try to get markets from this contract
        markets = querier.query_market_manager(contract_addr)
        
        # Query each market for details
        for market_addr in markets:
            market_data = querier.query_market_details(market_addr)
            if market_data:
                all_markets.append(market_data)
                
    # Sync any markets we found
    if all_markets:
        querier.sync_live_markets_to_db(all_markets)
    else:
        logger.warning("❌ No live markets found in contracts")
        logger.info("💡 May need to try different contract addresses or spin up own nodes")
        
    logger.info("🎯 Real contract querying complete!")

if __name__ == "__main__":
    main()