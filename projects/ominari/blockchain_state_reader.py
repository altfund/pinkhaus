#!/usr/bin/env python3
"""
Read current blockchain state for live Overtime V2 markets
Focus on state queries rather than events
"""

import os
os.environ['PG_PORT'] = '5999'

from web3 import Web3
import requests
import json
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RPC endpoints - let's use public RPC first, then we can spin up nodes if needed
OPTIMISM_RPC = "https://mainnet.optimism.io"
ARBITRUM_RPC = "https://arb1.arbitrum.io/rpc"

class BlockchainStateReader:
    def __init__(self):
        self.optimism_w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        self.arbitrum_w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC))
        
        logger.info(f"Optimism connected: {self.optimism_w3.is_connected()}")
        logger.info(f"Arbitrum connected: {self.arbitrum_w3.is_connected()}")
        
    def find_real_overtime_addresses(self):
        """Find real Overtime contract addresses from blockchain explorers."""
        logger.info("🔍 Finding real Overtime V2 addresses...")
        
        # Let's check Optimism blockchain explorer for Overtime contracts
        # We can use Etherscan API or Optimistic Etherscan API
        
        # Known deployed addresses (from Overtime documentation/etherscan)
        # These might be different - let's search for them
        potential_addresses = [
            # Sports AMM V2 addresses
            "0xB4395993E4F9eC0DE60A6B9e7BE8E6BD1ef38b2D",  # Possible Sports AMM
            "0x5Ae7454827D83526261F3871C1029792644Ef1B1",  # Possible Manager
            "0x1f98415757620B543A52E61c46B32eB19261F984",  # Factory style contract
            
            # Try some standard patterns
            "0x1234567890123456789012345678901234567890",
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        ]
        
        real_contracts = []
        
        for address in potential_addresses:
            try:
                checksum_addr = Web3.to_checksum_address(address)
                code = self.optimism_w3.eth.get_code(checksum_addr)
                
                if code and len(code) > 2:  # Contract exists
                    logger.info(f"✅ Found contract at {checksum_addr}")
                    real_contracts.append(checksum_addr)
                    
                    # Get basic info
                    balance = self.optimism_w3.eth.get_balance(checksum_addr)
                    logger.info(f"   Balance: {Web3.from_wei(balance, 'ether')} ETH")
                    logger.info(f"   Code size: {len(code)} bytes")
                    
            except Exception as e:
                logger.debug(f"Error checking {address}: {e}")
                
        return real_contracts
        
    def read_market_manager_state(self, manager_address):
        """Read current state from a market manager contract."""
        logger.info(f"📊 Reading market manager state: {manager_address}")
        
        # Standard ERC20/Manager ABI functions
        basic_abi = [
            {"constant": True, "inputs": [], "name": "activeMarkets", "outputs": [{"name": "", "type": "address[]"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "numActiveMarkets", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [{"name": "index", "type": "uint256"}], "name": "activeMarketsPerIndex", "outputs": [{"name": "", "type": "address"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
        ]
        
        try:
            contract = self.optimism_w3.eth.contract(
                address=manager_address,
                abi=basic_abi
            )
            
            # Try different function calls to understand the contract
            functions_to_try = ['name', 'totalSupply', 'numActiveMarkets', 'activeMarkets']
            
            for func_name in functions_to_try:
                try:
                    if hasattr(contract.functions, func_name):
                        result = getattr(contract.functions, func_name)().call()
                        logger.info(f"  {func_name}(): {result}")
                        
                        if func_name == 'activeMarkets' and result:
                            logger.info(f"  Found {len(result)} active markets!")
                            return result  # Return list of market addresses
                            
                except Exception as e:
                    logger.debug(f"  {func_name}() failed: {e}")
                    
        except Exception as e:
            logger.error(f"Error reading manager state: {e}")
            
        return []
        
    def read_market_state(self, market_address):
        """Read current state from a market contract."""
        logger.info(f"📈 Reading market state: {market_address}")
        
        # Standard market contract ABI
        market_abi = [
            {"constant": True, "inputs": [], "name": "homeTeam", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "awayTeam", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "maturityDate", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "gameId", "outputs": [{"name": "", "type": "bytes32"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "isResolved", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "homeOdds", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "awayOdds", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "drawOdds", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "totalDeposited", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
        ]
        
        try:
            contract = self.optimism_w3.eth.contract(
                address=market_address,
                abi=market_abi
            )
            
            market_data = {}
            
            # Try to read all market data
            functions = ['homeTeam', 'awayTeam', 'maturityDate', 'gameId', 'isResolved', 
                        'homeOdds', 'awayOdds', 'drawOdds', 'totalDeposited']
            
            for func_name in functions:
                try:
                    if hasattr(contract.functions, func_name):
                        result = getattr(contract.functions, func_name)().call()
                        market_data[func_name] = result
                        
                        if func_name == 'maturityDate' and result:
                            # Convert timestamp to readable date
                            maturity = datetime.fromtimestamp(result, tz=timezone.utc)
                            logger.info(f"  Maturity: {maturity}")
                            
                except Exception as e:
                    logger.debug(f"  {func_name}() failed: {e}")
                    
            logger.info(f"  Market data: {market_data}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error reading market state: {e}")
            return {}
            
    def explore_overtime_via_thegraph(self):
        """Use The Graph to find live Overtime data."""
        logger.info("🔍 Exploring Overtime via The Graph...")
        
        # The Graph has multiple Overtime subgraphs
        subgraph_endpoints = [
            "https://api.thegraph.com/subgraphs/name/thales-markets/thales-optimism",
            "https://api.thegraph.com/subgraphs/name/overtimemarket/overtime-optimism", 
            "https://thegraph.com/hosted-service/subgraph/overtimemarket/overtime-optimism",
        ]
        
        # Query for active markets
        query = """
        {
          markets(first: 20, where: {isResolved: false, maturityDate_gt: "1726425600"}) {
            id
            gameId
            homeTeam
            awayTeam
            maturityDate
            homeOdds
            awayOdds
            drawOdds
            isResolved
            totalVolume
            creator
            marketAddress
          }
        }
        """
        
        for endpoint in subgraph_endpoints:
            try:
                logger.info(f"Trying: {endpoint}")
                
                response = requests.post(
                    endpoint,
                    json={'query': query},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Response: {json.dumps(data, indent=2)[:500]}...")
                    
                    if 'data' in data and 'markets' in data['data']:
                        markets = data['data']['markets']
                        if markets:
                            logger.info(f"✅ Found {len(markets)} markets from The Graph!")
                            return markets
                            
                else:
                    logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                logger.warning(f"Subgraph {endpoint} failed: {e}")
                
        return []
        
    def discover_markets_via_direct_state_reading(self):
        """Try to discover markets by reading state directly."""
        logger.info("🎯 Discovering markets via direct state reading...")
        
        # Try reading state from well-known patterns
        # Overtime might use factory patterns or registry patterns
        
        # Let's try some common factory/registry addresses
        factory_patterns = [
            "0x" + "1" * 40,  # Simple pattern
            "0x" + "a" * 40,  # Another pattern
        ]
        
        for address in factory_patterns:
            try:
                checksum_addr = Web3.to_checksum_address(address)
                markets = self.read_market_manager_state(checksum_addr)
                if markets:
                    logger.info(f"Found markets from {checksum_addr}: {markets}")
                    return markets
                    
            except Exception as e:
                logger.debug(f"Factory pattern {address} failed: {e}")
                
        return []
        
    def sync_live_markets(self, markets_data):
        """Sync live markets to database."""
        if not markets_data:
            logger.warning("No live markets to sync")
            return 0
            
        logger.info(f"💾 Syncing {len(markets_data)} live markets...")
        
        with db_manager.get_db_session() as db:
            added = 0
            
            for market in markets_data:
                try:
                    # Extract market data
                    market_id = f"live_{market.get('id', market.get('marketAddress', 'unknown'))}"
                    
                    # Check if exists
                    existing = db.query(Market).filter(Market.source_id == market_id).first()
                    if existing:
                        continue
                        
                    # Parse maturity date
                    maturity_timestamp = int(market.get('maturityDate', 0))
                    maturity_date = datetime.fromtimestamp(maturity_timestamp, tz=timezone.utc)
                    
                    # Skip past markets
                    if maturity_date < datetime.now(timezone.utc):
                        continue
                        
                    # Create market
                    new_market = Market(
                        source_id=market_id,
                        source='blockchain_live',
                        sport='Soccer',  # Most are soccer
                        league_name='Live Blockchain',
                        market_type='winner',
                        home_team=market.get('homeTeam', 'Home')[:50],
                        away_team=market.get('awayTeam', 'Away')[:50],
                        maturity_date=maturity_date,
                        is_finished=market.get('isResolved', False),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(new_market)
                    
                    # Add live odds
                    home_odds = float(market.get('homeOdds', 0))
                    away_odds = float(market.get('awayOdds', 0)) 
                    draw_odds = float(market.get('drawOdds', 0))
                    
                    # Convert from wei if needed
                    if home_odds > 1000:  # Likely in wei
                        home_odds = home_odds / 1e18
                        away_odds = away_odds / 1e18  
                        draw_odds = draw_odds / 1e18
                        
                    if home_odds > 0:
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
                                    normalized_implied=1.0 / decimal_odds if decimal_odds > 0 else 0,
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
    reader = BlockchainStateReader()
    
    # Try multiple approaches to find live markets
    logger.info("🚀 Starting blockchain state reading for live markets...")
    
    # 1. Find real contract addresses
    contracts = reader.find_real_overtime_addresses()
    
    # 2. Try The Graph subgraph
    markets = reader.explore_overtime_via_thegraph()
    
    # 3. Try direct state reading
    if not markets:
        markets = reader.discover_markets_via_direct_state_reading()
    
    # 4. Sync any found markets
    if markets:
        reader.sync_live_markets(markets)
    else:
        logger.warning("❌ No live markets found - may need to spin up blockchain nodes or find correct addresses")
        
    logger.info("🎯 Blockchain state reading complete!")

if __name__ == "__main__":
    main()