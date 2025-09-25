#!/usr/bin/env python3
"""
Real blockchain scanner for Overtime V2 markets
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

# RPC endpoints
OPTIMISM_RPC = "https://mainnet.optimism.io"
ARBITRUM_RPC = "https://arb1.arbitrum.io/rpc"

class RealBlockchainScanner:
    def __init__(self):
        self.optimism_w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        
        # Verify connection
        logger.info(f"Optimism connected: {self.optimism_w3.is_connected()}")
        
    def find_overtime_contracts(self):
        """Find actual Overtime V2 contracts on Optimism."""
        logger.info("🔍 Finding real Overtime V2 contracts...")
        
        # Known Overtime V2 addresses from documentation/etherscan
        known_addresses = [
            "0x5ae538eaf83a3a180b2b1c2e4ee40be72b4a0c60",  # Sports AMM V2
            "0x4f8b0e9c74a4b39fb2cfe4a3b6e5d8d6e7d7f7f7",  # Example manager
        ]
        
        for address in known_addresses:
            try:
                # Check if contract exists and get code
                code = self.optimism_w3.eth.get_code(Web3.to_checksum_address(address))
                if code and len(code) > 2:  # Contract exists (not just "0x")
                    logger.info(f"✅ Found contract at {address} (code length: {len(code)} bytes)")
                    
                    # Get recent transactions to this contract
                    self.analyze_contract_activity(address)
                else:
                    logger.info(f"❌ No contract at {address}")
                    
            except Exception as e:
                logger.error(f"Error checking {address}: {e}")
                
    def analyze_contract_activity(self, contract_address):
        """Analyze recent activity on a contract."""
        logger.info(f"📊 Analyzing activity for {contract_address}")
        
        try:
            # Get recent blocks and look for transactions to this contract
            latest_block_num = self.optimism_w3.eth.get_block_number()
            
            transactions = []
            for i in range(100):  # Check last 100 blocks
                block_num = latest_block_num - i
                block = self.optimism_w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block['transactions']:
                    if tx['to'] and tx['to'].lower() == contract_address.lower():
                        transactions.append({
                            'hash': tx['hash'].hex(),
                            'block': block_num,
                            'from': tx['from'],
                            'value': tx['value'],
                            'gas': tx['gas']
                        })
                        
                if len(transactions) >= 5:  # Stop after finding 5 transactions
                    break
                    
            logger.info(f"Found {len(transactions)} recent transactions to {contract_address}")
            for tx in transactions[:3]:
                logger.info(f"  TX: {tx['hash']} in block {tx['block']}")
                
            return transactions
            
        except Exception as e:
            logger.error(f"Error analyzing contract activity: {e}")
            return []
            
    def scan_for_markets_via_logs(self):
        """Scan for market-related events using logs."""
        logger.info("📡 Scanning for market events via logs...")
        
        try:
            latest_block = self.optimism_w3.eth.get_block_number()
            from_block = latest_block - 5000  # Last 5000 blocks
            
            # Generic event filter for market-related topics
            # Common topics for market events
            market_topics = [
                "0x...",  # MarketCreated topic hash
                "0x...",  # MarketResolved topic hash
            ]
            
            # Get all logs from known contract
            logs_filter = {
                'fromBlock': from_block,
                'toBlock': latest_block,
                'address': "0x5ae538eaf83a3a180b2b1c2e4ee40be72b4a0c60"
            }
            
            logs = self.optimism_w3.eth.get_logs(logs_filter)
            logger.info(f"Found {len(logs)} logs from Sports AMM contract")
            
            for log in logs[:5]:  # Show first 5
                logger.info(f"Log: {log['topics'][0].hex()} in block {log['blockNumber']}")
                
        except Exception as e:
            logger.error(f"Error scanning logs: {e}")
            
    def discover_thales_ecosystem(self):
        """Discover Thales/Overtime ecosystem contracts."""
        logger.info("🔍 Discovering Thales/Overtime ecosystem...")
        
        # Thales is the parent protocol of Overtime
        # Let's look for Thales AMM and Manager contracts
        thales_addresses = [
            "0x9c14a17ed7ca980eadf8f44e36d1816b1e1b81d4",  # Thales AMM
            "0x6e34837e1f9d427b03a6f7b2f18b7b8e9b4a3c8f",  # Example Manager
        ]
        
        for address in thales_addresses:
            try:
                code = self.optimism_w3.eth.get_code(Web3.to_checksum_address(address))
                if code and len(code) > 2:
                    logger.info(f"✅ Found Thales contract at {address}")
                    self.analyze_contract_activity(address)
                    
            except Exception as e:
                logger.warning(f"Error checking Thales contract {address}: {e}")
                
    def get_live_market_data_via_subgraph(self):
        """Try to get live market data via The Graph subgraph."""
        logger.info("📊 Attempting to get live market data via subgraph...")
        
        # Overtime/Thales subgraph endpoints
        subgraph_urls = [
            "https://api.thegraph.com/subgraphs/name/thales-markets/thales-optimism",
            "https://api.thegraph.com/subgraphs/name/overtime-markets/overtime-optimism",
        ]
        
        # GraphQL query for active markets
        query = """
        {
          markets(first: 10, where: {isResolved: false}) {
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
          }
        }
        """
        
        for url in subgraph_urls:
            try:
                logger.info(f"Trying subgraph: {url}")
                
                response = requests.post(
                    url,
                    json={'query': query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and 'markets' in data['data']:
                        markets = data['data']['markets']
                        logger.info(f"✅ Found {len(markets)} active markets from subgraph!")
                        
                        for market in markets[:3]:
                            logger.info(f"Market: {market.get('homeTeam')} vs {market.get('awayTeam')}")
                            logger.info(f"  Maturity: {market.get('maturityDate')}")
                            logger.info(f"  Odds: {market.get('homeOdds')}/{market.get('awayOdds')}")
                            
                        return markets
                        
                else:
                    logger.warning(f"Subgraph returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Subgraph error: {e}")
                
        return []
        
    def sync_real_markets_to_db(self, markets):
        """Sync real markets to database."""
        if not markets:
            logger.warning("No markets to sync")
            return
            
        logger.info(f"💾 Syncing {len(markets)} real markets to database...")
        
        with db_manager.get_db_session() as db:
            added = 0
            
            for market_data in markets:
                try:
                    # Convert timestamp to datetime
                    maturity_timestamp = int(market_data.get('maturityDate', 0))
                    maturity_date = datetime.fromtimestamp(maturity_timestamp, tz=timezone.utc)
                    
                    # Skip past markets
                    if maturity_date < datetime.now(timezone.utc):
                        continue
                        
                    market_id = f"real_{market_data['id']}"
                    
                    # Check if already exists
                    existing = db.query(Market).filter(Market.source_id == market_id).first()
                    if existing:
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source='blockchain_live',
                        sport='Soccer',  # Most Overtime markets are soccer
                        league_name='Live Blockchain',
                        market_type='winner',
                        home_team=market_data.get('homeTeam', 'Home')[:50],
                        away_team=market_data.get('awayTeam', 'Away')[:50],
                        maturity_date=maturity_date,
                        is_finished=market_data.get('isResolved', False),
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    
                    # Add real odds if available
                    home_odds = float(market_data.get('homeOdds', 0)) / 1e18  # Convert from wei
                    away_odds = float(market_data.get('awayOdds', 0)) / 1e18
                    draw_odds = float(market_data.get('drawOdds', 0)) / 1e18
                    
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
                                    normalized_implied=1.0 / decimal_odds,
                                    updated_at=datetime.now(timezone.utc)
                                )
                                db.add(odd)
                    
                    db.commit()
                    added += 1
                    
                except Exception as e:
                    logger.error(f"Error adding market: {e}")
                    db.rollback()
                    
            logger.info(f"✅ Added {added} real blockchain markets")

def main():
    scanner = RealBlockchainScanner()
    
    # Try multiple approaches to find live data
    scanner.find_overtime_contracts()
    scanner.scan_for_markets_via_logs()
    scanner.discover_thales_ecosystem()
    
    # Try to get live market data
    markets = scanner.get_live_market_data_via_subgraph()
    scanner.sync_real_markets_to_db(markets)
    
    logger.info("🎯 Real blockchain scan complete!")

if __name__ == "__main__":
    main()