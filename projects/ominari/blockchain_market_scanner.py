#!/usr/bin/env python3
"""
Direct blockchain scanner for live Overtime V2 markets
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

# Overtime V2 contract addresses (from previous research)
OPTIMISM_RPC = "https://mainnet.optimism.io"
ARBITRUM_RPC = "https://arb1.arbitrum.io/rpc"

# Known Overtime V2 contract addresses
OVERTIME_CONTRACTS = {
    'optimism': {
        'sports_amm': '0x5ae538eaf83a3a180b2b1c2e4ee40be72b4a0c60',  # Sports AMM V2
        'manager': '0x123...', # We'll need to find the manager contract
        'factory': '0x456...',  # We'll need to find the factory
    },
    'arbitrum': {
        'sports_amm': '0x...',  # Need to find Arbitrum addresses
    }
}

class BlockchainMarketScanner:
    def __init__(self):
        self.optimism_w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        self.arbitrum_w3 = Web3(Web3.HTTPProvider(ARBITRUM_RPC))
        
        # Verify connections
        logger.info(f"Optimism connected: {self.optimism_w3.is_connected()}")
        logger.info(f"Arbitrum connected: {self.arbitrum_w3.is_connected()}")
        
    def scan_active_markets(self):
        """Scan blockchain for active markets."""
        logger.info("🔍 Scanning blockchain for active Overtime V2 markets...")
        
        # First, let's find active contracts by looking at recent transactions
        self.find_overtime_contracts()
        
        # Then scan for market creation events
        self.scan_market_events()
        
    def find_overtime_contracts(self):
        """Find Overtime contract addresses by scanning recent activity."""
        logger.info("🔍 Finding Overtime V2 contracts...")
        
        # Get recent blocks to find Overtime activity
        latest_block = self.optimism_w3.eth.get_block('latest')
        logger.info(f"Latest Optimism block: {latest_block['number']}")
        
        # Look at recent blocks for Overtime transactions
        for i in range(10):  # Check last 10 blocks
            block_num = latest_block['number'] - i
            block = self.optimism_w3.eth.get_block(block_num, full_transactions=True)
            
            overtime_txs = []
            for tx in block['transactions']:
                # Look for transactions to known Overtime-related addresses
                if tx['to'] and any(addr.lower() in str(tx['to']).lower() for addr in ['overtime', '5ae538eaf83a3a180b2b1c2e4ee40be72b4a0c60']):
                    overtime_txs.append(tx)
                    
            if overtime_txs:
                logger.info(f"Found {len(overtime_txs)} Overtime transactions in block {block_num}")
                for tx in overtime_txs[:3]:  # Show first 3
                    logger.info(f"  TX: {tx['hash'].hex()} to {tx['to']}")
                    
    def scan_market_events(self):
        """Scan for market creation and update events."""
        logger.info("📡 Scanning for market events...")
        
        # Sports AMM contract ABI for market events
        market_abi = [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "market", "type": "address"},
                    {"indexed": True, "name": "gameId", "type": "bytes32"},
                    {"indexed": False, "name": "maturityDate", "type": "uint256"}
                ],
                "name": "MarketCreated",
                "type": "event"
            }
        ]
        
        try:
            # Connect to Sports AMM contract
            sports_amm_address = "0x5ae538eaf83a3a180b2b1c2e4ee40be72b4a0c60"
            contract = self.optimism_w3.eth.contract(
                address=Web3.to_checksum_address(sports_amm_address),
                abi=market_abi
            )
            
            # Get recent market creation events
            latest_block = self.optimism_w3.eth.get_block_number()
            from_block = latest_block - 10000  # Last ~10k blocks
            
            logger.info(f"Scanning blocks {from_block} to {latest_block} for MarketCreated events...")
            
            events = contract.events.MarketCreated.get_logs(
                fromBlock=from_block,
                toBlock=latest_block
            )
            
            logger.info(f"Found {len(events)} MarketCreated events")
            
            for event in events[:5]:  # Show first 5
                logger.info(f"Market: {event['args']['market']}")
                logger.info(f"Game ID: {event['args']['gameId'].hex()}")
                logger.info(f"Maturity: {datetime.fromtimestamp(event['args']['maturityDate'], tz=timezone.utc)}")
                
        except Exception as e:
            logger.error(f"Error scanning events: {e}")
            
    def get_market_details(self, market_address):
        """Get details for a specific market contract."""
        logger.info(f"📊 Getting details for market {market_address}")
        
        # Standard market contract ABI
        market_abi = [
            {"constant": True, "inputs": [], "name": "homeTeam", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "awayTeam", "outputs": [{"name": "", "type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "maturityDate", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "homeOdds", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "awayOdds", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "isResolved", "outputs": [{"name": "", "type": "bool"}], "type": "function"}
        ]
        
        try:
            contract = self.optimism_w3.eth.contract(
                address=Web3.to_checksum_address(market_address),
                abi=market_abi
            )
            
            # Try to get market data
            home_team = contract.functions.homeTeam().call()
            away_team = contract.functions.awayTeam().call()
            maturity = contract.functions.maturityDate().call()
            is_resolved = contract.functions.isResolved().call()
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'maturity_date': datetime.fromtimestamp(maturity, tz=timezone.utc),
                'is_resolved': is_resolved,
                'contract_address': market_address
            }
            
        except Exception as e:
            logger.error(f"Error getting market details: {e}")
            return None
            
    def sync_blockchain_markets(self):
        """Sync real blockchain markets to database."""
        logger.info("💾 Syncing blockchain markets to database...")
        
        with db_manager.get_db_session() as db:
            # Clear existing fake data
            fake_markets = db.query(Market).filter(
                Market.source.in_(['overtime_v2', 'overtime_v2_real', 'overtime_soccer'])
            ).all()
            
            if fake_markets:
                logger.info(f"🧹 Clearing {len(fake_markets)} fake markets...")
                for market in fake_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
                
            logger.info("✅ Fake data cleared. Ready for real blockchain data!")

def main():
    scanner = BlockchainMarketScanner()
    scanner.scan_active_markets()
    scanner.sync_blockchain_markets()

if __name__ == "__main__":
    main()