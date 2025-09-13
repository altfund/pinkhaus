#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blockchain Reader for Direct Chain Access
Connects to Optimism/Arbitrum to read Overtime Markets data directly.
"""

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging
import asyncio
import sqlite3
import os

logger = logging.getLogger(__name__)


class BlockchainConfig:
    """Chain configurations for different networks."""
    
    NETWORKS = {
        'optimism': {
            'chain_id': 10,
            'rpc_url': 'https://mainnet.optimism.io',
            'explorer': 'https://optimistic.etherscan.io',
            'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',  # Correct SportsAMMV2 address
            'block_time': 2,
        },
        'arbitrum': {
            'chain_id': 42161,
            'rpc_url': 'https://arb1.arbitrum.io/rpc',
            'explorer': 'https://arbiscan.io',
            'sports_amm_v2': '0x7465c5d60d3d095443CF9991Da03304A30D42Eae',
            'block_time': 0.25,
        },
        'optimism_sepolia': {
            'chain_id': 11155420,
            'rpc_url': 'https://sepolia.optimism.io',
            'explorer': 'https://sepolia-optimism.etherscan.io',
            'sports_amm_v2': '0x5e2c7704F5f784B2e4A5C1bD5b7fBE578A2b6EdC',
            'block_time': 2,
        }
    }
    
    # Load SportsAMMV2 ABI
    @staticmethod
    def load_abi():
        """Load ABI from file or use default."""
        try:
            with open('contract_abis.json', 'r') as f:
                abis = json.load(f)
            return abis['optimism']['SportsAMMV2']
        except Exception:
            # Fallback to minimal ABI
            return [
                {
                    "name": "MarketCreated",
                    "type": "event",
                    "anonymous": False,
                    "inputs": [
                        {"name": "market", "type": "address", "indexed": True},
                        {"name": "gameId", "type": "bytes32", "indexed": True},
                        {"name": "gameLabel", "type": "string", "indexed": False},
                        {"name": "maturityDate", "type": "uint256", "indexed": False},
                        {"name": "tags", "type": "uint256[]", "indexed": False},
                        {"name": "normalizedOdds", "type": "uint256[]", "indexed": False}
                    ]
                },
                {
                    "name": "BoughtFromAmm",
                    "type": "event",
                    "anonymous": False,
                    "inputs": [
                        {"name": "buyer", "type": "address", "indexed": True},
                        {"name": "market", "type": "address", "indexed": True},
                        {"name": "position", "type": "uint8", "indexed": False},
                        {"name": "amount", "type": "uint256", "indexed": False},
                        {"name": "sUSDPaid", "type": "uint256", "indexed": False},
                        {"name": "susd", "type": "address", "indexed": False},
                        {"name": "asset", "type": "address", "indexed": False}
                    ]
                }
            ]
    
    SPORTS_AMM_ABI = load_abi()


class BlockchainReader:
    """Reads market data directly from blockchain."""
    
    def __init__(self, network: str = 'optimism', db_path: str = "blockchain_data.db"):
        if network not in BlockchainConfig.NETWORKS:
            raise ValueError(f"Unknown network: {network}")
        self.network = network
        self.config = BlockchainConfig.NETWORKS[network]
        self.db_path = db_path
        
        # Use RPC manager for better endpoint management
        try:
            from rpc_config import RPCManager
            self.rpc_manager = RPCManager(network)
            logger.info(f"Using RPC manager for {network}")
        except Exception as e:
            logger.warning(f"RPC manager initialization failed: {e}, falling back to default")
            self.rpc_manager = None
            self.rpc_url = os.getenv(f'{network.upper()}_RPC_URL', self.config['rpc_url'])
        
        # Initialize connection
        self._connect()
        
    def _connect(self):
        """Initialize or reinitialize Web3 connection."""
        # Initialize Web3
        if self.rpc_manager:
            try:
                self.w3, endpoint = self.rpc_manager.get_web3()
                logger.info(f"Connected via {endpoint.name}: {endpoint.url[:50]}...")
            except Exception as e:
                logger.error(f"RPC manager failed: {e}, falling back")
                self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
        else:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Add POA middleware for some networks
        if self.network in ['optimism', 'arbitrum']:
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.network}")
            
        logger.info(f"Connected to {self.network} at block {self.w3.eth.block_number}")
        
        # Initialize contract
        self.sports_amm = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.config['sports_amm_v2']),
            abi=BlockchainConfig.SPORTS_AMM_ABI
        )
        
        self._init_database()
        
    def _init_database(self):
        """Initialize blockchain data storage."""
        conn = sqlite3.connect(self.db_path)
        
        # Markets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blockchain_markets (
                market_address TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                game_label TEXT,
                maturity_date INTEGER,
                tags TEXT,
                normalized_odds TEXT,
                creation_block INTEGER,
                creation_tx TEXT,
                network TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trades table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blockchain_trades (
                tx_hash TEXT PRIMARY KEY,
                block_number INTEGER,
                timestamp INTEGER,
                buyer TEXT,
                market_address TEXT,
                position INTEGER,
                amount REAL,
                susd_paid REAL,
                price REAL,
                gas_used INTEGER,
                gas_price REAL,
                network TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Odds history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blockchain_odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_address TEXT NOT NULL,
                timestamp INTEGER,
                block_number INTEGER,
                position INTEGER,
                buy_odds REAL,
                sell_odds REAL,
                liquidity REAL,
                network TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
    def get_current_odds(self, market_address: str) -> Dict[int, Dict[str, float]]:
        """Get current odds for all positions in a market."""
        try:
            # Get buy odds
            buy_odds = self.sports_amm.functions.getMarketDefaultOdds(
                Web3.to_checksum_address(market_address),
                False  # isSell = False for buy
            ).call()
            
            # Get sell odds
            sell_odds = self.sports_amm.functions.getMarketDefaultOdds(
                Web3.to_checksum_address(market_address),
                True  # isSell = True for sell
            ).call()
            
            # Convert to decimal odds
            odds = {}
            for i, (buy, sell) in enumerate(zip(buy_odds, sell_odds)):
                if buy > 0:
                    odds[i] = {
                        'buy': 1e18 / buy,  # Convert from Wei representation
                        'sell': 1e18 / sell if sell > 0 else 0,
                        'spread': (1e18 / buy - 1e18 / sell) if sell > 0 else 0
                    }
                    
            return odds
            
        except Exception as e:
            logger.error(f"Error getting odds for {market_address}: {e}")
            return {}
            
    def scan_market_creations(self, from_block: int, to_block: Optional[int] = None):
        """Scan blockchain for new market creation events."""
        if to_block is None:
            to_block = self.w3.eth.block_number
            
        logger.info(f"Scanning blocks {from_block} to {to_block} for new markets")
        
        # Use getLogs directly instead of event filters
        try:
            # Calculate event signature
            event_signature = self.w3.to_hex(self.w3.keccak(
                text='MarketCreated(address,bytes32,string,uint256,uint256[],uint256[])'
            ))
            
            # Get logs directly - this works with most RPC providers
            logs = self.w3.eth.get_logs({
                'address': Web3.to_checksum_address(self.sports_amm.address),
                'fromBlock': from_block,
                'toBlock': to_block,
                'topics': [event_signature]
            })
            
            # Process logs into events
            events = []
            if hasattr(self.sports_amm.events, 'MarketCreated'):
                # Use contract to decode if available
                for log in logs:
                    try:
                        decoded = self.sports_amm.events.MarketCreated.process_log(log)
                        events.append(decoded)
                    except Exception as e:
                        logger.warning(f"Failed to decode log: {e}")
            else:
                # Manual decoding
                for log in logs:
                    events.append(log)
                    
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            
            # Try alternative approach with smaller block ranges
            if to_block - from_block > 100:
                logger.info("Retrying with smaller block ranges...")
                events = []
                batch_size = 100
                for start in range(from_block, to_block, batch_size):
                    end = min(start + batch_size, to_block)
                    try:
                        batch_events = self.scan_market_creations(start, end)
                        events.extend(batch_events)
                    except Exception as batch_e:
                        logger.error(f"Batch {start}-{end} failed: {batch_e}")
                return events
            else:
                return []
        markets = []
        
        for event in events:
            try:
                # Handle different event structures between web3 versions
                args = event.get('args', event)
                market_data = {
                    'market_address': args.get('market', ''),
                    'game_id': args.get('gameId', b'').hex() if isinstance(args.get('gameId'), bytes) else str(args.get('gameId', '')),
                    'game_label': args.get('gameLabel', ''),
                    'maturity_date': args.get('maturityDate', 0),
                    'tags': json.dumps(args.get('tags', [])),
                    'normalized_odds': json.dumps(args.get('normalizedOdds', [])),
                    'creation_block': event.get('blockNumber', 0),
                    'creation_tx': event.get('transactionHash', b'').hex() if isinstance(event.get('transactionHash'), bytes) else '',
                    'network': self.network
                }
            except Exception as e:
                logger.error(f"Error parsing event: {e}")
                continue
            
            markets.append(market_data)
            self._store_market(market_data)
            
        logger.info(f"Found {len(markets)} new markets")
        return markets
        
    def scan_trades(self, from_block: int, to_block: Optional[int] = None):
        """Scan blockchain for trading activity."""
        if to_block is None:
            to_block = self.w3.eth.block_number
            
        logger.info(f"Scanning blocks {from_block} to {to_block} for trades")
        
        # Use getLogs directly
        try:
            # Calculate event signature for BoughtFromAmm
            event_signature = self.w3.to_hex(self.w3.keccak(
                text='BoughtFromAmm(address,address,uint8,uint256,uint256,address,address)'
            ))
            
            # Get logs
            logs = self.w3.eth.get_logs({
                'address': Web3.to_checksum_address(self.sports_amm.address),
                'fromBlock': from_block,
                'toBlock': to_block,
                'topics': [event_signature]
            })
            
            # Process logs into events
            events = []
            if hasattr(self.sports_amm.events, 'BoughtFromAmm'):
                for log in logs:
                    try:
                        decoded = self.sports_amm.events.BoughtFromAmm.process_log(log)
                        events.append(decoded)
                    except Exception as e:
                        logger.warning(f"Failed to decode trade log: {e}")
            else:
                events = logs
                
        except Exception as e:
            logger.error(f"Error fetching trade logs: {e}")
            return []
        trades = []
        
        for event in events:
            # Get block timestamp
            block = self.w3.eth.get_block(event['blockNumber'])
            
            trade_data = {
                'tx_hash': event['transactionHash'].hex(),
                'block_number': event['blockNumber'],
                'timestamp': block['timestamp'],
                'buyer': event['args']['buyer'],
                'market_address': event['args']['market'],
                'position': event['args']['position'],
                'amount': float(Web3.from_wei(event['args']['amount'], 'ether')),
                'susd_paid': float(Web3.from_wei(event['args']['sUSDPaid'], 'ether')),
                'network': self.network
            }
            
            # Calculate implied price
            if trade_data['amount'] > 0:
                trade_data['price'] = trade_data['susd_paid'] / trade_data['amount']
            else:
                trade_data['price'] = 0
                
            # Get gas info
            tx_receipt = self.w3.eth.get_transaction_receipt(event['transactionHash'])
            trade_data['gas_used'] = tx_receipt['gasUsed']
            trade_data['gas_price'] = float(Web3.from_wei(
                tx_receipt['effectiveGasPrice'], 'gwei'
            ))
            
            trades.append(trade_data)
            self._store_trade(trade_data)
            
        logger.info(f"Found {len(trades)} trades")
        return trades
        
    def monitor_market_odds(self, market_addresses: List[str], 
                           interval_seconds: int = 60):
        """Monitor odds changes for specified markets."""
        logger.info(f"Monitoring {len(market_addresses)} markets")
        
        while True:
            for market in market_addresses:
                try:
                    odds = self.get_current_odds(market)
                    
                    # Store odds snapshot
                    for position, odds_data in odds.items():
                        self._store_odds_snapshot(
                            market, position, odds_data
                        )
                        
                except Exception as e:
                    logger.error(f"Error monitoring {market}: {e}")
                    
            # Sleep until next check
            asyncio.sleep(interval_seconds)
            
    def _store_market(self, market_data: Dict[str, Any]):
        """Store market creation data."""
        conn = sqlite3.connect(self.db_path)
        
        # Use INSERT OR REPLACE to handle duplicates
        conn.execute("""
            INSERT OR REPLACE INTO blockchain_markets (
                market_address, game_id, game_label, maturity_date,
                tags, normalized_odds, creation_block, creation_tx, network
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_data['market_address'],
            market_data['game_id'],
            market_data['game_label'],
            market_data['maturity_date'],
            market_data['tags'],
            market_data['normalized_odds'],
            market_data['creation_block'],
            market_data['creation_tx'],
            market_data['network']
        ))
        
        conn.commit()
        conn.close()
        
    def _store_trade(self, trade_data: Dict[str, Any]):
        """Store trade data."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO blockchain_trades (
                tx_hash, block_number, timestamp, buyer, market_address,
                position, amount, susd_paid, price, gas_used, gas_price, network
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data['tx_hash'],
            trade_data['block_number'],
            trade_data['timestamp'],
            trade_data['buyer'],
            trade_data['market_address'],
            trade_data['position'],
            trade_data['amount'],
            trade_data['susd_paid'],
            trade_data['price'],
            trade_data['gas_used'],
            trade_data['gas_price'],
            trade_data['network']
        ))
        
        conn.commit()
        conn.close()
        
    def _store_odds_snapshot(self, market_address: str, position: int,
                           odds_data: Dict[str, float]):
        """Store odds snapshot."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO blockchain_odds (
                market_address, timestamp, block_number, position,
                buy_odds, sell_odds, liquidity, network
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_address,
            int(datetime.now(timezone.utc).timestamp()),
            self.w3.eth.block_number,
            position,
            odds_data.get('buy', 0),
            odds_data.get('sell', 0),
            0,  # Would need to calculate from contract
            self.network
        ))
        
        conn.commit()
        conn.close()
        
    def get_market_history(self, market_address: str) -> pd.DataFrame:
        """Get complete history for a market."""
        conn = sqlite3.connect(self.db_path)
        
        # Get market info
        market_info = pd.read_sql_query("""
            SELECT * FROM blockchain_markets
            WHERE market_address = ?
        """, conn, params=(market_address,))
        
        # Get trades
        trades = pd.read_sql_query("""
            SELECT * FROM blockchain_trades
            WHERE market_address = ?
            ORDER BY timestamp
        """, conn, params=(market_address,))
        
        # Get odds history
        odds = pd.read_sql_query("""
            SELECT * FROM blockchain_odds
            WHERE market_address = ?
            ORDER BY timestamp
        """, conn, params=(market_address,))
        
        conn.close()
        
        return {
            'market_info': market_info,
            'trades': trades,
            'odds_history': odds
        }
    
    def check_connection(self) -> bool:
        """Check if blockchain connection is working."""
        try:
            # Try to get latest block
            latest_block = self.w3.eth.block_number
            return latest_block > 0
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def fetch_recent_markets(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch markets created in the last N hours."""
        try:
            # Ensure connection is alive
            if not self.w3.is_connected():
                logger.warning("Blockchain connection lost, reconnecting...")
                self._connect()
            
            # Calculate block range
            block_time = self.config.get('block_time', 2)  # Default to 2 seconds
            blocks_per_hour = int(3600 / block_time)
            
            # Get current block number with error handling
            try:
                current_block = self.w3.eth.block_number
                if current_block is None or current_block == 0:
                    logger.error("Unable to get current block number, attempting reconnect")
                    self._connect()
                    current_block = self.w3.eth.block_number
                    if current_block is None:
                        logger.error("Still unable to get block number after reconnect")
                        return []
            except Exception as e:
                logger.error(f"Failed to get current block number: {e}")
                return []
                
            from_block = max(0, current_block - int(hours_back * blocks_per_hour))
            
            # Scan for market creations
            markets = self.scan_market_creations(from_block)
            
            # Convert to standard format
            formatted_markets = []
            for market in markets:
                formatted_markets.append({
                    'source': 'blockchain',
                    'source_id': market['game_id'],
                    'address': market['market_address'],
                    'sport': self._decode_sport(market.get('tags', [])),
                    'league': self._decode_league(market.get('tags', [])),
                    'home_team': market.get('game_label', '').split(' vs ')[0] if ' vs ' in market.get('game_label', '') else '',
                    'away_team': market.get('game_label', '').split(' vs ')[1] if ' vs ' in market.get('game_label', '') else '',
                    'market_type': 'moneyline',
                    'maturity_date': datetime.fromtimestamp(market['maturity_date']),
                    'normalized_odds': market.get('normalized_odds', [])
                })
            
            return formatted_markets
        except Exception as e:
            logger.error(f"Error fetching recent markets: {e}")
            return []
    
    def _decode_sport(self, tags) -> str:
        """Decode sport from tags array."""
        # Handle JSON string input
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                return 'unknown'
        
        # Use enhanced tag mapper
        from enhanced_tag_mappings import tag_mapper
        return tag_mapper.get_sport_name(tags[0] if tags else 0)
    
    def _decode_league(self, tags) -> str:
        """Decode league from tags array."""
        # Handle JSON string input
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                return 'unknown'
                
        # Use enhanced tag mapper
        from enhanced_tag_mappings import tag_mapper
        return tag_mapper.get_league_name(tags[1] if len(tags) > 1 else 0)


class ChainSyncService:
    """Service to keep blockchain data synchronized."""
    
    def __init__(self, reader: BlockchainReader, 
                 start_block: Optional[int] = None):
        self.reader = reader
        self.last_synced_block = start_block or self._get_last_synced_block()
        
    def _get_last_synced_block(self) -> int:
        """Get the last block we've synced."""
        conn = sqlite3.connect(self.reader.db_path)
        
        result = conn.execute("""
            SELECT MAX(creation_block) as last_block
            FROM blockchain_markets
            WHERE network = ?
        """, (self.reader.network,)).fetchone()
        
        conn.close()
        
        if result and result[0]:
            return result[0]
        else:
            # Default to recent blocks if nothing synced
            return self.reader.w3.eth.block_number - 1000
            
    async def sync_continuously(self, batch_size: int = 100):
        """Continuously sync blockchain data."""
        while True:
            try:
                current_block = self.reader.w3.eth.block_number
                
                if self.last_synced_block < current_block:
                    # Process in batches
                    to_block = min(
                        self.last_synced_block + batch_size,
                        current_block
                    )
                    
                    logger.info(f"Syncing blocks {self.last_synced_block} "
                               f"to {to_block}")
                    
                    # Scan for new markets
                    self.reader.scan_market_creations(
                        self.last_synced_block,
                        to_block
                    )
                    
                    # Scan for trades
                    self.reader.scan_trades(
                        self.last_synced_block,
                        to_block
                    )
                    
                    self.last_synced_block = to_block
                    
                # Sleep based on block time
                await asyncio.sleep(self.reader.config['block_time'] * 2)
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(10)  # Wait before retry


def main():
    """Example usage."""
    # Initialize reader
    reader = BlockchainReader(network='optimism')
    
    # Get current odds for a market
    market = "0x1234..."  # Example market address
    # odds = reader.get_current_odds(market)
    # print(f"Current odds: {odds}")
    
    # Scan recent blocks
    recent_markets = reader.scan_market_creations(
        from_block=reader.w3.eth.block_number - 100
    )
    print(f"Found {len(recent_markets)} markets")
    
    # Start continuous sync
    sync_service = ChainSyncService(reader)
    # asyncio.run(sync_service.sync_continuously())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()