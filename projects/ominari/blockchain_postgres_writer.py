#!/usr/bin/env python3
"""
PostgreSQL Writer for Blockchain Data
Writes blockchain market and odds data directly to PostgreSQL hybrid system.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlockchainPostgresWriter:
    """Writes blockchain data to PostgreSQL hybrid system."""
    
    def __init__(self):
        # PostgreSQL connection details
        self.pg_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5435'),
            'user': os.getenv('PG_USER', 'ominari_user'),
            'password': os.getenv('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.getenv('PG_DB', 'ominari_production')
        }
        
        self._test_connection()
        logger.info("PostgreSQL blockchain writer initialized")
    
    def _test_connection(self):
        """Test PostgreSQL connection."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"Connected to PostgreSQL: {version[:50]}...")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def store_market(self, market_data: Dict[str, Any]) -> bool:
        """Store blockchain market in PostgreSQL."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    # Parse tags and odds if they're JSON strings
                    tags = market_data.get('tags', '[]')
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except:
                            tags = []
                    
                    odds = market_data.get('normalized_odds', '[]')
                    if isinstance(odds, str):
                        try:
                            odds = json.loads(odds)
                        except:
                            odds = []
                    
                    # Decode sport and league from tags
                    sport = self._decode_sport(tags)
                    league = self._decode_league(tags)
                    
                    # Parse team names from game label
                    game_label = market_data.get('game_label', '')
                    if ' vs ' in game_label:
                        teams = game_label.split(' vs ', 1)
                        home_team = teams[0].strip()
                        away_team = teams[1].strip()
                    else:
                        home_team = 'Home'
                        away_team = 'Away'
                    
                    # Convert maturity timestamp to datetime
                    maturity_timestamp = market_data.get('maturity_date', 0)
                    if maturity_timestamp:
                        start_time = datetime.fromtimestamp(maturity_timestamp, tz=timezone.utc)
                    else:
                        start_time = datetime.now(tz=timezone.utc)
                    
                    # Map network to chain_id
                    network_to_chain = {
                        'optimism': 10,
                        'arbitrum': 42161,
                        'optimism_sepolia': 11155420,
                        'test_network': 999
                    }
                    chain_id = network_to_chain.get(market_data.get('network', 'unknown'), 1)
                    
                    # Insert market
                    cursor.execute("""
                        INSERT INTO blockchain.markets (
                            chain_id, chain_name, market_address, market_id,
                            sport, league, home_team, away_team, start_time, resolved,
                            block_number, tx_hash
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (market_id) DO UPDATE SET
                            sport = EXCLUDED.sport,
                            league = EXCLUDED.league,
                            home_team = EXCLUDED.home_team,
                            away_team = EXCLUDED.away_team,
                            start_time = EXCLUDED.start_time,
                            resolved = EXCLUDED.resolved
                    """, (
                        chain_id,
                        market_data.get('network', 'unknown'),
                        market_data.get('market_address', ''),  
                        market_data.get('market_address', ''),  # Use address as market_id
                        sport,
                        league, 
                        home_team,
                        away_team,
                        start_time,
                        False,  # resolved - assume new markets are not resolved
                        market_data.get('creation_block', 0),
                        market_data.get('creation_tx', '')
                    ))
                    
                    # Also store initial odds if available
                    if odds:
                        self._store_initial_odds(cursor, market_data, odds, start_time)
                    
                    logger.debug(f"Stored market {market_data.get('market_address', 'unknown')}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing market: {e}")
            return False
    
    def _store_initial_odds(self, cursor, market_data: Dict[str, Any], 
                           odds: List[float], timestamp: datetime):
        """Store initial odds for the market."""
        market_id = market_data.get('market_address', '')
        
        # Map network to chain_id
        network_to_chain = {
            'optimism': 10,
            'arbitrum': 42161,
            'optimism_sepolia': 11155420,
            'test_network': 999
        }
        chain_id = network_to_chain.get(market_data.get('network', 'unknown'), 1)
        
        for position, odd_value in enumerate(odds):
            if odd_value > 0:
                # Convert from normalized odds to decimal odds
                decimal_odds = 1.0 / (odd_value / 1e18) if odd_value > 0 else 1.0
                implied_prob = (odd_value / 1e18) if odd_value > 0 else 0.0
                
                cursor.execute("""
                    INSERT INTO blockchain.odds (
                        market_id, chain_id, outcome, decimal_odds, 
                        implied_probability, timestamp, block_number, 
                        tx_hash, source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    market_id,
                    chain_id,
                    f'position_{position}',
                    decimal_odds,
                    implied_prob,
                    timestamp,
                    market_data.get('creation_block', 0),
                    market_data.get('creation_tx', ''),
                    'market_creation'
                ))
    
    def store_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Store blockchain trade in PostgreSQL."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    # Convert timestamp
                    trade_timestamp = trade_data.get('timestamp', 0)
                    if trade_timestamp:
                        timestamp = datetime.fromtimestamp(trade_timestamp, tz=timezone.utc)
                    else:
                        timestamp = datetime.now(tz=timezone.utc)
                    
                    # Calculate implied odds from trade
                    amount = trade_data.get('amount', 0)
                    susd_paid = trade_data.get('susd_paid', 0)
                    implied_odds = amount / susd_paid if susd_paid > 0 else 1.0
                    
                    # Map network to chain_id
                    network_to_chain = {
                        'optimism': 10,
                        'arbitrum': 42161,
                        'optimism_sepolia': 11155420,
                        'test_network': 999
                    }
                    chain_id = network_to_chain.get(trade_data.get('network', 'unknown'), 1)
                    
                    # Store trade as an odds entry
                    cursor.execute("""
                        INSERT INTO blockchain.odds (
                            market_id, chain_id, outcome, decimal_odds, timestamp,
                            block_number, tx_hash, source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        trade_data.get('market_address', ''),
                        chain_id,
                        f"position_{trade_data.get('position', 0)}",
                        implied_odds,
                        timestamp,
                        trade_data.get('block_number', 0),
                        trade_data.get('tx_hash', ''),
                        'trade'
                    ))
                    
                    logger.debug(f"Stored trade {trade_data.get('tx_hash', 'unknown')}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
            return False
    
    def store_odds_snapshot(self, market_id: str, position: int, 
                           odds_data: Dict[str, float], network: str) -> bool:
        """Store current odds snapshot."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    timestamp = datetime.now(tz=timezone.utc)
                    
                    # Map network to chain_id
                    network_to_chain = {
                        'optimism': 10,
                        'arbitrum': 42161,
                        'optimism_sepolia': 11155420,
                        'test_network': 999
                    }
                    chain_id = network_to_chain.get(network, 1)
                    
                    # Store buy odds
                    if 'buy' in odds_data and odds_data['buy'] > 0:
                        cursor.execute("""
                            INSERT INTO blockchain.odds (
                                market_id, chain_id, outcome, decimal_odds, 
                                timestamp, block_number, source
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            market_id,
                            chain_id,
                            f"position_{position}",
                            odds_data['buy'],
                            timestamp,
                            0,  # No block number for snapshots
                            'snapshot_buy'
                        ))
                    
                    # Store sell odds if available
                    if 'sell' in odds_data and odds_data['sell'] > 0:
                        cursor.execute("""
                            INSERT INTO blockchain.odds (
                                market_id, chain_id, outcome, decimal_odds, 
                                timestamp, block_number, source
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            market_id,
                            chain_id,
                            f"position_{position}_sell",
                            odds_data['sell'],
                            timestamp,
                            0,  # No block number for snapshots
                            'snapshot_sell'
                        ))
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing odds snapshot: {e}")
            return False
    
    def _decode_sport(self, tags: List[int]) -> str:
        """Decode sport from tags array."""
        if not tags:
            return 'Unknown'
        
        # Enhanced sport mapping
        sport_map = {
            0: 'American Football',
            1: 'Basketball', 
            2: 'Soccer',
            3: 'Baseball',
            4: 'Ice Hockey',
            5: 'Tennis',
            6: 'MMA',
            7: 'Boxing',
            8: 'Motor Racing',
            9: 'eSports',
            10: 'Cricket',
            11: 'Rugby',
            12: 'Golf'
        }
        
        return sport_map.get(tags[0], f'Sport_{tags[0]}')
    
    def _decode_league(self, tags: List[int]) -> str:
        """Decode league from tags array."""
        if len(tags) < 2:
            return 'Unknown'
        
        # Enhanced league mapping - simplified version
        league_map = {
            # NFL
            (0, 1): 'NFL',
            # NBA
            (1, 1): 'NBA',
            # Soccer leagues
            (2, 1): 'Premier League',
            (2, 2): 'La Liga', 
            (2, 3): 'Bundesliga',
            (2, 4): 'Serie A',
            (2, 5): 'Ligue 1',
            (2, 6): 'Champions League',
            (2, 7): 'Europa League',
            # MLB
            (3, 1): 'MLB',
            # NHL  
            (4, 1): 'NHL',
            # Tennis
            (5, 1): 'ATP',
            (5, 2): 'WTA'
        }
        
        key = (tags[0], tags[1])
        return league_map.get(key, f'League_{tags[1]}')
    
    def get_recent_markets(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recently stored markets from PostgreSQL."""
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT market_id, chain_name, sport, league, home_team, away_team,
                               start_time, resolved, created_at
                        FROM blockchain.markets
                        WHERE created_at >= NOW() - INTERVAL '%s hours'
                        ORDER BY created_at DESC
                        LIMIT 100
                    """, (hours_back,))
                    
                    markets = []
                    for row in cursor.fetchall():
                        markets.append(dict(row))
                    
                    return markets
                    
        except Exception as e:
            logger.error(f"Error getting recent markets: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain data statistics."""
        stats = {}
        
        try:
            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cursor:
                    # Market counts
                    cursor.execute("SELECT COUNT(*) FROM blockchain.markets")
                    stats['total_markets'] = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT COUNT(*) FROM blockchain.markets
                        WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """)
                    stats['markets_24h'] = cursor.fetchone()[0]
                    
                    # Odds counts
                    cursor.execute("SELECT COUNT(*) FROM blockchain.odds")
                    stats['total_odds'] = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT COUNT(*) FROM blockchain.odds
                        WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    """)
                    stats['odds_24h'] = cursor.fetchone()[0]
                    
                    # Network distribution
                    cursor.execute("""
                        SELECT chain_name, COUNT(*) as count
                        FROM blockchain.markets
                        GROUP BY chain_name
                        ORDER BY count DESC
                    """)
                    stats['networks'] = dict(cursor.fetchall())
                    
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            
        return stats
    
    def test_write_operations(self):
        """Test the PostgreSQL write operations."""
        logger.info("🧪 Testing PostgreSQL blockchain writer...")
        
        # Test market storage
        test_market = {
            'market_address': '0xtest123456789abcdef',
            'game_id': 'test_game_123',
            'game_label': 'Test Team A vs Test Team B',
            'maturity_date': int(datetime.now(tz=timezone.utc).timestamp()) + 3600,
            'tags': [2, 1],  # Soccer, Premier League
            'normalized_odds': [600000000000000000, 400000000000000000],  # 1.67, 2.5 odds
            'creation_block': 12345,
            'creation_tx': '0xtestransaction',
            'network': 'test_network'
        }
        
        if self.store_market(test_market):
            logger.info("✅ Market storage test passed")
        else:
            logger.error("❌ Market storage test failed")
        
        # Test trade storage
        test_trade = {
            'tx_hash': '0xtesttrade123',
            'block_number': 12346,
            'timestamp': int(datetime.now(tz=timezone.utc).timestamp()),
            'buyer': '0xtestbuyer',
            'market_address': '0xtest123456789abcdef',
            'position': 0,
            'amount': 10.0,
            'susd_paid': 6.0,
            'price': 0.6,
            'network': 'test_network'
        }
        
        if self.store_trade(test_trade):
            logger.info("✅ Trade storage test passed")
        else:
            logger.error("❌ Trade storage test failed")
        
        # Test statistics
        stats = self.get_statistics()
        logger.info(f"📊 Current stats: {stats['total_markets']} markets, {stats['total_odds']} odds")
        
        logger.info("✅ PostgreSQL writer testing complete")


def main():
    """Test the PostgreSQL writer."""
    try:
        writer = BlockchainPostgresWriter()
        writer.test_write_operations()
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()