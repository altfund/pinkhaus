#!/usr/bin/env python3
"""
Blockchain to Database Migrator

Migrates blockchain market data to the existing sport_odds.db schema,
allowing seamless integration with existing trading systems.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
import time

from enhanced_tag_mappings import tag_mapper
from team_metadata_service import TeamMetadataService
from market_enrichment import MarketEnrichmentService

logger = logging.getLogger(__name__)


class BlockchainToDbMigrator:
    """Migrates blockchain data to existing database schema."""
    
    def __init__(self, 
                 sport_odds_db: str = "sport_odds.db",
                 blockchain_db: str = "blockchain_data.db"):
        self.sport_odds_db = sport_odds_db
        self.blockchain_db = blockchain_db
        self.metadata_service = TeamMetadataService()
        self.enrichment_service = MarketEnrichmentService()
        
        # Track migrated markets
        self._init_migration_tracking()
    
    def _init_migration_tracking(self):
        """Initialize migration tracking table."""
        conn = sqlite3.connect(self.sport_odds_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blockchain_migration (
                blockchain_address TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                migrated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                network TEXT,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def migrate_markets(self, network: str = 'optimism', 
                       batch_size: int = 100) -> Dict[str, int]:
        """
        Migrate markets from blockchain to sport_odds.db.
        
        Returns:
            Statistics about migration
        """
        stats = {
            'total_markets': 0,
            'new_markets': 0,
            'updated_markets': 0,
            'skipped_markets': 0,
            'errors': 0
        }
        
        logger.info(f"Starting migration from {network}...")
        
        # Get blockchain markets
        blockchain_conn = sqlite3.connect(self.blockchain_db)
        
        # Get already migrated markets
        sport_conn = sqlite3.connect(self.sport_odds_db)
        migrated = sport_conn.execute("""
            SELECT blockchain_address FROM blockchain_migration
            WHERE network = ?
        """, (network,)).fetchall()
        migrated_addresses = {row[0] for row in migrated}
        sport_conn.close()
        
        # Get markets to migrate
        markets = blockchain_conn.execute("""
            SELECT * FROM blockchain_markets
            WHERE network = ?
            ORDER BY creation_block DESC
        """, (network,)).fetchall()
        
        blockchain_conn.close()
        
        stats['total_markets'] = len(markets)
        logger.info(f"Found {len(markets)} markets on {network}")
        
        # Process in batches
        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} markets)")
            
            for market_row in batch:
                market_address = market_row[0]
                
                # Skip if already migrated
                if market_address in migrated_addresses:
                    stats['skipped_markets'] += 1
                    continue
                
                try:
                    # Enrich market
                    enriched = self.enrichment_service.enrich_market(
                        market_address, network
                    )
                    
                    if not enriched:
                        logger.warning(f"Could not enrich {market_address}")
                        stats['errors'] += 1
                        continue
                    
                    # Migrate to sport_odds.db
                    self._migrate_market(enriched, network)
                    stats['new_markets'] += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating {market_address}: {e}")
                    stats['errors'] += 1
            
            # Brief pause between batches
            time.sleep(0.1)
        
        logger.info(f"Migration complete: {stats}")
        return stats
    
    def _migrate_market(self, enriched_market, network: str):
        """Migrate a single enriched market."""
        conn = sqlite3.connect(self.sport_odds_db)
        
        try:
            # Generate market_id compatible with existing schema
            market_id = f"blockchain_{enriched_market.game_id}"
            
            # Insert into market table (using actual schema)
            conn.execute("""
                INSERT OR REPLACE INTO market (
                    source_id, source, sport, league_name,
                    home_team, away_team, market_type, maturity_date,
                    is_finished, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_id,
                f'blockchain_{network}',
                enriched_market.sport.name,
                enriched_market.league.name,
                enriched_market.home_team.full_name,
                enriched_market.away_team.full_name,
                'moneyline',
                enriched_market.starts_at.isoformat(),
                enriched_market.starts_at < datetime.now(timezone.utc),
                datetime.now(timezone.utc).isoformat()
            ))
            
            # Insert current odds (using actual schema)
            if enriched_market.current_odds:
                # Home odds
                if enriched_market.current_odds.get('home', 0) > 0:
                    conn.execute("""
                        INSERT OR REPLACE INTO odd (
                            source_id, position, market_type, outcome, source, bookmaker,
                            decimal_odds, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, 0, 'moneyline', 'Home', f'blockchain_{network}',
                        f'blockchain_{network}', enriched_market.current_odds.get('home', 0),
                        datetime.now(timezone.utc).isoformat()
                    ))
                
                # Away odds
                if enriched_market.current_odds.get('away', 0) > 0:
                    conn.execute("""
                        INSERT OR REPLACE INTO odd (
                            source_id, position, market_type, outcome, source, bookmaker,
                            decimal_odds, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, 1, 'moneyline', 'Away', f'blockchain_{network}',
                        f'blockchain_{network}', enriched_market.current_odds.get('away', 0),
                        datetime.now(timezone.utc).isoformat()
                    ))
                
                # Draw odds if available
                if enriched_market.current_odds.get('draw') and enriched_market.current_odds.get('draw') > 0:
                    conn.execute("""
                        INSERT OR REPLACE INTO odd (
                            source_id, position, market_type, outcome, source, bookmaker,
                            decimal_odds, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, 2, 'moneyline', 'Draw', f'blockchain_{network}',
                        f'blockchain_{network}', enriched_market.current_odds.get('draw', 0),
                        datetime.now(timezone.utc).isoformat()
                    ))
            
            # Track migration
            conn.execute("""
                INSERT INTO blockchain_migration (
                    blockchain_address, market_id, network, status
                ) VALUES (?, ?, ?, ?)
            """, (
                enriched_market.blockchain_address,
                market_id,
                network,
                'completed'
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def sync_odds_updates(self, network: str = 'optimism', 
                         hours_back: int = 24) -> Dict[str, int]:
        """
        Sync recent odds updates from blockchain.
        
        Returns:
            Statistics about updates
        """
        stats = {
            'markets_checked': 0,
            'odds_updated': 0,
            'errors': 0
        }
        
        logger.info(f"Syncing odds updates for last {hours_back} hours...")
        
        # Get recent odds from blockchain
        blockchain_conn = sqlite3.connect(self.blockchain_db)
        
        cutoff_timestamp = int((datetime.now(timezone.utc).timestamp() - 
                               hours_back * 3600))
        
        odds_updates = blockchain_conn.execute("""
            SELECT DISTINCT market_address, MAX(block_number) as latest_block
            FROM blockchain_odds
            WHERE network = ? AND timestamp > ?
            GROUP BY market_address
        """, (network, cutoff_timestamp)).fetchall()
        
        blockchain_conn.close()
        
        stats['markets_checked'] = len(odds_updates)
        
        for market_address, latest_block in odds_updates:
            try:
                # Get enriched market
                enriched = self.enrichment_service.enrich_market(
                    market_address, network
                )
                
                if enriched and enriched.current_odds:
                    # Update odds in sport_odds.db
                    self._update_odds(enriched, network)
                    stats['odds_updated'] += 1
                    
            except Exception as e:
                logger.error(f"Error updating odds for {market_address}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Odds sync complete: {stats}")
        return stats
    
    def _update_odds(self, enriched_market, network: str):
        """Update odds for an existing market."""
        conn = sqlite3.connect(self.sport_odds_db)
        
        try:
            market_id = f"blockchain_{enriched_market.game_id}"
            
            # Update odds for existing market (using actual schema)
            # Home odds
            if enriched_market.current_odds.get('home', 0) > 0:
                conn.execute("""
                    INSERT OR REPLACE INTO odd (
                        source_id, position, market_type, outcome, source, bookmaker,
                        decimal_odds, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, 0, 'moneyline', 'Home', f'blockchain_{network}',
                    f'blockchain_{network}', enriched_market.current_odds.get('home', 0),
                    datetime.now(timezone.utc).isoformat()
                ))
            
            # Away odds
            if enriched_market.current_odds.get('away', 0) > 0:
                conn.execute("""
                    INSERT OR REPLACE INTO odd (
                        source_id, position, market_type, outcome, source, bookmaker,
                        decimal_odds, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, 1, 'moneyline', 'Away', f'blockchain_{network}',
                    f'blockchain_{network}', enriched_market.current_odds.get('away', 0),
                    datetime.now(timezone.utc).isoformat()
                ))
            
            # Draw odds if available
            if enriched_market.current_odds.get('draw') and enriched_market.current_odds.get('draw') > 0:
                conn.execute("""
                    INSERT OR REPLACE INTO odd (
                        source_id, position, market_type, outcome, source, bookmaker,
                        decimal_odds, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, 2, 'moneyline', 'Draw', f'blockchain_{network}',
                    f'blockchain_{network}', enriched_market.current_odds.get('draw', 0),
                    datetime.now(timezone.utc).isoformat()
                ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        sport_conn = sqlite3.connect(self.sport_odds_db)
        blockchain_conn = sqlite3.connect(self.blockchain_db)
        
        status = {}
        
        # Get blockchain market counts
        for network in ['optimism', 'arbitrum']:
            blockchain_count = blockchain_conn.execute("""
                SELECT COUNT(*) FROM blockchain_markets
                WHERE network = ?
            """, (network,)).fetchone()[0]
            
            migrated_count = sport_conn.execute("""
                SELECT COUNT(*) FROM blockchain_migration
                WHERE network = ?
            """, (network,)).fetchone()[0]
            
            status[network] = {
                'total_markets': blockchain_count,
                'migrated_markets': migrated_count,
                'pending_markets': blockchain_count - migrated_count,
                'completion_pct': (migrated_count / blockchain_count * 100) 
                                 if blockchain_count > 0 else 0
            }
        
        # Get overall stats
        total_blockchain_markets = sport_conn.execute("""
            SELECT COUNT(*) FROM market
            WHERE source LIKE 'blockchain_%'
        """).fetchone()[0]
        
        total_blockchain_odds = sport_conn.execute("""
            SELECT COUNT(*) FROM odd
            WHERE bookmaker LIKE 'blockchain_%'
        """).fetchone()[0]
        
        status['overall'] = {
            'total_markets': total_blockchain_markets,
            'total_odds_records': total_blockchain_odds
        }
        
        sport_conn.close()
        blockchain_conn.close()
        
        return status


def main():
    """Run migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate blockchain data')
    parser.add_argument('--network', default='optimism', 
                       choices=['optimism', 'arbitrum'],
                       help='Network to migrate from')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for migration')
    parser.add_argument('--sync-odds', action='store_true',
                       help='Sync recent odds updates')
    parser.add_argument('--status', action='store_true',
                       help='Show migration status')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    migrator = BlockchainToDbMigrator()
    
    if args.status:
        status = migrator.get_migration_status()
        print("\nMigration Status:")
        print("-" * 50)
        for network, stats in status.items():
            if network != 'overall':
                print(f"\n{network.upper()}:")
                print(f"  Total markets: {stats['total_markets']:,}")
                print(f"  Migrated: {stats['migrated_markets']:,}")
                print(f"  Pending: {stats['pending_markets']:,}")
                print(f"  Completion: {stats['completion_pct']:.1f}%")
        
        print(f"\nOverall:")
        print(f"  Blockchain markets in DB: {status['overall']['total_markets']:,}")
        print(f"  Blockchain odds records: {status['overall']['total_odds_records']:,}")
        
    elif args.sync_odds:
        stats = migrator.sync_odds_updates(args.network)
        print(f"\nOdds sync complete: {stats}")
        
    else:
        stats = migrator.migrate_markets(args.network, args.batch_size)
        print(f"\nMigration complete: {stats}")


if __name__ == "__main__":
    main()