#!/usr/bin/env python3
"""
Demo of Complete Blockchain Data Collection System

This script demonstrates the full blockchain-based sports betting system
by creating mock data and showing how it integrates with the trading system.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import time

from enhanced_tag_mappings import tag_mapper, SportID, LeagueID
from team_metadata_service import TeamMetadataService, TeamMetadata
from market_enrichment import MarketEnrichmentService
from blockchain_to_db_migrator import BlockchainToDbMigrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlockchainSystemDemo:
    """Demonstrates the complete blockchain data collection system."""
    
    def __init__(self):
        self.metadata_service = TeamMetadataService()
        self.enrichment_service = MarketEnrichmentService()
        self.migrator = BlockchainToDbMigrator()
        
    def create_demo_blockchain_data(self):
        """Create demo blockchain data for testing."""
        logger.info("Creating demo blockchain market data...")
        
        # Connect to blockchain database
        conn = sqlite3.connect("blockchain_data.db")
        
        # Create demo markets
        demo_markets = [
            {
                'market_address': '0x1234567890123456789012345678901234567890',
                'game_id': 'demo_soccer_001',
                'game_label': 'Liverpool vs Manchester City',
                'maturity_date': int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()),
                'tags': json.dumps([SportID.SOCCER, 501]),  # EPL
                'normalized_odds': json.dumps([2.1, 3.5, 3.2]),
                'creation_block': 141000000,
                'creation_tx': '0xabcd...',
                'network': 'optimism'
            },
            {
                'market_address': '0x2345678901234567890123456789012345678901',
                'game_id': 'demo_nfl_001', 
                'game_label': 'Cowboys vs Patriots',
                'maturity_date': int((datetime.now(timezone.utc) + timedelta(hours=48)).timestamp()),
                'tags': json.dumps([SportID.FOOTBALL, 101]),  # NFL
                'normalized_odds': json.dumps([1.9, 2.0]),
                'creation_block': 141000100,
                'creation_tx': '0xefgh...',
                'network': 'optimism'
            },
            {
                'market_address': '0x3456789012345678901234567890123456789012',
                'game_id': 'demo_nba_001',
                'game_label': 'Lakers vs Warriors',
                'maturity_date': int((datetime.now(timezone.utc) + timedelta(hours=12)).timestamp()),
                'tags': json.dumps([SportID.BASKETBALL, 201]),  # NBA
                'normalized_odds': json.dumps([2.3, 1.7]),
                'creation_block': 141000200,
                'creation_tx': '0xijkl...',
                'network': 'optimism'
            }
        ]
        
        # Insert demo markets
        for market in demo_markets:
            conn.execute("""
                INSERT OR REPLACE INTO blockchain_markets (
                    market_address, game_id, game_label, maturity_date,
                    tags, normalized_odds, creation_block, creation_tx, network
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market['market_address'], market['game_id'], market['game_label'],
                market['maturity_date'], market['tags'], market['normalized_odds'],
                market['creation_block'], market['creation_tx'], market['network']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Created {len(demo_markets)} demo blockchain markets")
        return demo_markets
    
    def create_demo_odds_data(self, market_addresses: List[str]):
        """Create demo odds data for the markets."""
        logger.info("Creating demo odds data...")
        
        conn = sqlite3.connect("blockchain_data.db")
        
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        current_block = 141000300
        
        # Demo odds for each market
        for market_address in market_addresses:
            # Create multiple odds updates over time
            for i in range(5):
                timestamp = current_timestamp - (i * 3600)  # Hourly updates
                
                # Soccer market - 3 positions
                if 'Liverpool' in market_address or 'demo_soccer' in market_address:
                    positions = [
                        (0, 2.1 + i * 0.1, 2.05 + i * 0.1),  # Home
                        (1, 3.5 - i * 0.1, 3.45 - i * 0.1),  # Away  
                        (2, 3.2 + i * 0.05, 3.15 + i * 0.05) # Draw
                    ]
                else:
                    # US sports - 2 positions
                    positions = [
                        (0, 1.9 + i * 0.05, 1.85 + i * 0.05),  # Home
                        (1, 2.0 - i * 0.05, 1.95 - i * 0.05)   # Away
                    ]
                
                for position, buy_odds, sell_odds in positions:
                    conn.execute("""
                        INSERT INTO blockchain_odds (
                            market_address, timestamp, block_number, position,
                            buy_odds, sell_odds, liquidity, network
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_address, timestamp, current_block + i,
                        position, buy_odds, sell_odds, 100000.0, 'optimism'
                    ))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Created demo odds data")
    
    def test_enrichment_service(self):
        """Test the market enrichment service."""
        logger.info("Testing market enrichment service...")
        
        # Test enriching a market
        market_address = '0x1234567890123456789012345678901234567890'
        enriched = self.enrichment_service.enrich_market(market_address, 'optimism')
        
        if enriched:
            logger.info("✅ Market enrichment successful!")
            logger.info(f"   Sport: {enriched.sport.name}")
            logger.info(f"   League: {enriched.league.name}")
            logger.info(f"   Match: {enriched.home_team.full_name} vs {enriched.away_team.full_name}")
            logger.info(f"   Venue: {enriched.home_team.venue}")
            logger.info(f"   Odds: Home {enriched.current_odds.get('home', 0):.2f}")
            logger.info(f"         Away {enriched.current_odds.get('away', 0):.2f}")
            if enriched.has_draw:
                logger.info(f"         Draw {enriched.current_odds.get('draw', 0):.2f}")
            return enriched
        else:
            logger.error("❌ Market enrichment failed")
            return None
    
    def test_migration_system(self):
        """Test migrating blockchain data to main database."""
        logger.info("Testing blockchain to database migration...")
        
        # Run migration
        stats = self.migrator.migrate_markets('optimism', batch_size=10)
        
        logger.info("✅ Migration completed!")
        logger.info(f"   New markets: {stats['new_markets']}")
        logger.info(f"   Updated markets: {stats['updated_markets']}")
        logger.info(f"   Errors: {stats['errors']}")
        
        return stats
    
    def test_odds_sync(self):
        """Test odds synchronization."""
        logger.info("Testing odds synchronization...")
        
        stats = self.migrator.sync_odds_updates('optimism', hours_back=1)
        
        logger.info("✅ Odds sync completed!")
        logger.info(f"   Markets checked: {stats['markets_checked']}")
        logger.info(f"   Odds updated: {stats['odds_updated']}")
        logger.info(f"   Errors: {stats['errors']}")
        
        return stats
    
    def show_final_status(self):
        """Show the final system status."""
        logger.info("Blockchain Data Collection System Status:")
        logger.info("=" * 60)
        
        status = self.migrator.get_migration_status()
        
        for network, stats in status.items():
            if network != 'overall':
                logger.info(f"\n{network.upper()}:")
                logger.info(f"  Total blockchain markets: {stats['total_markets']:,}")
                logger.info(f"  Migrated to main DB: {stats['migrated_markets']:,}")
                logger.info(f"  Completion: {stats['completion_pct']:.1f}%")
        
        logger.info(f"\nOverall System:")
        logger.info(f"  Markets in main DB: {status['overall']['total_markets']:,}")
        logger.info(f"  Odds records: {status['overall']['total_odds_records']:,}")
        
        # Show team metadata stats
        team_stats = self.metadata_service.get_stats()
        logger.info(f"  Teams in database: {team_stats['total_teams']}")
        logger.info(f"  Sports covered: {len(team_stats['sports'])}")
        logger.info(f"  Leagues covered: {len(team_stats['leagues'])}")
    
    def run_complete_demo(self):
        """Run the complete system demonstration."""
        logger.info("🚀 Starting Complete Blockchain Data Collection System Demo")
        logger.info("=" * 80)
        
        try:
            # 1. Create demo data
            demo_markets = self.create_demo_blockchain_data()
            market_addresses = [m['market_address'] for m in demo_markets]
            
            # 2. Create odds data
            self.create_demo_odds_data(market_addresses)
            
            # 3. Test enrichment
            enriched = self.test_enrichment_service()
            
            if enriched:
                # 4. Test migration
                migration_stats = self.test_migration_system()
                
                # 5. Test odds sync
                odds_stats = self.test_odds_sync()
                
                # 6. Show final status
                self.show_final_status()
                
                logger.info("\n🎉 Demo completed successfully!")
                logger.info("The blockchain data collection system is ready for use.")
                logger.info("\nNext steps:")
                logger.info("1. Start the sync daemon: python blockchain_sync_daemon.py")
                logger.info("2. Use the unified CLI: python ominari_unified.py quick-start")
                logger.info("3. Run backtests with blockchain data")
                
            else:
                logger.error("❌ Demo failed during enrichment step")
                
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo = BlockchainSystemDemo()
    demo.run_complete_demo()