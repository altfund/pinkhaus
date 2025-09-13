#!/usr/bin/env python3
"""
Run complete historical data backfill.

This script runs all the backfill operations in the correct order:
1. Backfill match results
2. Enrich paper trading positions with odds/edges
3. Set up automated catch-up
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("="*70)
    print("OMINARI HISTORICAL DATA BACKFILL")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Step 1: Run the main backfill
    print("\n📊 STEP 1: Backfilling historical match results...")
    print("-"*50)
    try:
        from backfill_historical_data import HistoricalDataBackfiller
        backfiller = HistoricalDataBackfiller()
        backfiller.run_full_backfill()
        print("✅ Match results backfill complete!")
    except Exception as e:
        logger.error(f"Error in match results backfill: {e}")
        print(f"❌ Failed: {e}")
        return 1
    
    # Step 2: Enrich trading history
    print("\n💰 STEP 2: Enriching paper trading history with odds/edges...")
    print("-"*50)
    try:
        from enrich_trading_history import TradingHistoryEnricher
        enricher = TradingHistoryEnricher()
        enricher.enrich_all_sessions(verify_pnl=True)
        print("✅ Trading history enrichment complete!")
    except Exception as e:
        logger.error(f"Error in trading history enrichment: {e}")
        print(f"❌ Failed: {e}")
        return 1
    
    # Step 3: Set up catch-up service
    print("\n🔄 STEP 3: Setting up automated catch-up service...")
    print("-"*50)
    try:
        from results_catchup_service import ResultsCatchupService
        service = ResultsCatchupService()
        
        # Run initial catch-up
        print("Running initial catch-up for last 24 hours...")
        service.run_catchup()
        
        # Create systemd service file
        create_systemd_service()
        
        print("✅ Catch-up service configured!")
    except Exception as e:
        logger.error(f"Error setting up catch-up service: {e}")
        print(f"❌ Failed: {e}")
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("BACKFILL COMPLETE!")
    print("="*70)
    print("\n✅ All historical data has been backfilled:")
    print("  - Match results updated")
    print("  - Paper trading positions enriched with odds/edges")
    print("  - Catch-up service configured")
    print("\n📋 Next steps:")
    print("  1. Check the enriched data in your paper trading sessions")
    print("  2. Set up the catch-up service to run automatically:")
    print("     - Using cron: python results_catchup_service.py --mode setup")
    print("     - Using systemd: sudo systemctl enable ominari-catchup.service")
    print("  3. Monitor logs/catchup.log for ongoing updates")
    print("\n✨ Your historical data is now complete and will stay up-to-date!")
    
    return 0

def create_systemd_service():
    """Create a systemd service file for the catch-up service."""
    service_content = """[Unit]
Description=Ominari Results Catch-up Service
After=network.target

[Service]
Type=simple
User=ess
WorkingDirectory=/home/ess/Documents/apps/ominari/projects/ominari
Environment="PATH=/home/ess/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /home/ess/Documents/apps/ominari/projects/ominari/results_catchup_service.py --mode continuous --interval 30
Restart=always
RestartSec=60
StandardOutput=append:/home/ess/Documents/apps/ominari/projects/ominari/logs/catchup.log
StandardError=append:/home/ess/Documents/apps/ominari/projects/ominari/logs/catchup_error.log

[Install]
WantedBy=multi-user.target
"""
    
    with open('ominari-catchup.service', 'w') as f:
        f.write(service_content)
    
    print("\n📝 Created systemd service file: ominari-catchup.service")
    print("To install:")
    print("  sudo cp ominari-catchup.service /etc/systemd/system/")
    print("  sudo systemctl daemon-reload")
    print("  sudo systemctl enable ominari-catchup.service")
    print("  sudo systemctl start ominari-catchup.service")

if __name__ == "__main__":
    sys.exit(main())