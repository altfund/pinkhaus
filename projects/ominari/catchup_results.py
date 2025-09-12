#!/usr/bin/env python3
"""Catch up on recent match results - run via cron."""

from backfill_historical_data import HistoricalDataBackfiller
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

# Check matches finished in last 24 hours
backfiller = HistoricalDataBackfiller()

# Get recently finished matches
from database_v2 import db_manager
from models import Market

with db_manager.get_db_session() as db:
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_finished = db.query(Market).filter(
        Market.is_finished == True,
        Market.last_update >= recent_cutoff
    ).all()
    
    print(f"Found {len(recent_finished)} recently finished matches")
    
    # Enrich any paper trading positions for these markets
    for market in recent_finished:
        backfiller.stats['markets_checked'] += 1
        # Process positions for this market
        
print("Catch-up complete:", backfiller.stats)
