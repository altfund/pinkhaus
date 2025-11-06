#!/usr/bin/env python3
"""Fix match times for our real soccer matches"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone, timedelta

with db_manager.get_db_session() as db:
    # Update our specific matches by source_id
    updates = [
        ('soccer_2024_epl_001', 3),   # Man United vs Liverpool in 3 hours
        ('soccer_2024_epl_002', 6),   # Arsenal vs Chelsea in 6 hours
        ('soccer_2024_laliga_001', 12), # Real Madrid vs Barcelona in 12 hours
        ('soccer_2024_bundesliga_001', 24), # Bayern vs Dortmund in 24 hours
        ('soccer_2024_seriea_001', 36)  # Juventus vs Milan in 36 hours
    ]
    
    for source_id, hours in updates:
        market = db.query(Market).filter(Market.source_id == source_id).first()
        if market:
            market.maturity_date = datetime.now(timezone.utc) + timedelta(hours=hours)
            print(f"Updated {market.home_team} vs {market.away_team} to kick off in {hours} hours")
    
    db.commit()
    print("\n✅ All match times updated successfully")