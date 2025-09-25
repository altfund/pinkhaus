#!/usr/bin/env python3
"""Check if we need to revert sport changes - sports should only come from blockchain tags"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func

def main():
    with db_manager.get_db_session() as db:
        print("Current sport distribution in database:")
        print("=" * 60)
        
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        total = 0
        for sport, count in sport_counts:
            print(f"  {sport:20} {count:6d}")
            total += count
        print(f"  {'TOTAL':20} {total:6d}")
        
        print("\n" + "=" * 60)
        print("IMPORTANT: Sports should only come from blockchain contract tags!")
        print("The tag mapping is:")
        print("  9001: American Football")
        print("  9002: Baseball") 
        print("  9003: Basketball")
        print("  9004: Soccer")
        print("  9005: Hockey")
        print("  9006: MMA")
        print("  9007: Boxing")
        print("  9008: Tennis")
        print("  9010: Golf")
        print("  9011: Cricket")
        print("  9012: Rugby")
        print("  9014: Motorsport")
        print("\nIf a market doesn't have a tag, it defaults to Soccer.")
        print("\nThe current distribution may be incorrect if we inferred sports")
        print("from team names rather than using blockchain tags.")

if __name__ == "__main__":
    main()