#!/usr/bin/env python3
"""Fix the last 60 unknown markets"""

from database_v2 import db_manager
from models import Market
from sqlalchemy import func

def main():
    fixes = {
        # Soccer clubs with clear indicators
        "Thor Akureyri vs Valur": "Soccer",  # Icelandic clubs
        "Rodez Aveyron Football vs Clermont Foot 63": "Soccer",  # French clubs
        "GKS Katowice vs MKS Cracovia": "Soccer",  # Polish clubs
        "AS Saint-Étienne vs Stade de Reims": "Soccer",  # French clubs
        "KVC Westerlo vs Royal Standard de Liège": "Soccer",  # Belgian clubs
        "GD Peniche vs 1o Dezembro": "Soccer",  # Portuguese clubs
        "Kristiansund BK vs Haugesund": "Soccer",  # Norwegian clubs
        "LR Vicenza Virtus vs Aurora Pro Patria 1919": "Soccer",  # Italian clubs
        "Mushuc Runa vs CSD Independiente del Valle": "Soccer",  # Ecuador clubs
        "En Avant Guingamp vs US du Littoral de Dunkerque": "Soccer",  # French clubs
        "FK Novi Pazar vs FK Radnički 1923 Kragujevac": "Soccer",  # Serbian clubs
        "AS Nancy-Lorraine vs Stade de Reims": "Soccer",  # French clubs
        "Montevideo Wanderers vs Cerro": "Soccer",  # Uruguayan clubs
        
        # eSports teams
        "ZETA DIVISION vs T1": "eSports",
        "Rising Bees vs Eterna": "eSports",
        
        # College Sports - US colleges
        "Carnegie Mellon vs Berry": "College Sports",
        "Trinity Bantams vs Bates": "College Sports",
        "Robert Morris vs Dayton": "College Sports",
        "Illinois Wesleyan vs Elmhurst": "College Sports",
        "South Dakota vs Drake": "College Sports",
        "The Citadel vs Mercer": "College Sports",
        "Ole Miss vs Tulane": "College Sports",
        "North Dakota vs Valparaiso": "College Sports",
        "Case Western Reserve vs Westminster": "College Sports",
        "Middle Tennessee vs Marshall": "College Sports",
        
        # Handball teams
        "Mrk Dugo Selo vs Porec": "Handball",
        "MRK Trogir vs Rudar": "Handball",
        "Csurgoi KK vs Neka": "Handball",
        
        # Volleyball
        "Raision Loimu vs Tiikerit": "Volleyball",  # Finnish volleyball
        
        # Tennis players (individual names)
        "Rafael Tobias vs Jair de Oliveira": "Tennis",
        "Juan Pablo Varillas vs Santiago De la Fuente": "Tennis",
        
        # Special markets
        "Heisman Trophy vs Winner": "American Football",  # College football award
    }
    
    with db_manager.get_db_session() as db:
        print("Fixing last unknown markets with manual mapping...")
        
        fixed_count = 0
        
        for match_text, sport in fixes.items():
            # Parse the match text
            parts = match_text.split(" vs ")
            if len(parts) == 2:
                home_team = parts[0]
                away_team = parts[1]
                
                # Find and update the market
                markets = db.query(Market).filter(
                    Market.home_team == home_team,
                    Market.away_team == away_team,
                    Market.sport == 'Unknown'
                ).all()
                
                for market in markets:
                    print(f"Fixing: {market.home_team} vs {market.away_team} -> {sport}")
                    market.sport = sport
                    fixed_count += 1
        
        db.commit()
        
        print(f"\n✅ Fixed {fixed_count} markets")
        
        # Show final counts
        print("\nFinal sport distribution:")
        sport_counts = db.query(
            Market.sport,
            func.count().label('count')
        ).group_by(Market.sport).order_by(func.count().desc()).all()
        
        total = 0
        unknown_count = 0
        for sport, count in sport_counts:
            print(f"  {sport:20} {count:6d}")
            total += count
            if sport == 'Unknown':
                unknown_count = count
        print(f"  {'TOTAL':20} {total:6d}")
        
        if unknown_count > 0:
            print(f"\n⚠️  Still have {unknown_count} Unknown markets")
            # Show what's left
            remaining = db.query(Market).filter(Market.sport == 'Unknown').limit(10).all()
            if remaining:
                print("\nFirst 10 remaining Unknown markets:")
                for market in remaining:
                    print(f"  - {market.home_team} vs {market.away_team}")

if __name__ == "__main__":
    main()