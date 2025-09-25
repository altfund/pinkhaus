#!/usr/bin/env python3
"""
Fetch real Overtime markets from their subgraph
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Overtime subgraph endpoints
SUBGRAPH_URLS = {
    'optimism': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism',
    'arbitrum': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum'
}

# GraphQL query for active markets
MARKETS_QUERY = """
query GetActiveMarkets($minMaturity: BigInt!) {
  sportMarkets(
    first: 100
    where: {
      maturityDate_gt: $minMaturity
      isCanceled: false
      isResolved: false
    }
    orderBy: maturityDate
    orderDirection: asc
  ) {
    id
    address
    gameId
    maturityDate
    homeTeam
    awayTeam
    homeOdds
    awayOdds
    drawOdds
    sport
    league
    isResolved
    isCanceled
    timestamp
    betType
  }
}
"""

# Alternative query
POSITIONS_QUERY = """
query GetMarketPositions {
  marketPositions(
    first: 50
    where: {
      market_: {
        isResolved: false
        isCanceled: false
      }
    }
  ) {
    id
    market {
      id
      address
      gameId
      maturityDate
      homeTeam
      awayTeam
      sport
      league
    }
  }
}
"""

def fetch_from_subgraph(network):
    """Fetch markets from Overtime subgraph."""
    url = SUBGRAPH_URLS.get(network)
    if not url:
        logger.error(f"No subgraph URL for {network}")
        return 0
        
    logger.info(f"🔍 Querying {network} subgraph...")
    
    markets_added = 0
    
    try:
        # Current timestamp
        min_maturity = str(int(time.time()))
        
        # Execute query
        response = requests.post(
            url,
            json={
                'query': MARKETS_QUERY,
                'variables': {'minMaturity': min_maturity}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                # Try alternative query
                return fetch_alternative(url, network)
                
            markets = data.get('data', {}).get('sportMarkets', [])
            logger.info(f"Found {len(markets)} active markets")
            
            for market_data in markets:
                try:
                    # Extract data
                    market_address = market_data['address']
                    home_team = market_data['homeTeam']
                    away_team = market_data['awayTeam']
                    
                    if not home_team or not away_team:
                        continue
                        
                    market_id = f"blockchain_{network}_subgraph_{market_address}"
                    
                    # Check if exists
                    with db_manager.get_db_session() as db:
                        if db.query(Market).filter(Market.source_id == market_id).first():
                            continue
                            
                        # Create market
                        maturity = datetime.fromtimestamp(
                            int(market_data['maturityDate']),
                            tz=timezone.utc
                        )
                        
                        market = Market(
                            source_id=market_id,
                            source=f"blockchain_{network}_subgraph",
                            sport=market_data.get('sport', 'Soccer'),
                            league_name=market_data.get('league', 'Unknown League'),
                            market_type=market_data.get('betType', 'winner').lower(),
                            home_team=home_team,
                            away_team=away_team,
                            maturity_date=maturity,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        db.commit()
                        
                        # Add odds if available
                        if market_data.get('homeOdds'):
                            for outcome, odds_value in [
                                ('Home', market_data.get('homeOdds')),
                                ('Away', market_data.get('awayOdds')),
                                ('Draw', market_data.get('drawOdds'))
                            ]:
                                if odds_value and float(odds_value) > 0:
                                    odd = Odd(
                                        source_id=market_id,
                                        outcome=outcome,
                                        decimal_odds=float(odds_value),
                                        market_type='moneyline',
                                        source=f"blockchain_{network}_subgraph",
                                        bookmaker='overtime',
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(odd)
                                    
                            db.commit()
                            
                        markets_added += 1
                        logger.info(f"✅ Added: {home_team} vs {away_team} ({market_data.get('sport')})")
                        logger.info(f"   League: {market_data.get('league')}")
                        logger.info(f"   Date: {maturity}")
                        logger.info(f"   Address: {market_address}")
                        
                except Exception as e:
                    logger.error(f"Error processing market: {e}")
                    continue
                    
        else:
            logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        logger.error(f"Error querying subgraph: {e}")
        
    return markets_added

def fetch_alternative(url, network):
    """Try alternative queries."""
    logger.info("Trying alternative query...")
    
    # Try a simpler query
    simple_query = """
    {
      sportMarkets(first: 10) {
        id
        homeTeam
        awayTeam
        maturityDate
        isResolved
      }
    }
    """
    
    try:
        response = requests.post(url, json={'query': simple_query}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Alternative response: {data}")
            
            # Check if subgraph exists but has different schema
            if 'data' in data and data['data']:
                logger.info("Subgraph accessible but schema may be different")
            
    except Exception as e:
        logger.error(f"Alternative query failed: {e}")
        
    return 0

def main():
    """Main function."""
    logger.info("📊 Overtime Subgraph Fetcher")
    logger.info("=" * 60)
    
    total_added = 0
    
    # Try each network
    for network in ['optimism', 'arbitrum']:
        added = fetch_from_subgraph(network)
        total_added += added
        logger.info(f"Added {added} markets from {network}")
        time.sleep(2)
    
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ SUBGRAPH FETCH COMPLETE ✨")
        logger.info(f"Total markets: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"New markets added: {total_added}")
        
        # Show samples
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.league_name} - {m.maturity_date}")

if __name__ == "__main__":
    main()