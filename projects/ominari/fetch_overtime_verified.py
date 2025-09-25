#!/usr/bin/env python3
"""
Fetch real Overtime markets using VERIFIED contract addresses from the user
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VERIFIED CONTRACT ADDRESSES FROM USER
VERIFIED_CONTRACTS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'explorer_api': 'https://api.arbiscan.io/api',
        'contracts': {
            'SportsAMM': '0x410AfcF3Abe7A72DD1B74942918d110Ef4a3eDB4',
            'SportPositionalMarketManager': '0x72ca0765d4bE0529377d656c9645600606214610',
            'SportPositionalMarketFactory': '0x85b827d133FEDC36B844b20f4a198dA583B25BAA',
            'SportPositionalMarketData': '0x503e7F2C19384Ff68B445E21850fDC61f34434e6'
        },
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'explorer_api': 'https://api-optimistic.etherscan.io/api',
        'redirector': 'https://contracts.overtime.io/',
        'name': 'Optimism'
    }
}

def get_implementation(explorer_api, address, api_key=''):
    """Get implementation address if this is a proxy."""
    params = {
        'module': 'contract',
        'action': 'getsourcecode',
        'address': address
    }
    if api_key:
        params['apikey'] = api_key
        
    try:
        response = requests.get(explorer_api, params=params, timeout=10)
        data = response.json()
        if data['status'] == '1' and data['result']:
            impl = data['result'][0].get('Implementation', '')
            return impl if impl else address
    except:
        pass
    return address

def get_abi(explorer_api, address, api_key=''):
    """Get ABI for a contract (handles proxies)."""
    # First get implementation if proxy
    impl = get_implementation(explorer_api, address, api_key)
    
    params = {
        'module': 'contract',
        'action': 'getabi',
        'address': impl
    }
    if api_key:
        params['apikey'] = api_key
        
    try:
        response = requests.get(explorer_api, params=params, timeout=10)
        data = response.json()
        if data['status'] == '1':
            return json.loads(data['result'])
    except:
        pass
    return None

def get_factory_creations(explorer_api, factory_address, api_key=''):
    """Get all contract creations from factory's internal transactions."""
    logger.info(f"🏭 Fetching factory creations from {factory_address}")
    
    params = {
        'module': 'account',
        'action': 'txlistinternal',
        'address': factory_address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 10000,
        'sort': 'desc'
    }
    if api_key:
        params['apikey'] = api_key
        
    markets = []
    
    try:
        response = requests.get(explorer_api, params=params, timeout=30)
        data = response.json()
        
        if data['status'] == '1' and data['result']:
            for tx in data['result']:
                # Look for contract creations
                if tx.get('type') == 'create' or (tx.get('contractAddress') and tx.get('isError') == '0'):
                    contract_addr = tx.get('contractAddress')
                    if contract_addr:
                        markets.append(Web3.to_checksum_address(contract_addr))
                        
            logger.info(f"  Found {len(markets)} market creations")
            
    except Exception as e:
        logger.error(f"Error fetching factory creations: {e}")
        
    return markets

def get_market_details(w3, market_address, market_abi):
    """Get details from a SportPositionalMarket clone."""
    try:
        contract = w3.eth.contract(address=market_address, abi=market_abi)
        
        details = {'address': market_address}
        
        # Get game details
        try:
            game_id, game_label = contract.functions.getGameDetails().call()
            details['gameId'] = game_id
            details['gameLabel'] = game_label
            
            # Parse teams from label
            if ' vs ' in game_label:
                parts = game_label.split(' vs ')
                details['homeTeam'] = parts[0].strip()
                details['awayTeam'] = parts[1].strip()
            elif ' @ ' in game_label:
                parts = game_label.split(' @ ')
                details['awayTeam'] = parts[0].strip()
                details['homeTeam'] = parts[1].strip()
        except:
            # Try alternative methods
            try:
                details['homeTeam'] = contract.functions.homeTeam().call()
                details['awayTeam'] = contract.functions.awayTeam().call()
            except:
                return None
                
        # Get times
        try:
            times = contract.functions.times().call()
            details['maturity'] = times[0]
            details['expiry'] = times[1]
        except:
            return None
            
        # Get status
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        try:
            details['cancelled'] = contract.functions.cancelled().call()
        except:
            details['cancelled'] = False
            
        # Get tags (sport)
        try:
            tags = contract.functions.tags().call()
            if tags:
                details['sportId'] = tags[0]
        except:
            details['sportId'] = 9004  # Default to soccer
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def get_market_odds(w3, amm_address, amm_abi, market_address):
    """Get odds from SportsAMM for a market."""
    try:
        amm = w3.eth.contract(address=amm_address, abi=amm_abi)
        
        odds = {}
        
        # Get odds for each position (0=HOME, 1=AWAY, 2=DRAW)
        for position in range(3):
            try:
                odd = amm.functions.obtainOdds(market_address, position).call()
                if odd > 0:
                    # Convert from AMM format (e.g., 1e18 = 1.0) to decimal
                    decimal_odd = odd / 1e18
                    if decimal_odd > 1.0:  # Valid odd
                        odds[position] = decimal_odd
            except:
                pass
                
        # Also try getMarketDefaultOdds
        if not odds:
            try:
                default_odds = amm.functions.getMarketDefaultOdds(market_address).call()
                if isinstance(default_odds, (list, tuple)) and len(default_odds) >= 2:
                    if default_odds[0] > 0:
                        odds[0] = default_odds[0] / 1e18  # HOME
                    if default_odds[1] > 0:
                        odds[1] = default_odds[1] / 1e18  # AWAY
                    if len(default_odds) > 2 and default_odds[2] > 0:
                        odds[2] = default_odds[2] / 1e18  # DRAW
            except:
                pass
                
        return odds
        
    except Exception as e:
        logger.debug(f"Error getting odds: {e}")
        return {}

def process_arbitrum():
    """Process Arbitrum markets using verified contracts."""
    config = VERIFIED_CONTRACTS['arbitrum']
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {config['name']}")
            return 0
            
        logger.info(f"Connected to {config['name']}")
        
        # Get Factory ABI
        factory_addr = config['contracts']['SportPositionalMarketFactory']
        logger.info(f"\n📜 Fetching Factory ABI...")
        factory_abi = get_abi(config['explorer_api'], factory_addr)
        
        # Get market creations from factory
        market_addresses = get_factory_creations(config['explorer_api'], factory_addr)
        
        if not market_addresses:
            logger.warning("No markets found from factory")
            return 0
            
        # Get a sample market ABI (they all use the same mastercopy)
        logger.info(f"\n📜 Fetching Market ABI...")
        if market_addresses:
            # The mastercopy ABI is what we need
            market_abi = get_abi(config['explorer_api'], market_addresses[0])
            if not market_abi:
                logger.warning("Could not get market ABI")
                return 0
        else:
            return 0
            
        # Get SportsAMM ABI for odds
        amm_addr = config['contracts']['SportsAMM']
        logger.info(f"\n📜 Fetching SportsAMM ABI...")
        amm_abi = get_abi(config['explorer_api'], amm_addr)
        
        # Process recent markets
        logger.info(f"\n🏆 Processing {len(market_addresses)} markets...")
        for i, market_addr in enumerate(market_addresses[:50]):  # Limit to 50 most recent
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{min(50, len(market_addresses))}")
                
            try:
                details = get_market_details(w3, market_addr, market_abi)
                
                if not details or not details.get('homeTeam'):
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                # Check maturity
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"arbitrum_verified_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Map sport ID
                    sport_map = {
                        9001: "American Football",
                        9002: "Baseball",
                        9003: "Basketball",
                        9004: "Soccer",
                        9005: "Hockey",
                        9006: "MMA",
                        9007: "Boxing",
                        9008: "Tennis"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source="arbitrum_verified",
                        sport=sport_map.get(details.get('sportId', 9004), 'Soccer'),
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=details['homeTeam'],
                        away_team=details['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.flush()
                    
                    # Get odds if AMM ABI available
                    if amm_abi:
                        odds = get_market_odds(w3, amm_addr, amm_abi, market_addr)
                        
                        for position, decimal_odds in odds.items():
                            outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                            outcome = outcome_map.get(position)
                            if outcome:
                                # Convert decimal to American
                                if decimal_odds >= 2.0:
                                    american = int((decimal_odds - 1) * 100)
                                else:
                                    american = int(-100 / (decimal_odds - 1))
                                    
                                odd = Odd(
                                    source_id=market.source_id,
                                    market_type="winner",
                                    outcome=outcome,
                                    source="arbitrum_amm",
                                    bookmaker="Overtime",
                                    decimal_odds=decimal_odds,
                                    american_odds=american,
                                    normalized_implied=1.0 / decimal_odds,
                                    updated_at=datetime.now(timezone.utc)
                                )
                                db.add(odd)
                                
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"✅ Added: {details['homeTeam']} vs {details['awayTeam']}")
                    logger.info(f"   Game: {details.get('gameLabel', 'N/A')}")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Maturity: {maturity}")
                    if odds:
                        logger.info(f"   Odds: {odds}")
                        
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing Arbitrum: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Overtime Market Fetcher - VERIFIED CONTRACTS")
    logger.info("=" * 60)
    logger.info("Using verified contract addresses from blockchain explorers")
    
    total_added = 0
    
    # Process Arbitrum (has all verified addresses)
    added = process_arbitrum()
    total_added += added
    
    # Summary
    with db_manager.get_db_session() as db:
        # Clear sample data if we got real data
        if total_added > 0:
            logger.info("\n🧹 Clearing sample data...")
            sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
            for market in sample_markets:
                # Delete odds first
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ VERIFIED FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total_added > 0:
            logger.info("\n🎆 SUCCESS! Real Overtime markets from verified contracts!")
            logger.info("The dashboard at http://localhost:8888/unified now shows REAL blockchain data.")
            
            samples = db.query(Market).filter(Market.source == 'arbitrum_verified').limit(5).all()
            logger.info("\n📊 Sample verified markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")

if __name__ == "__main__":
    main()