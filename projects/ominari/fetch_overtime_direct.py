#!/usr/bin/env python3
"""
Direct fetcher for Overtime V2 markets - no API keys required
Uses raw transaction analysis and direct contract calls
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 Contracts we know have activity
V2_CONTRACTS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'SportsAMMV2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'Manager': '0xB155685132eEd3cD848d220e25a9607DD8871D38',
        'name': 'Arbitrum'
    }
}

# Known method selectors
SELECTORS = {
    'getGameDetails': '0x5ee0fe28',
    'times': '0xd0370218',
    'resolved': '0x5fe138b5',
    'tags': '0x4bbf5252',
    'homeTeam': '0x8de859d8',
    'awayTeam': '0x36c78516',
    'obtainOdds': '0x942b67dc',  # obtainOdds(address,uint8)
    'getMarketDefaultOdds': '0xb12df8e3'  # getMarketDefaultOdds(address)
}

def find_markets_in_recent_transactions(w3, amm_address, blocks_to_scan=1000):
    """Find market addresses from recent AMM transactions."""
    logger.info(f"🔍 Scanning last {blocks_to_scan} blocks for market activity...")
    
    markets = set()
    current_block = w3.eth.block_number
    found_transactions = 0
    
    # Scan blocks
    for i in range(blocks_to_scan):
        if i % 100 == 0 and i > 0:
            logger.info(f"  Progress: {i}/{blocks_to_scan} blocks scanned, found {len(markets)} markets")
            
        block_num = current_block - i
        
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block['transactions']:
                # Check if transaction is to our AMM
                if tx['to'] and tx['to'].lower() == amm_address.lower():
                    found_transactions += 1
                    input_data = tx['input']
                    
                    # Look for common patterns in transaction data
                    # Market address is usually first parameter after method selector
                    if len(input_data) >= 74:  # 0x + 8 chars method + 64 chars address
                        try:
                            # Extract potential address from first parameter
                            potential_addr = '0x' + input_data[34:74]
                            
                            # Verify it looks like an address
                            if len(potential_addr) == 42:
                                # Check if it's a contract
                                code = w3.eth.get_code(potential_addr)
                                if code and len(code) > 0:
                                    markets.add(Web3.to_checksum_address(potential_addr))
                                    logger.debug(f"Found potential market: {potential_addr}")
                        except:
                            pass
                            
        except Exception as e:
            logger.debug(f"Error processing block {block_num}: {e}")
            continue
            
    logger.info(f"✅ Scanned {blocks_to_scan} blocks, found {found_transactions} AMM transactions")
    logger.info(f"🎯 Identified {len(markets)} unique market addresses")
    
    return list(markets)

def raw_call(w3, contract_address, selector):
    """Make a raw call to a contract."""
    try:
        result = w3.eth.call({
            'to': contract_address,
            'data': selector
        })
        return result
    except:
        return None

def decode_string(data):
    """Decode a string from contract return data."""
    if not data or len(data) < 64:
        return None
        
    try:
        # Skip offset (32 bytes) and read length (32 bytes)
        offset = int.from_bytes(data[0:32], 'big')
        length = int.from_bytes(data[offset:offset+32], 'big')
        
        if length > 0 and length < 1000:  # Sanity check
            string_data = data[offset+32:offset+32+length].decode('utf-8', errors='ignore').strip()
            return string_data
    except:
        # Try alternative decoding (no offset)
        try:
            length = int.from_bytes(data[32:64], 'big')
            if length > 0 and length < 1000:
                string_data = data[64:64+length].decode('utf-8', errors='ignore').strip()
                return string_data
        except:
            pass
            
    return None

def get_market_info_raw(w3, market_address):
    """Get market information using raw calls."""
    info = {}
    
    # Try getGameDetails
    game_details = raw_call(w3, market_address, SELECTORS['getGameDetails'])
    if game_details and len(game_details) > 96:
        # First 32 bytes is game ID
        # Then comes the game label string
        game_label = decode_string(game_details[32:])
        if game_label:
            info['gameLabel'] = game_label
            # Parse teams
            if ' vs ' in game_label:
                parts = game_label.split(' vs ')
                info['homeTeam'] = parts[0].strip()
                info['awayTeam'] = parts[1].strip()
            elif ' @ ' in game_label:
                parts = game_label.split(' @ ')
                info['awayTeam'] = parts[0].strip()
                info['homeTeam'] = parts[1].strip()
    
    # If no game details, try individual team calls
    if 'homeTeam' not in info:
        home_data = raw_call(w3, market_address, SELECTORS['homeTeam'])
        if home_data:
            home = decode_string(home_data)
            if home:
                info['homeTeam'] = home
                
        away_data = raw_call(w3, market_address, SELECTORS['awayTeam'])
        if away_data:
            away = decode_string(away_data)
            if away:
                info['awayTeam'] = away
    
    # Get times
    times_data = raw_call(w3, market_address, SELECTORS['times'])
    if times_data and len(times_data) >= 64:
        maturity = int.from_bytes(times_data[0:32], 'big')
        info['maturity'] = maturity
    
    # Get resolved status
    resolved_data = raw_call(w3, market_address, SELECTORS['resolved'])
    if resolved_data:
        info['resolved'] = resolved_data[-1] == 1
    else:
        info['resolved'] = False
    
    # Get sport tags
    tags_data = raw_call(w3, market_address, SELECTORS['tags'])
    if tags_data and len(tags_data) > 96:
        # Array data: offset, length, then elements
        try:
            sport_id = int.from_bytes(tags_data[96:128], 'big')
            info['sportId'] = sport_id
        except:
            info['sportId'] = 9004  # Default to soccer
    
    return info

def get_odds_from_amm(w3, amm_address, market_address):
    """Try to get odds from AMM using raw calls."""
    odds = {}
    
    # Try getMarketDefaultOdds first
    # Encode market address as parameter
    encoded_market = market_address[2:].lower().rjust(64, '0')
    call_data = SELECTORS['getMarketDefaultOdds'] + encoded_market
    
    try:
        result = w3.eth.call({
            'to': amm_address,
            'data': call_data
        })
        
        if result and len(result) >= 96:
            # Result is an array of odds
            offset = int.from_bytes(result[0:32], 'big')
            length = int.from_bytes(result[offset:offset+32], 'big')
            
            if length >= 2:
                # Read odds values
                home_odds = int.from_bytes(result[offset+64:offset+96], 'big')
                away_odds = int.from_bytes(result[offset+96:offset+128], 'big')
                
                if home_odds > 0:
                    odds[0] = home_odds / 1e18
                if away_odds > 0:
                    odds[1] = away_odds / 1e18
                    
                if length >= 3:
                    draw_odds = int.from_bytes(result[offset+128:offset+160], 'big')
                    if draw_odds > 0:
                        odds[2] = draw_odds / 1e18
    except Exception as e:
        logger.debug(f"Error getting odds: {e}")
    
    return odds

def main():
    """Main function."""
    logger.info("🎯 Overtime V2 Direct Fetcher - No API Keys Required")
    logger.info("=" * 60)
    
    config = V2_CONTRACTS['arbitrum']
    
    try:
        # Connect to Arbitrum
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error("Failed to connect to Arbitrum")
            return
            
        logger.info(f"✅ Connected to {config['name']} at block {w3.eth.block_number:,}")
        
        # Find markets from recent transactions
        market_addresses = find_markets_in_recent_transactions(w3, config['SportsAMMV2'])
        
        if not market_addresses:
            logger.warning("No markets found in recent transactions")
            logger.info("\n💡 Possible reasons:")
            logger.info("  1. Markets may be seasonal (no active sports events)")
            logger.info("  2. V2 activity might be on different methods")
            logger.info("  3. Markets might be created differently in V2")
            return
        
        # Process each market
        markets_added = 0
        logger.info(f"\n📊 Processing {len(market_addresses)} potential markets...")
        
        for i, market_addr in enumerate(market_addresses):
            if i % 5 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{len(market_addresses)}")
                
            try:
                # Get market info using raw calls
                info = get_market_info_raw(w3, market_addr)
                
                if not info.get('homeTeam') or not info.get('awayTeam'):
                    logger.debug(f"  Skipping {market_addr} - no team data")
                    continue
                    
                if info.get('resolved'):
                    logger.debug(f"  Skipping {market_addr} - already resolved")
                    continue
                    
                if not info.get('maturity'):
                    logger.debug(f"  Skipping {market_addr} - no maturity date")
                    continue
                    
                # Check if future game
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    logger.debug(f"  Skipping {market_addr} - past game")
                    continue
                    
                # This is a valid future market!
                market_id = f"arbitrum_v2_direct_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    # Check if already exists
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        logger.info(f"  Market already in database: {info['homeTeam']} vs {info['awayTeam']}")
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
                    
                    # Create market
                    market = Market(
                        source_id=market_id,
                        source="arbitrum_v2_direct",
                        sport=sport_map.get(info.get('sportId', 9004), 'Soccer'),
                        league_name="Overtime Markets V2",
                        market_type="winner",
                        home_team=info['homeTeam'],
                        away_team=info['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.flush()
                    
                    # Try to get odds
                    odds = get_odds_from_amm(w3, config['SportsAMMV2'], market_addr)
                    
                    for position, decimal_odds in odds.items():
                        outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                        outcome = outcome_map.get(position)
                        
                        if outcome and decimal_odds > 1.0:
                            # Convert to American
                            if decimal_odds >= 2.0:
                                american = int((decimal_odds - 1) * 100)
                            else:
                                american = int(-100 / (decimal_odds - 1))
                                
                            odd = Odd(
                                source_id=market.source_id,
                                market_type="winner",
                                outcome=outcome,
                                source="arbitrum_v2_amm",
                                bookmaker="Overtime V2",
                                decimal_odds=decimal_odds,
                                american_odds=american,
                                normalized_implied=1.0 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                    
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"✅ Added: {info['homeTeam']} vs {info['awayTeam']}")
                    logger.info(f"   Sport: {sport_map.get(info.get('sportId', 9004), 'Soccer')}")
                    logger.info(f"   Maturity: {maturity}")
                    logger.info(f"   Contract: {market_addr}")
                    if odds:
                        logger.info(f"   Odds: Home={odds.get(0, 'N/A')}, Away={odds.get(1, 'N/A')}, Draw={odds.get(2, 'N/A')}")
                        
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
        
        # Summary
        with db_manager.get_db_session() as db:
            # If we found real markets, clear sample data
            if markets_added > 0:
                logger.info("\n🧹 Clearing sample data...")
                sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
                for market in sample_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
                logger.info(f"Removed {len(sample_markets)} sample markets")
            
            total = db.query(Market).count()
            active = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).count()
            
            logger.info(f"\n✨ DIRECT FETCH COMPLETE ✨")
            logger.info(f"Total markets in database: {total}")
            logger.info(f"Active future markets: {active}")
            logger.info(f"Markets added this run: {markets_added}")
            
            if markets_added > 0:
                logger.info("\n🎆 SUCCESS! Found real Overtime V2 markets!")
                logger.info("The dashboard at http://localhost:8888/unified now shows REAL blockchain data!")
            else:
                logger.info("\n⚠️ No new markets found. This could mean:")
                logger.info("  1. No active sports events currently")
                logger.info("  2. Markets are created differently in V2")
                logger.info("  3. Need to check different contract methods")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()