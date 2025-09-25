#!/usr/bin/env python3
"""
Fetch Overtime markets using the correct approach from user's guide:
1. Resolve current addresses via redirector
2. Use SportsAMM from the redirector (not hardcoded)
3. Scan for real market activity
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAINS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'redirector_path': 'mainnet-ovm',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'redirector_path': 'mainnet-arbitrum',
        'name': 'Arbitrum'
    }
}

def resolve_address(redirector_path, contract_name):
    """
    Resolve current contract address from Overtime's redirector.
    """
    url = f"https://contracts.overtime.io/{redirector_path}/{contract_name}"
    
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        final_url = response.url
        
        # Extract address from final URL
        match = re.search(r'0x[a-fA-F0-9]{40}', final_url)
        if match:
            address = Web3.to_checksum_address(match.group(0))
            logger.info(f"✅ {contract_name}: {address}")
            return address
        else:
            logger.error(f"Could not parse address from: {final_url}")
            return None
            
    except Exception as e:
        logger.error(f"Error resolving {contract_name}: {e}")
        return None

def find_markets_from_transactions(w3, amm_address):
    """
    Find markets from recent AMM transactions.
    """
    logger.info("🔍 Scanning recent AMM transactions...")
    
    markets = set()
    current_block = w3.eth.block_number
    
    # Common AMM method signatures
    method_sigs = [
        '0x942b67dc',  # buyFromAMM
        '0x8cc7bcc5',  # sellToAMM
        '0x1ed66ad7',  # buyFromAMMWithDifferentCollateral
        '0x4917dc72'   # sellToAMMWithDifferentCollateral
    ]
    
    # Check last 500 blocks in smaller chunks
    blocks_to_scan = 500
    chunk_size = 50
    
    for chunk_start in range(0, blocks_to_scan, chunk_size):
        from_block = current_block - blocks_to_scan + chunk_start
        to_block = min(from_block + chunk_size - 1, current_block)
        
        try:
            block_range = w3.eth.filter({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': amm_address
            })
            
            logs = block_range.get_all_entries()
            
            # Also check transactions
            for block_num in range(from_block, to_block + 1):
                try:
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block['transactions']:
                        if tx['to'] and tx['to'].lower() == amm_address.lower():
                            input_data = tx['input']
                            
                            # Check if it's an AMM method
                            for sig in method_sigs:
                                if input_data.startswith(sig):
                                    # Extract market address (first parameter)
                                    try:
                                        data = input_data[10:]  # Skip method signature
                                        if len(data) >= 64:
                                            addr_hex = '0x' + data[24:64]
                                            market_addr = Web3.to_checksum_address(addr_hex)
                                            
                                            # Verify it's a contract
                                            code = w3.eth.get_code(market_addr)
                                            if code:
                                                markets.add(market_addr)
                                                logger.debug(f"Found market: {market_addr}")
                                    except:
                                        pass
                                        
                except Exception as e:
                    continue
                    
            if len(markets) >= 20:  # Found enough
                break
                
        except Exception as e:
            logger.debug(f"Error scanning chunk: {e}")
            
        logger.info(f"  Scanned blocks {from_block}-{to_block}, found {len(markets)} markets so far")
        
    return list(markets)

def decode_proxy_string(w3, proxy_address, selector):
    """
    Decode a string return value from a proxy contract.
    """
    try:
        result = w3.eth.call({
            'to': proxy_address,
            'data': selector
        })
        
        if result and len(result) > 64:
            # Skip offset (32 bytes) and read length (32 bytes)
            string_length = int.from_bytes(result[32:64], 'big')
            if string_length > 0 and string_length < 1000:  # Sanity check
                string_data = result[64:64+string_length].decode('utf-8', errors='ignore').strip()
                return string_data
    except:
        pass
        
    return None

def get_market_info(w3, market_address):
    """
    Get market information handling proxy contracts.
    """
    try:
        # Check contract code size
        code = w3.eth.get_code(market_address)
        logger.debug(f"  Contract size: {len(code)} bytes")
        
        details = {}
        
        # Method selectors
        selectors = {
            'homeTeam': '0x8de859d8',
            'awayTeam': '0x36c78516',
            'times': '0xd0370218',
            'resolved': '0x5fe138b5',
            'tags': '0x4bbf5252'
        }
        
        # Get teams
        home = decode_proxy_string(w3, market_address, selectors['homeTeam'])
        away = decode_proxy_string(w3, market_address, selectors['awayTeam'])
        
        if not home or not away:
            # Try getGameDetails as fallback
            game_details_selector = '0x5ee0fe28'
            try:
                result = w3.eth.call({
                    'to': market_address,
                    'data': game_details_selector
                })
                
                if result and len(result) > 96:
                    # Skip game ID (32 bytes) and offset (32 bytes)
                    string_length = int.from_bytes(result[64:96], 'big')
                    if string_length > 0:
                        game_label = result[96:96+string_length].decode('utf-8', errors='ignore')
                        if ' vs ' in game_label:
                            parts = game_label.split(' vs ')
                            home = parts[0].strip()
                            away = parts[1].strip()
            except:
                pass
                
        if not home or not away:
            return None
            
        details['homeTeam'] = home
        details['awayTeam'] = away
        
        # Get times
        try:
            result = w3.eth.call({
                'to': market_address,
                'data': selectors['times']
            })
            
            if result and len(result) >= 64:
                maturity = int.from_bytes(result[0:32], 'big')
                details['maturity'] = maturity
        except:
            pass
            
        # Get resolved status
        try:
            result = w3.eth.call({
                'to': market_address,
                'data': selectors['resolved']
            })
            
            if result:
                details['resolved'] = result[-1] == 1
            else:
                details['resolved'] = False
        except:
            details['resolved'] = False
            
        # Get sport tags
        try:
            result = w3.eth.call({
                'to': market_address,
                'data': selectors['tags']
            })
            
            if result and len(result) > 96:
                # First tag is at position 96 (after offset and length)
                sport_id = int.from_bytes(result[96:128], 'big')
                details['sportId'] = sport_id
        except:
            details['sportId'] = 9004  # Default to soccer
            
        return details
        
    except Exception as e:
        logger.debug(f"Error getting market info: {e}")
        return None

def process_chain(chain):
    """
    Process markets from a specific chain.
    """
    config = CHAINS[chain]
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        # 1. Resolve SportsAMM address
        amm_address = resolve_address(config['redirector_path'], 'SportsAMM')
        if not amm_address:
            logger.error("Failed to resolve SportsAMM address")
            return 0
            
        # Connect to chain
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {chain}")
            return 0
            
        logger.info(f"Connected at block {w3.eth.block_number:,}")
        
        # 2. Find markets from transactions
        market_addresses = find_markets_from_transactions(w3, amm_address)
        
        if not market_addresses:
            logger.warning("No markets found in recent transactions")
            return 0
            
        logger.info(f"\n🎯 Found {len(market_addresses)} potential markets")
        
        # 3. Process each market
        for i, market_addr in enumerate(market_addresses):
            logger.info(f"\nProcessing market {i+1}/{len(market_addresses)}: {market_addr}")
            
            try:
                info = get_market_info(w3, market_addr)
                
                if not info:
                    logger.warning("  Could not get market info")
                    continue
                    
                if info.get('resolved'):
                    logger.info("  Market resolved, skipping")
                    continue
                    
                if not info.get('maturity'):
                    logger.warning("  No maturity date")
                    continue
                    
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    logger.info("  Past game, skipping")
                    continue
                    
                market_id = f"blockchain_{chain}_real_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        logger.info("  Already in database")
                        continue
                        
                    # Map sport ID
                    sport_map = {
                        9001: "American Football",
                        9002: "Baseball",
                        9003: "Basketball",
                        9004: "Soccer",
                        9005: "Hockey",
                        9006: "MMA",
                        9008: "Tennis"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{chain}_real",
                        sport=sport_map.get(info.get('sportId', 9004), 'Soccer'),
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=info['homeTeam'],
                        away_team=info['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"  ✅ Added: {info['homeTeam']} vs {info['awayTeam']}")
                    logger.info(f"     Sport: {sport_map.get(info.get('sportId', 9004), 'Soccer')}")
                    logger.info(f"     Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing {chain}: {e}")
        
    return markets_added

def main():
    """
    Main function.
    """
    logger.info("🎯 Overtime Market Fetcher - Correct Approach")
    logger.info("=" * 60)
    logger.info("Using redirector to find current contracts")
    
    total_added = 0
    
    # Process each chain
    for chain in ['optimism', 'arbitrum']:
        added = process_chain(chain)
        total_added += added
        time.sleep(2)
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total_added > 0:
            logger.info("\n🎆 SUCCESS! Real Overtime markets have been added to the database!")
            logger.info("The dashboard at http://localhost:8888/unified should now show real blockchain data.")

if __name__ == "__main__":
    main()