#!/usr/bin/env python3
"""
Deep analysis of V2 transactions to find market patterns
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

def analyze_transaction_data(w3, tx_hash):
    """Analyze a transaction to extract market information."""
    try:
        tx = w3.eth.get_transaction(tx_hash)
        
        # Decode the input data
        input_data = tx['input'].hex() if hasattr(tx['input'], 'hex') else tx['input']
        
        # Method signature is first 10 chars
        method_sig = input_data[:10] if len(input_data) >= 10 else None
        
        # Look for addresses in the input data
        addresses = []
        
        # Each parameter is 32 bytes (64 hex chars)
        # Skip method signature (10 chars with 0x)
        data = input_data[10:]
        
        # Extract potential addresses (every 64 chars could be a parameter)
        for i in range(0, len(data), 64):
            param = data[i:i+64]
            if len(param) == 64:
                # Address would be in the last 40 chars
                potential_addr = '0x' + param[-40:]
                try:
                    # Validate it's a valid address
                    addr = Web3.to_checksum_address(potential_addr)
                    if addr != '0x' + '0' * 40:  # Not zero address
                        addresses.append(addr)
                except:
                    pass
                    
        return {
            'method': method_sig,
            'addresses': addresses,
            'from': tx['from'],
            'value': tx['value']
        }
        
    except Exception as e:
        logger.error(f"Error analyzing transaction: {e}")
        return None

def get_market_info_raw(w3, market_address):
    """Get basic market info using raw calls."""
    info = {'address': market_address}
    
    # getGameDetails
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5ee0fe28'
        })
        
        if result and len(result) > 96:
            try:
                offset = int.from_bytes(result[32:64], 'big')
                length = int.from_bytes(result[offset:offset+32], 'big')
                if length > 0 and length < 1000:
                    game_label = result[offset+32:offset+32+length].decode('utf-8', errors='ignore').strip()
                    if game_label:
                        info['gameLabel'] = game_label
                        if ' vs ' in game_label:
                            parts = game_label.split(' vs ')
                            info['homeTeam'] = parts[0].strip()
                            info['awayTeam'] = parts[1].strip()
                        elif ' @ ' in game_label:
                            parts = game_label.split(' @ ')
                            info['awayTeam'] = parts[0].strip()
                            info['homeTeam'] = parts[1].strip()
            except:
                pass
    except:
        pass
        
    # times
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0xd0370218'
        })
        if result and len(result) >= 32:
            maturity = int.from_bytes(result[:32], 'big')
            if maturity > 0:
                info['maturity'] = maturity
    except:
        pass
        
    # resolved
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5fe138b5'
        })
        if result:
            info['resolved'] = result[-1] == 1
    except:
        pass
        
    return info

def main():
    """Main analysis function."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    amm_address = Web3.to_checksum_address('0xfb64E79A562F7250131cf528242CEB10fDC82395')
    
    logger.info("🔍 Deep Transaction Analysis for V2 Markets")
    logger.info("=" * 60)
    
    current_block = w3.eth.block_number
    markets_found = set()
    
    # Get recent transactions to the AMM
    logger.info("\n📡 Scanning recent AMM transactions...")
    
    tx_count = 0
    blocks_to_scan = 1000
    
    for i in range(blocks_to_scan):
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{blocks_to_scan} blocks")
            
        block_num = current_block - i
        
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block['transactions']:
                if tx['to'] and tx['to'].lower() == amm_address.lower():
                    tx_count += 1
                    
                    # Analyze the transaction
                    analysis = analyze_transaction_data(w3, tx['hash'])
                    
                    if analysis and analysis['addresses']:
                        # Check each address to see if it's a market
                        for addr in analysis['addresses']:
                            try:
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:  # Minimal proxy!
                                    markets_found.add(addr)
                                    logger.info(f"\n🎯 Found market via transaction: {addr}")
                                    logger.info(f"  Method: {analysis['method']}")
                                    logger.info(f"  Transaction: {tx['hash'].hex()}")
                                    
                                    # Get market info
                                    info = get_market_info_raw(w3, addr)
                                    if info.get('gameLabel'):
                                        logger.info(f"  Game: {info['gameLabel']}")
                                        
                                    # Stop after finding a few to test
                                    if len(markets_found) >= 5:
                                        break
                            except:
                                pass
                                
                    if len(markets_found) >= 5:
                        break
                        
        except Exception as e:
            continue
            
        if len(markets_found) >= 5:
            break
            
    logger.info(f"\n📊 Transaction Analysis Summary:")
    logger.info(f"Total AMM transactions analyzed: {tx_count}")
    logger.info(f"Markets found: {len(markets_found)}")
    
    # Process found markets
    if markets_found:
        logger.info(f"\n💾 Processing {len(markets_found)} markets...")
        
        markets_added = 0
        
        for market_addr in markets_found:
            try:
                info = get_market_info_raw(w3, market_addr)
                
                if not info.get('homeTeam') or not info.get('maturity'):
                    continue
                    
                if info.get('resolved', False):
                    continue
                    
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"arbitrum_v2_tx_{market_addr.lower()[-8:]}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source="arbitrum_v2_transactions",
                        sport="Soccer",
                        league_name="Overtime V2",
                        market_type="winner",
                        home_team=info.get('homeTeam', 'Unknown'),
                        away_team=info.get('awayTeam', 'Unknown'),
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"\n✅ Added: {info.get('gameLabel', 'Unknown')}")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                
        if markets_added > 0:
            # Clear sample data
            with db_manager.get_db_session() as db:
                sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
                for market in sample_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
                
            logger.info(f"\n🎆 SUCCESS! Added {markets_added} real V2 markets from transactions!")
            logger.info("Dashboard at http://localhost:8888/unified now shows REAL blockchain data!")
            
    # Alternative: Look for market creation by checking receipt logs
    logger.info("\n🔍 Checking transaction receipts for market creation...") 
    
    # Get a sample transaction that might have created markets
    if tx_count > 0:
        # Get the most recent AMM transaction
        for i in range(100):
            block_num = current_block - i
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
                for tx in block['transactions']:
                    if tx['to'] and tx['to'].lower() == amm_address.lower():
                        # Get receipt
                        receipt = w3.eth.get_transaction_receipt(tx['hash'])
                        
                        if receipt['logs']:
                            logger.info(f"\n🧐 Transaction {tx['hash'].hex()} has {len(receipt['logs'])} logs")
                            
                            # Check logs for contract creation
                            for log in receipt['logs'][:5]:
                                log_addr = log['address']
                                
                                # Check if this is a new contract
                                code = w3.eth.get_code(log_addr)
                                if code and len(code) == 45:
                                    logger.info(f"  Log from market contract: {log_addr}")
                                    if log_addr not in markets_found:
                                        markets_found.add(log_addr)
                                        
                                        # Get info
                                        info = get_market_info_raw(w3, log_addr)
                                        if info.get('gameLabel'):
                                            logger.info(f"    Game: {info['gameLabel']}")
                                            
                            return  # Just check one transaction for now
                            
            except:
                continue

if __name__ == "__main__":
    main()