#!/usr/bin/env python3
"""
Blockchain liquidity fetcher for Overtime Markets
Gets actual liquidity data from smart contracts to ensure trades are executable
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainLiquidityFetcher:
    """Fetches real liquidity data from Overtime smart contracts"""
    
    def __init__(self):
        # Initialize Web3 connections
        self.w3_connections = {
            'arbitrum': Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc')),
            'optimism': Web3(Web3.HTTPProvider('https://mainnet.optimism.io')),
            'base': Web3(Web3.HTTPProvider('https://mainnet.base.org'))
        }
        
        # SportsAMMV2 contract addresses
        self.amm_contracts = {
            'arbitrum': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
            'optimism': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
            'base': '0xC3E7f5a2548c446555bb3D99EdE57e73b02fEb58'
        }
        
        # Collateral contracts (sUSD on Optimism, USDC on Arbitrum/Base)
        self.collateral_contracts = {
            'arbitrum': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',  # USDC
            'optimism': '0x8c6f28f2F1A3C87F0f938b96d27520d9751ec8d9',  # sUSD
            'base': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913'       # USDC
        }
        
        # Liquidity Pool contracts
        self.pool_contracts = {
            'arbitrum': '0xD6BdDE3aeB61292F6EbCe6E3d93D5291844923a3',
            'optimism': '0x6D1ee6c0Bc92c0b33781E40dA1a2097D7DFa5F3d',
            'base': '0xbE551054801fC8d2E22044c3E28fb51565529c88'
        }
        
        # Contract ABIs
        self.amm_abi = [
            {
                "inputs": [
                    {"name": "_market", "type": "address"},
                    {"name": "_position", "type": "uint8"},
                    {"name": "_amount", "type": "uint256"}
                ],
                "name": "buyFromAmmQuote",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "_market", "type": "address"},
                    {"name": "_position", "type": "uint8"}, 
                    {"name": "_amount", "type": "uint256"}
                ],
                "name": "buyFromAmmQuoteForPayout",
                "outputs": [
                    {"name": "susdQuote", "type": "uint256"},
                    {"name": "additionalSlippage", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "_market", "type": "address"}],
                "name": "getMarketDefaultOdds",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "liquidityPool",
                "outputs": [{"name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        self.pool_abi = [
            {
                "inputs": [],
                "name": "getMaxAvailableToBuyFromAMM",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "totalDeposited",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
    def get_market_liquidity(self, market_address: str, network: str = 'arbitrum') -> Dict:
        """
        Get liquidity information for a specific market
        Returns liquidity depth, max bet size, and slippage estimates
        """
        try:
            w3 = self.w3_connections.get(network)
            if not w3 or not w3.is_connected():
                logger.error(f"Web3 not connected for {network}")
                return {}
                
            amm_address = self.amm_contracts.get(network)
            if not amm_address:
                logger.error(f"No AMM address for {network}")
                return {}
                
            # Get AMM contract
            amm = w3.eth.contract(
                address=Web3.to_checksum_address(amm_address),
                abi=self.amm_abi
            )
            
            # Get liquidity pool address
            try:
                pool_address = amm.functions.liquidityPool().call()
            except:
                pool_address = self.pool_contracts.get(network)
                
            if not pool_address:
                logger.error("Could not get pool address")
                return {}
                
            # Get pool contract
            pool = w3.eth.contract(
                address=Web3.to_checksum_address(pool_address),
                abi=self.pool_abi
            )
            
            # Get max available liquidity
            max_liquidity = pool.functions.getMaxAvailableToBuyFromAMM().call()
            max_liquidity_usd = max_liquidity / 1e18  # Convert from 18 decimals
            
            # Get total deposited in pool
            total_deposited = pool.functions.totalDeposited().call()
            total_deposited_usd = total_deposited / 1e18
            
            # Get default odds for the market
            odds_raw = amm.functions.getMarketDefaultOdds(
                Web3.to_checksum_address(market_address)
            ).call()
            
            # Calculate liquidity for each outcome
            liquidity_by_outcome = {}
            
            for position in range(len(odds_raw)):
                if odds_raw[position] == 0:
                    continue
                    
                # Test different bet sizes to understand liquidity depth
                test_amounts = [10, 50, 100, 500, 1000, 5000]  # USD
                quotes = []
                
                for amount in test_amounts:
                    amount_wei = int(amount * 1e18)
                    try:
                        # Get quote for buying this amount
                        quote_wei = amm.functions.buyFromAmmQuote(
                            Web3.to_checksum_address(market_address),
                            position,
                            amount_wei
                        ).call()
                        
                        quote_usd = quote_wei / 1e18
                        effective_odds = amount / quote_usd if quote_usd > 0 else 0
                        
                        quotes.append({
                            'amount': amount,
                            'cost': quote_usd,
                            'effective_odds': effective_odds,
                            'slippage': self._calculate_slippage(odds_raw[position], effective_odds)
                        })
                    except Exception as e:
                        # This amount is too large for current liquidity
                        logger.debug(f"Cannot quote {amount} USD for position {position}: {e}")
                        break
                        
                liquidity_by_outcome[position] = {
                    'base_odds': 1e18 / odds_raw[position],
                    'quotes': quotes,
                    'max_bet': quotes[-1]['amount'] if quotes else 0
                }
                
            return {
                'market_address': market_address,
                'network': network,
                'max_liquidity_usd': max_liquidity_usd,
                'total_pool_size': total_deposited_usd,
                'liquidity_by_outcome': liquidity_by_outcome,
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error fetching liquidity for {market_address}: {e}")
            return {}
            
    def _calculate_slippage(self, base_odds_raw: int, effective_odds: float) -> float:
        """Calculate slippage percentage"""
        if base_odds_raw == 0:
            return 0
        base_odds = 1e18 / base_odds_raw
        return ((effective_odds - base_odds) / base_odds) * 100
        
    def get_optimal_bet_size(self, market_address: str, outcome: int, 
                           max_slippage_pct: float = 1.0, network: str = 'arbitrum') -> Dict:
        """
        Calculate optimal bet size given slippage constraints
        """
        liquidity = self.get_market_liquidity(market_address, network)
        if not liquidity or 'liquidity_by_outcome' not in liquidity:
            return {'optimal_size': 0, 'reason': 'No liquidity data'}
            
        outcome_liquidity = liquidity['liquidity_by_outcome'].get(outcome)
        if not outcome_liquidity:
            return {'optimal_size': 0, 'reason': 'No liquidity for outcome'}
            
        quotes = outcome_liquidity.get('quotes', [])
        if not quotes:
            return {'optimal_size': 0, 'reason': 'No quotes available'}
            
        # Find largest bet size within slippage tolerance
        optimal_size = 0
        optimal_quote = None
        
        for quote in quotes:
            if abs(quote['slippage']) <= max_slippage_pct:
                optimal_size = quote['amount']
                optimal_quote = quote
            else:
                break
                
        return {
            'optimal_size': optimal_size,
            'effective_odds': optimal_quote['effective_odds'] if optimal_quote else 0,
            'cost': optimal_quote['cost'] if optimal_quote else 0,
            'slippage': optimal_quote['slippage'] if optimal_quote else 0,
            'max_liquidity': liquidity['max_liquidity_usd'],
            'reason': 'Within slippage tolerance' if optimal_size > 0 else 'Exceeds slippage tolerance'
        }
        
    def update_database_with_liquidity(self):
        """Update database with current liquidity information"""
        logger.info("Updating liquidity data from blockchain...")
        
        with db_manager.get_db_session() as db:
            # Get active markets that might have blockchain addresses
            active_markets = db.query(Market).filter(
                Market.is_active == True,
                Market.source.in_(['overtime_v2', 'blockchain'])
            ).limit(50).all()
            
            updated_count = 0
            
            for market in active_markets:
                # Extract blockchain address from source_id
                # Format: v2_0x... or just 0x...
                market_address = None
                if market.source_id.startswith('v2_0x'):
                    market_address = market.source_id[3:]  # Remove 'v2_' prefix
                elif market.source_id.startswith('0x'):
                    market_address = market.source_id
                    
                if not market_address:
                    continue
                    
                # Determine network (could be stored in market metadata)
                network = 'arbitrum'  # Default to Arbitrum for now
                
                # Get liquidity data
                liquidity = self.get_market_liquidity(market_address, network)
                
                if liquidity and liquidity.get('max_liquidity_usd', 0) > 0:
                    # Store liquidity data as market metadata
                    if not hasattr(market, 'metadata') or market.metadata is None:
                        market.metadata = {}
                        
                    market.metadata['liquidity'] = {
                        'max_usd': liquidity['max_liquidity_usd'],
                        'pool_size': liquidity['total_pool_size'],
                        'last_updated': liquidity['timestamp'].isoformat(),
                        'network': network
                    }
                    
                    # Store per-outcome liquidity
                    for outcome_idx, outcome_data in liquidity['liquidity_by_outcome'].items():
                        outcome_name = ['home', 'away', 'draw'][outcome_idx] if outcome_idx < 3 else str(outcome_idx)
                        market.metadata['liquidity'][f'{outcome_name}_max_bet'] = outcome_data['max_bet']
                        
                    updated_count += 1
                    logger.info(f"Updated liquidity for {market.home_team} vs {market.away_team}: ${liquidity['max_liquidity_usd']:.2f}")
                    
            db.commit()
            logger.info(f"Liquidity update complete. Updated {updated_count} markets.")
            

def main():
    """Test the liquidity fetcher"""
    fetcher = BlockchainLiquidityFetcher()
    
    # Test with a known market address (example)
    test_address = "0x1234567890123456789012345678901234567890"
    
    logger.info("Testing liquidity fetch...")
    liquidity = fetcher.get_market_liquidity(test_address)
    
    if liquidity:
        print(f"\nLiquidity for market {test_address}:")
        print(f"  Max liquidity: ${liquidity['max_liquidity_usd']:,.2f}")
        print(f"  Pool size: ${liquidity['total_pool_size']:,.2f}")
        print(f"  Outcomes: {len(liquidity['liquidity_by_outcome'])}")
        
        # Test optimal bet sizing
        optimal = fetcher.get_optimal_bet_size(test_address, 0, max_slippage_pct=1.0)
        print(f"\nOptimal bet size for outcome 0:")
        print(f"  Size: ${optimal['optimal_size']}")
        print(f"  Slippage: {optimal['slippage']:.2f}%")
        print(f"  Reason: {optimal['reason']}")
    else:
        print("No liquidity data available")
        
    # Update database
    fetcher.update_database_with_liquidity()
    

if __name__ == "__main__":
    main()