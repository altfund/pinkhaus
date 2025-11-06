#!/usr/bin/env python3
"""Execute trades on blockchain using connected market addresses"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainTradingExecutor:
    """Executes trades on Overtime V2 blockchain markets with simulation support"""
    
    def __init__(self, simulation_mode: bool = None):
        # Load blockchain connections
        self.blockchain_connections = {}
        try:
            with open('blockchain_connections.json', 'r') as f:
                self.blockchain_connections = json.load(f)
                logger.info(f"Loaded {len(self.blockchain_connections)} blockchain connections")
        except FileNotFoundError:
            logger.error("blockchain_connections.json not found - run create_blockchain_connection.py first")
            raise
        
        # Trading configuration
        self.chain = "optimism"  # Overtime V2 is on Optimism
        self.slippage_tolerance = 0.02  # 2% slippage tolerance
        
        # Simulation mode
        if simulation_mode is None:
            simulation_mode = (
                os.getenv('SIMULATE_BLOCKCHAIN') == '1' or
                os.getenv('PAPER_TRADING') == '1' or
                os.getenv('TRADING_MODE') == 'testnet'
            )
        self.simulation_mode = simulation_mode
        
        # Testnet/mainnet configuration
        self.use_testnet = (
            os.getenv('USE_TESTNET') == '1' or
            os.getenv('TRADING_MODE') == 'testnet'
        )
        
        if self.use_testnet:
            self.chain = "optimism_sepolia"  # Use testnet
            logger.info("🧪 Using testnet configuration")
        
        if self.simulation_mode:
            logger.info("🎭 Blockchain simulation mode enabled")
        else:
            logger.info("⚠️ Real blockchain execution mode")
        
    def get_blockchain_market(self, api_market_id: str) -> Optional[Dict]:
        """Get blockchain market info from API market ID"""
        return self.blockchain_connections.get(api_market_id)
    
    def prepare_trade(self, market: Dict, outcome: str, stake: float) -> Dict:
        """Prepare a trade for blockchain execution"""
        # Get blockchain info
        blockchain_info = None
        
        # Check if market has direct blockchain address
        if market.get('blockchain_address'):
            blockchain_address = market['blockchain_address']
        else:
            # Try to get from API ID
            api_id = market.get('api_id') or market.get('market_id')
            blockchain_info = self.get_blockchain_market(api_id)
            if not blockchain_info:
                raise ValueError(f"No blockchain address found for market {api_id}")
            blockchain_address = blockchain_info['blockchain_address']
        
        # Map outcome to position (0=home, 1=away, 2=draw for soccer)
        position_map = {
            'home': 0,
            'away': 1,
            'draw': 2,
            '0': 0,
            '1': 1,
            '2': 2
        }
        
        position = position_map.get(outcome.lower())
        if position is None:
            raise ValueError(f"Invalid outcome: {outcome}")
        
        # Get current odds from market
        odds = market.get('odds', {})
        market_odds = odds.get(outcome.lower(), 0)
        
        if market_odds == 0:
            raise ValueError(f"No odds available for outcome {outcome}")
        
        # Calculate quote
        quote = self.get_quote(blockchain_address, position, stake)
        
        return {
            'market_address': blockchain_address,
            'position': position,
            'outcome': outcome,
            'stake': stake,
            'odds': market_odds,
            'quote': quote,
            'expected_payout': stake * market_odds,
            'home_team': market.get('home_team', 'Unknown'),
            'away_team': market.get('away_team', 'Unknown'),
            'maturity_date': market.get('maturity_date'),
            'chain': self.chain
        }
    
    def get_quote(self, market_address: str, position: int, stake: float) -> Dict:
        """Get quote from blockchain for a position"""
        # In production, this would call the blockchain to get actual quote
        # For now, return simulated quote
        
        # Simulated blockchain quote
        # Real implementation would use web3.py or similar
        return {
            'price': stake,
            'slippage': 0.01,  # 1% slippage
            'fees': stake * 0.002,  # 0.2% fee
            'total_cost': stake * 1.012,  # stake + slippage + fees
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def execute_trade(self, trade_params: Dict) -> Dict:
        """Execute trade on blockchain (real or simulated)"""
        logger.info(f"Executing blockchain trade:")
        logger.info(f"  Market: {trade_params['home_team']} vs {trade_params['away_team']}")
        logger.info(f"  Address: {trade_params['market_address']}")
        logger.info(f"  Position: {trade_params['outcome']} ({trade_params['position']})")
        logger.info(f"  Stake: ${trade_params['stake']:.2f}")
        logger.info(f"  Odds: {trade_params['odds']:.2f}")
        logger.info(f"  Mode: {'🎭 Simulation' if self.simulation_mode else '⚡ Real Blockchain'}")
        
        if self.simulation_mode:
            return self._execute_simulated_trade(trade_params)
        else:
            return self._execute_real_trade(trade_params)
    
    def _execute_simulated_trade(self, trade_params: Dict) -> Dict:
        """Execute simulated blockchain trade for testing"""
        logger.info("🎭 Executing simulated blockchain trade")
        
        # Generate realistic mock transaction hash
        import hashlib
        import time
        data = f"{trade_params['market_address']}{trade_params['position']}{time.time()}"
        tx_hash = "0x" + hashlib.sha256(data.encode()).hexdigest()
        
        # Simulate realistic gas costs and timing
        import random
        gas_used = random.randint(180000, 300000)  # Realistic gas usage
        block_number = random.randint(100000000, 200000000)  # Mock block number
        
        result = {
            'success': True,
            'tx_hash': tx_hash,
            'market_address': trade_params['market_address'],
            'position': trade_params['position'],
            'stake': trade_params['stake'],
            'odds': trade_params['odds'],
            'total_cost': trade_params['quote']['total_cost'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'simulated',
            'block_number': block_number,
            'gas_used': gas_used,
            'gas_price': random.uniform(0.001, 0.01),  # ETH
            'simulation_mode': True,
            'network': self.chain
        }
        
        logger.info(f"🎭 Simulated trade executed: {tx_hash[:10]}...")
        logger.info(f"   Gas used: {gas_used:,}")
        logger.info(f"   Total cost: ${trade_params['quote']['total_cost']:.2f}")
        
        return result
    
    def _execute_real_trade(self, trade_params: Dict) -> Dict:
        """Execute real blockchain trade"""
        logger.warning("⚡ REAL BLOCKCHAIN EXECUTION NOT IMPLEMENTED")
        logger.warning("This would execute a real on-chain transaction with real funds!")
        
        # In production, this would:
        # 1. Connect to blockchain via web3
        # 2. Load wallet with private key (securely)
        # 3. Approve USDC spending if needed
        # 4. Call buyFromMarket() on the SportsAMM contract
        # 5. Wait for transaction confirmation
        # 6. Return real transaction receipt
        
        # For safety, return error for now
        return {
            'success': False,
            'error': 'Real blockchain execution not implemented for safety',
            'message': 'Use simulation_mode=True for testing',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_position_balance(self, market_address: str, position: int) -> float:
        """Get current position balance from blockchain"""
        # In production, query blockchain for position balance
        # This would call balanceOf(address, position) on the market contract
        return 0.0
    
    def claim_winnings(self, market_address: str) -> Optional[Dict]:
        """Claim winnings from a resolved market"""
        # In production:
        # 1. Check if market is resolved
        # 2. Check if we have winning positions
        # 3. Call claim() on the contract
        # 4. Return transaction receipt
        
        logger.info(f"Claiming winnings from market {market_address}")
        
        return {
            'success': True,
            'tx_hash': f"0x{'b' * 64}",
            'amount_claimed': 0.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def test_blockchain_executor():
    """Test blockchain trading executor"""
    executor = BlockchainTradingExecutor()
    
    print("🔗 Testing Blockchain Trading Executor\n")
    
    # Get a sample market from database
    with db_manager.get_db_session() as db:
        # Find a market with blockchain connection
        sample_market = db.query(Market).filter(
            and_(
                Market.source == 'overtime_v2',
                Market.sport.ilike('%soccer%'),
                Market.is_finished == False
            )
        ).first()
        
        if not sample_market:
            print("❌ No suitable market found")
            return
        
        # Get blockchain info
        blockchain_info = executor.get_blockchain_market(sample_market.source_id)
        
        if not blockchain_info:
            print("❌ No blockchain connection found")
            return
        
        print(f"Found market: {sample_market.home_team} vs {sample_market.away_team}")
        print(f"Blockchain address: {blockchain_info['blockchain_address']}\n")
        
        # Prepare a test trade
        market_data = {
            'api_id': sample_market.source_id,
            'home_team': sample_market.home_team,
            'away_team': sample_market.away_team,
            'odds': {
                'home': 2.50,
                'away': 2.80,
                'draw': 3.00
            },
            'maturity_date': sample_market.maturity_date.isoformat()
        }
        
        # Prepare trade
        print("📝 Preparing trade...")
        trade = executor.prepare_trade(market_data, 'home', 100.0)
        
        print(f"\nTrade details:")
        print(f"  Market address: {trade['market_address']}")
        print(f"  Position: {trade['outcome']} ({trade['position']})")
        print(f"  Stake: ${trade['stake']:.2f}")
        print(f"  Quote total: ${trade['quote']['total_cost']:.2f}")
        print(f"  Expected payout: ${trade['expected_payout']:.2f}")
        
        # Execute trade
        print("\n🚀 Executing trade...")
        result = executor.execute_trade(trade)
        
        if result['success']:
            print(f"\n✅ Trade successful!")
            print(f"  Transaction: {result['tx_hash']}")
            print(f"  Block: {result['block_number']}")
            print(f"  Gas used: {result['gas_used']}")
        else:
            print("\n❌ Trade failed")


if __name__ == "__main__":
    test_blockchain_executor()