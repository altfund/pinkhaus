#!/usr/bin/env python3
"""
Real trading engine - executes actual trades on the blockchain
Handles transaction management, gas optimization, and safety
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from web3 import Web3
from eth_account import Account

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from real_trading_config import RealTradingConfig
from blockchain_liquidity_fetcher import BlockchainLiquidityFetcher
from notifications.discord_notifier import discord_notifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTradingEngine:
    """Executes real trades on blockchain"""
    
    def __init__(self, config: Optional[RealTradingConfig] = None):
        self.config = config or RealTradingConfig()
        self.liquidity_fetcher = BlockchainLiquidityFetcher()
        self.w3_connections = {}
        self.account = None
        self.daily_stats = {'trades': 0, 'volume': 0.0, 'losses': 0.0}
        self.pending_txs = {}
        
        # Initialize if configured
        if self.config.is_configured():
            self._initialize_connections()
            
    def _initialize_connections(self):
        """Initialize Web3 connections and account"""
        private_key = self.config.get_private_key()
        if not private_key:
            raise ValueError("No private key configured")
            
        self.account = Account.from_key(private_key)
        logger.info(f"Trading account: {self.account.address}")
        
        # Initialize Web3 connections for each network
        for network, config in self.config.config['networks'].items():
            w3 = Web3(Web3.HTTPProvider(config['rpc_url']))
            
            # POA middleware is now automatically handled in newer web3 versions
                
            self.w3_connections[network] = w3
            
            # Check connection
            if w3.is_connected():
                logger.info(f"Connected to {network}")
                balance = w3.eth.get_balance(self.account.address)
                logger.info(f"  Balance: {Web3.from_wei(balance, 'ether')} ETH")
            else:
                logger.error(f"Failed to connect to {network}")
                
    def check_collateral_balance(self, network: str) -> Decimal:
        """Check collateral token balance (USDC/sUSD)"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return Decimal('0')
            
        contracts = self.config.get_approved_contracts(network)
        collateral_address = contracts.get('collateral')
        
        if not collateral_address:
            return Decimal('0')
            
        # ERC20 balance check ABI
        abi = [{
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }]
        
        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(collateral_address),
                abi=abi
            )
            
            balance_wei = contract.functions.balanceOf(self.account.address).call()
            # Most stablecoins use 6 decimals (USDC) or 18 (sUSD)
            decimals = 6 if network in ['arbitrum', 'base'] else 18
            balance = Decimal(str(balance_wei)) / Decimal(10 ** decimals)
            
            logger.info(f"Collateral balance on {network}: ${balance}")
            return balance
            
        except Exception as e:
            logger.error(f"Error checking balance on {network}: {e}")
            return Decimal('0')
            
    def approve_collateral_if_needed(self, network: str, amount: Decimal) -> Optional[str]:
        """Approve collateral spending if needed"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return None
            
        contracts = self.config.get_approved_contracts(network)
        collateral_address = contracts.get('collateral')
        spender_address = contracts.get('sports_amm_v2')
        
        if not collateral_address or not spender_address:
            return None
            
        # Check current allowance
        abi = [
            {
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(collateral_address),
                abi=abi
            )
            
            # Check allowance
            current_allowance = contract.functions.allowance(
                self.account.address,
                Web3.to_checksum_address(spender_address)
            ).call()
            
            decimals = 6 if network in ['arbitrum', 'base'] else 18
            amount_wei = int(amount * Decimal(10 ** decimals))
            
            if current_allowance >= amount_wei:
                return None  # Already approved
                
            # Build approval transaction
            logger.info(f"Approving {amount} collateral on {network}")
            
            # Approve max uint256 for convenience
            max_approval = 2**256 - 1
            
            tx = contract.functions.approve(
                Web3.to_checksum_address(spender_address),
                max_approval
            ).build_transaction({
                'from': self.account.address,
                'nonce': w3.eth.get_transaction_count(self.account.address),
                'gas': 100000,
                'gasPrice': self._get_gas_price(network)
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Approval tx sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error approving collateral: {e}")
            return None
            
    def _get_gas_price(self, network: str) -> int:
        """Get appropriate gas price for network"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return 0
            
        try:
            # Get current gas price
            base_price = w3.eth.gas_price
            
            # Apply network-specific multipliers
            if network == 'arbitrum':
                # Arbitrum is usually cheap
                return int(base_price * 1.1)
            elif network == 'optimism':
                # Optimism moderate
                return int(base_price * 1.2)
            elif network == 'base':
                # Base is cheap
                return int(base_price * 1.1)
            else:
                return int(base_price * 1.5)
                
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            # Fallback prices in Wei
            fallback = {
                'arbitrum': Web3.to_wei(0.1, 'gwei'),
                'optimism': Web3.to_wei(0.01, 'gwei'),
                'base': Web3.to_wei(0.01, 'gwei')
            }
            return fallback.get(network, Web3.to_wei(1, 'gwei'))
            
    async def place_real_bet(self, opportunity: Dict, network: str = None) -> Optional[Dict]:
        """Place a real bet on the blockchain"""
        network = network or self.config.config['default_network']
        
        # Safety checks
        market = opportunity['market']
        amount_usd = opportunity['adjusted_bet_size']
        edge = opportunity['edge']
        
        # Check if we're in testnet mode
        if self.config.get_mode() == 'testnet':
            logger.warning("In testnet mode - would place bet but not executing")
            return None
            
        # Validate trade
        valid, reason = self.config.validate_trade(
            amount_usd,
            self._get_current_exposure(),
            self.daily_stats['losses'],
            edge
        )
        
        if not valid:
            logger.warning(f"Trade rejected: {reason}")
            discord_notifier.send_error_alert(reason, "Trade Validation")
            return None
            
        # Check collateral balance
        balance = self.check_collateral_balance(network)
        if balance < amount_usd:
            logger.error(f"Insufficient balance: ${balance} < ${amount_usd}")
            return None
            
        # Get market address from source_id
        market_address = None
        if market.source_id.startswith('v2_0x'):
            market_address = market.source_id[3:]
        elif market.source_id.startswith('0x'):
            market_address = market.source_id
            
        if not market_address:
            logger.error("No blockchain market address found")
            return None
            
        # Approve collateral if needed
        approval_tx = self.approve_collateral_if_needed(network, Decimal(str(amount_usd)))
        if approval_tx:
            # Wait for approval
            await self._wait_for_transaction(approval_tx, network)
            
        # Execute the trade
        try:
            tx_hash = await self._execute_buy_from_amm(
                network,
                market_address,
                opportunity['outcome'],
                Decimal(str(amount_usd))
            )
            
            if tx_hash:
                # Record in database
                self._record_real_bet(opportunity, tx_hash, network)
                
                # Update daily stats
                self.daily_stats['trades'] += 1
                self.daily_stats['volume'] += amount_usd
                
                # Send notification
                discord_notifier.send_trade_alert({
                    'type': 'NEW',
                    'market': f"{market.home_team} vs {market.away_team}",
                    'outcome': opportunity['outcome'],
                    'amount': amount_usd,
                    'odds': opportunity['odds'],
                    'edge': edge,
                    'tx_hash': tx_hash,
                    'network': network,
                    'mode': 'REAL',
                    'bankroll': float(balance)
                })
                
                return {
                    'tx_hash': tx_hash,
                    'amount': amount_usd,
                    'market': market_address,
                    'outcome': opportunity['outcome']
                }
                
        except Exception as e:
            logger.error(f"Error placing bet: {e}")
            discord_notifier.send_error_alert(str(e), "Trade Execution")
            return None
            
    async def _execute_buy_from_amm(self, network: str, market_address: str, 
                                   outcome: str, amount: Decimal) -> Optional[str]:
        """Execute buyFromAMM transaction"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return None
            
        contracts = self.config.get_approved_contracts(network)
        amm_address = contracts.get('sports_amm_v2')
        
        if not amm_address:
            return None
            
        # Convert outcome to position index
        position_map = {'home': 0, 'away': 1, 'draw': 2}
        position = position_map.get(outcome, 0)
        
        # Convert amount to wei (considering decimals)
        decimals = 6 if network in ['arbitrum', 'base'] else 18
        amount_wei = int(amount * Decimal(10 ** decimals))
        
        # Build transaction
        abi = [{
            "inputs": [
                {"name": "_market", "type": "address"},
                {"name": "_position", "type": "uint8"},
                {"name": "_amount", "type": "uint256"},
                {"name": "_expectedQuote", "type": "uint256"},
                {"name": "_additionalSlippage", "type": "uint256"}
            ],
            "name": "buyFromAMM",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }]
        
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(amm_address),
            abi=abi
        )
        
        # Get quote first
        quote_wei = self._get_buy_quote(network, market_address, position, amount_wei)
        if not quote_wei:
            return None
            
        # Allow 1% slippage
        slippage_bps = 100  # basis points
        
        logger.info(f"Executing buy: {amount} {outcome} on {market_address[:8]}...")
        
        # Build transaction
        tx = contract.functions.buyFromAMM(
            Web3.to_checksum_address(market_address),
            position,
            amount_wei,
            quote_wei,
            slippage_bps
        ).build_transaction({
            'from': self.account.address,
            'nonce': w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self._get_gas_price(network)
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        self.pending_txs[tx_hash.hex()] = {
            'network': network,
            'amount': float(amount),
            'timestamp': datetime.now(timezone.utc)
        }
        
        return tx_hash.hex()
        
    def _get_buy_quote(self, network: str, market_address: str, 
                      position: int, amount_wei: int) -> Optional[int]:
        """Get quote for buying from AMM"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return None
            
        contracts = self.config.get_approved_contracts(network)
        amm_address = contracts.get('sports_amm_v2')
        
        abi = [{
            "inputs": [
                {"name": "_market", "type": "address"},
                {"name": "_position", "type": "uint8"},
                {"name": "_amount", "type": "uint256"}
            ],
            "name": "buyFromAmmQuote",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }]
        
        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(amm_address),
                abi=abi
            )
            
            quote = contract.functions.buyFromAmmQuote(
                Web3.to_checksum_address(market_address),
                position,
                amount_wei
            ).call()
            
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None
            
    async def _wait_for_transaction(self, tx_hash: str, network: str, timeout: int = 120):
        """Wait for transaction confirmation"""
        w3 = self.w3_connections.get(network)
        if not w3:
            return None
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    if receipt['status'] == 1:
                        logger.info(f"Transaction confirmed: {tx_hash}")
                        return receipt
                    else:
                        logger.error(f"Transaction failed: {tx_hash}")
                        return None
            except Exception:
                pass
                
            await asyncio.sleep(2)
            
        logger.error(f"Transaction timeout: {tx_hash}")
        return None
        
    def _record_real_bet(self, opportunity: Dict, tx_hash: str, network: str):
        """Record real bet in database"""
        with db_manager.get_db_session() as db:
            market = opportunity['market']
            
            bet = Bet(
                session_id=opportunity.get('session_id'),
                source_id=market.source_id,
                sport=market.sport,
                league=market.league,
                home_team=market.home_team,
                away_team=market.away_team,
                outcome=opportunity['outcome'],
                stake=opportunity['adjusted_bet_size'],
                decimal_odds=opportunity['odds'],
                placed_at=datetime.now(timezone.utc),
                status='pending',
                metadata={
                    'edge': opportunity['edge'],
                    'tx_hash': tx_hash,
                    'network': network,
                    'is_real': True,
                    'wallet': self.account.address
                }
            )
            db.add(bet)
            db.commit()
            
    def _get_current_exposure(self) -> float:
        """Get current exposure across all active bets"""
        # This would check actual on-chain positions
        # For now return estimate from pending txs
        total = 0.0
        for tx_data in self.pending_txs.values():
            # Only count recent transactions
            if (datetime.now(timezone.utc) - tx_data['timestamp']).seconds < 3600:
                total += tx_data['amount']
        return total
        
    async def monitor_positions(self):
        """Monitor on-chain positions and settle completed bets"""
        # This would monitor blockchain for resolved markets
        # and update database accordingly
        pass


def main():
    """Test real trading engine"""
    config = RealTradingConfig()
    
    if not config.is_configured():
        print("Trading not configured. Set up wallet first.")
        return
        
    engine = RealTradingEngine(config)
    
    # Check balances
    for network in ['arbitrum', 'optimism', 'base']:
        balance = engine.check_collateral_balance(network)
        print(f"{network}: ${balance}")
        

if __name__ == "__main__":
    main()