#!/usr/bin/env python3
"""
Blockchain Trading System for Ominari
Integrates blockchain reading with trading execution on Overtime Markets.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
from web3 import Web3
from eth_account import Account

from blockchain_reader import BlockchainReader, BlockchainConfig
from database_v2 import db_manager
from models import Market, Odd
from paper_trading_engine import PaperTradingEngine
try:
    from telemetry import get_telemetry, traced, trace_api_call
except ImportError:
    from mock_telemetry import get_telemetry, traced, trace_api_call

logger = logging.getLogger(__name__)


@dataclass
class BlockchainPosition:
    """Represents a position to take on the blockchain."""
    market_address: str
    position: int  # 0=home, 1=away, 2=draw
    amount: Decimal
    expected_odds: Decimal
    slippage_tolerance: Decimal = Decimal('0.02')  # 2% slippage


@dataclass
class TradeResult:
    """Result of a blockchain trade execution."""
    success: bool
    tx_hash: Optional[str]
    position: BlockchainPosition
    actual_odds: Optional[Decimal]
    susd_paid: Optional[Decimal]
    gas_used: Optional[int]
    error: Optional[str]


class BlockchainTrader:
    """Executes trades on Overtime Markets via blockchain."""
    
    def __init__(self, 
                 network: str = 'optimism',
                 private_key: Optional[str] = None):
        self.network = network
        self.config = BlockchainConfig.NETWORKS[network]
        self.reader = BlockchainReader(network)
        self.telemetry = get_telemetry()
        
        # Load private key from env or parameter
        self.private_key = private_key or os.getenv('TRADING_PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            logger.info(f"Trading account: {self.account.address}")
        else:
            self.account = None
            logger.warning("No private key provided - running in read-only mode")
            
        # Initialize contracts
        self._init_contracts()
        
    def _init_contracts(self):
        """Initialize smart contract interfaces."""
        # SportsAMMV2 contract
        self.sports_amm = self.reader.sports_amm
        
        # SUSD contract (for approvals)
        susd_addresses = {
            'optimism': '0x8c6f28f2F1A3C87F0f938b96d27520d9751ec8d9',
            'arbitrum': '0xA970AF1a584579B618be4d69aD6F73459D112F95',
            'optimism_sepolia': '0x4C5d8A75F3762c1561D96f177694f67378705E98'
        }
        
        susd_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        self.susd = self.reader.w3.eth.contract(
            address=Web3.to_checksum_address(susd_addresses[self.network]),
            abi=susd_abi
        )
        
    def check_balance(self) -> Dict[str, Decimal]:
        """Check SUSD balance and ETH for gas."""
        if not self.account:
            return {'susd': Decimal('0'), 'eth': Decimal('0')}
            
        # Get SUSD balance
        susd_balance = self.susd.functions.balanceOf(self.account.address).call()
        susd_decimal = Decimal(str(susd_balance)) / Decimal('1e18')
        
        # Get ETH balance for gas
        eth_balance = self.reader.w3.eth.get_balance(self.account.address)
        eth_decimal = Decimal(str(eth_balance)) / Decimal('1e18')
        
        return {
            'susd': susd_decimal,
            'eth': eth_decimal
        }
        
    @traced("blockchain.quote_trade")
    def quote_trade(self, position: BlockchainPosition) -> Dict[str, Any]:
        """Get a quote for a trade without executing."""
        try:
            # Convert amount to Wei
            buy_amount_wei = int(position.amount * Decimal('1e18'))
            
            # Get quote from contract
            quote_data = self.sports_amm.functions.buyFromAmmQuote(
                Web3.to_checksum_address(position.market_address),
                position.position,
                buy_amount_wei
            ).call()
            
            # Parse quote response
            susd_needed = Decimal(str(quote_data[0])) / Decimal('1e18')
            additional_slippage = Decimal(str(quote_data[1])) / Decimal('1e18')
            
            # Calculate effective odds
            if position.amount > 0:
                effective_odds = susd_needed / position.amount
            else:
                effective_odds = Decimal('0')
                
            # Check slippage
            slippage = abs(effective_odds - position.expected_odds) / position.expected_odds
            acceptable = slippage <= position.slippage_tolerance
            
            return {
                'susd_needed': susd_needed,
                'additional_slippage': additional_slippage,
                'effective_odds': effective_odds,
                'slippage': slippage,
                'acceptable': acceptable
            }
            
        except Exception as e:
            logger.error(f"Quote failed: {e}")
            return {
                'error': str(e),
                'acceptable': False
            }
            
    def ensure_approval(self, amount: Decimal) -> bool:
        """Ensure SUSD spending is approved for SportsAMM."""
        if not self.account:
            return False
            
        try:
            # Check current allowance
            current_allowance = self.susd.functions.allowance(
                self.account.address,
                self.sports_amm.address
            ).call()
            
            current_decimal = Decimal(str(current_allowance)) / Decimal('1e18')
            
            if current_decimal >= amount:
                return True
                
            # Need to approve
            logger.info(f"Approving {amount} SUSD for SportsAMM")
            
            # Build approval transaction
            amount_wei = int(amount * Decimal('1e18'))
            
            tx = self.susd.functions.approve(
                self.sports_amm.address,
                amount_wei
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.reader.w3.eth.get_transaction_count(self.account.address),
                'gas': 100000,
                'gasPrice': self.reader.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.reader.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.reader.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.reader.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Approval successful: {tx_hash.hex()}")
                return True
            else:
                logger.error(f"Approval failed: {tx_hash.hex()}")
                return False
                
        except Exception as e:
            logger.error(f"Approval error: {e}")
            return False
            
    @traced("blockchain.execute_trade")
    def execute_trade(self, position: BlockchainPosition) -> TradeResult:
        """Execute a trade on the blockchain."""
        if not self.account:
            return TradeResult(
                success=False,
                tx_hash=None,
                position=position,
                actual_odds=None,
                susd_paid=None,
                gas_used=None,
                error="No trading account configured"
            )
            
        try:
            # Get quote first
            quote = self.quote_trade(position)
            
            if not quote.get('acceptable', False):
                return TradeResult(
                    success=False,
                    tx_hash=None,
                    position=position,
                    actual_odds=quote.get('effective_odds'),
                    susd_paid=None,
                    gas_used=None,
                    error=f"Slippage too high: {quote.get('slippage', 0):.2%}"
                )
                
            # Ensure approval
            susd_needed = quote['susd_needed'] + quote['additional_slippage']
            if not self.ensure_approval(susd_needed):
                return TradeResult(
                    success=False,
                    tx_hash=None,
                    position=position,
                    actual_odds=None,
                    susd_paid=None,
                    gas_used=None,
                    error="Failed to approve SUSD spending"
                )
                
            # Build trade transaction
            buy_amount_wei = int(position.amount * Decimal('1e18'))
            expected_payout_wei = int(susd_needed * Decimal('1e18') * Decimal('1.02'))  # 2% buffer
            additional_slippage_wei = int(quote['additional_slippage'] * Decimal('1e18'))
            
            tx = self.sports_amm.functions.buyFromAmm(
                Web3.to_checksum_address(position.market_address),
                position.position,
                buy_amount_wei,
                expected_payout_wei,
                additional_slippage_wei
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.reader.w3.eth.get_transaction_count(self.account.address),
                'gas': 300000,
                'gasPrice': self.reader.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.reader.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.reader.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Trade submitted: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.reader.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            
            if receipt['status'] == 1:
                # Parse logs to get actual amounts
                # This would require decoding the BoughtFromAmm event
                
                # Record metrics
                self.telemetry.record_bet_placed(
                    amount=float(position.amount),
                    market="blockchain",
                    strategy="blockchain_direct"
                )
                
                return TradeResult(
                    success=True,
                    tx_hash=tx_hash.hex(),
                    position=position,
                    actual_odds=quote['effective_odds'],
                    susd_paid=susd_needed,
                    gas_used=receipt['gasUsed'],
                    error=None
                )
            else:
                return TradeResult(
                    success=False,
                    tx_hash=tx_hash.hex(),
                    position=position,
                    actual_odds=None,
                    susd_paid=None,
                    gas_used=receipt['gasUsed'],
                    error="Transaction reverted"
                )
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return TradeResult(
                success=False,
                tx_hash=None,
                position=position,
                actual_odds=None,
                susd_paid=None,
                gas_used=None,
                error=str(e)
            )
            
    def monitor_position(self, market_address: str, tx_hash: str) -> Dict[str, Any]:
        """Monitor an open position for settlement."""
        # This would check if the market has resolved
        # and calculate P&L
        pass


class BlockchainIntegration:
    """Integrates blockchain reading and trading with the main system."""
    
    def __init__(self,
                 network: str = 'optimism',
                 paper_trading_mode: bool = True):
        self.network = network
        self.paper_trading_mode = paper_trading_mode
        self.reader = BlockchainReader(network)
        
        if not paper_trading_mode:
            self.trader = BlockchainTrader(network)
        else:
            self.trader = None
            
        self.telemetry = get_telemetry()
        
    @traced("blockchain.sync_markets")
    def sync_blockchain_markets(self, hours_back: int = 24):
        """Sync recent markets from blockchain to database."""
        logger.info(f"Syncing blockchain markets from last {hours_back} hours")
        
        # Fetch recent markets from blockchain
        blockchain_markets = self.reader.fetch_recent_markets(hours_back)
        
        # Store in database
        stored_count = 0
        with db_manager.get_db_session() as db:
            for bm in blockchain_markets:
                # Check if market exists
                existing = db.query(Market).filter_by(
                    source_id=bm['source_id']
                ).first()
                
                if not existing:
                    market = Market(
                        source='blockchain',
                        source_id=bm['source_id'],
                        sport=bm['sport'],
                        league_name=bm['league'],
                        home_team=bm['home_team'],
                        away_team=bm['away_team'],
                        market_type=bm['market_type'],
                        maturity_date=bm['maturity_date'],
                        is_finished=False
                    )
                    db.add(market)
                    stored_count += 1
                    
                    # Add initial odds
                    for i, odds_value in enumerate(bm.get('normalized_odds', [])):
                        if odds_value > 0:
                            decimal_odds = 1e18 / odds_value  # Convert from protocol format
                            
                            odd = Odd(
                                source_id=bm['source_id'],
                                source='blockchain',
                                bookmaker='overtime_blockchain',
                                market_type=bm['market_type'],
                                outcome=f'option_{i}',
                                position=i,
                                line=0,
                                decimal_odds=decimal_odds,
                                american_odds=self._decimal_to_american(decimal_odds),
                                normalized_implied=100 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
            db.commit()
            
        logger.info(f"Stored {stored_count} new markets from blockchain")
        return stored_count
        
    @traced("blockchain.update_odds")
    def update_blockchain_odds(self, market_addresses: Optional[List[str]] = None):
        """Update odds for blockchain markets."""
        if market_addresses is None:
            # Get active blockchain markets from DB
            with db_manager.get_db_session() as db:
                active_markets = db.query(Market).filter(
                    Market.source == 'blockchain',
                    Market.is_finished == False,
                    Market.maturity_date > datetime.now(timezone.utc)
                ).all()
                
                market_addresses = [m.source_id for m in active_markets]
                
        logger.info(f"Updating odds for {len(market_addresses)} markets")
        
        updated_count = 0
        for address in market_addresses:
            try:
                # Get current odds from blockchain
                odds_data = self.reader.get_current_odds(address)
                
                with db_manager.get_db_session() as db:
                    for position, odds_info in odds_data.items():
                        # Update or create odd record
                        odd = db.query(Odd).filter(
                            Odd.source_id == address,
                            Odd.source == 'blockchain',
                            Odd.outcome == f'option_{position}'
                        ).first()
                        
                        if odd:
                            odd.decimal_odds = odds_info['buy']
                            odd.normalized_implied = 100 / odds_info['buy'] if odds_info['buy'] > 0 else 0
                            odd.updated_at = datetime.now(timezone.utc)
                        else:
                            odd = Odd(
                                source_id=address,
                                source='blockchain',
                                bookmaker='overtime_blockchain',
                                market_type='moneyline',
                                outcome=f'option_{position}',
                                position=position,
                                line=0,
                                decimal_odds=odds_info['buy'],
                                american_odds=self._decimal_to_american(odds_info['buy']),
                                normalized_implied=100 / odds_info['buy'] if odds_info['buy'] > 0 else 0,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
                    db.commit()
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Error updating odds for {address}: {e}")
                
        logger.info(f"Updated odds for {updated_count} markets")
        return updated_count
        
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
            
    def evaluate_blockchain_opportunities(self, 
                                        min_edge: float = 0.02,
                                        max_odds: float = 10.0) -> List[Dict[str, Any]]:
        """Find trading opportunities on blockchain markets."""
        opportunities = []
        
        with db_manager.get_db_session() as db:
            # Get active blockchain markets with odds
            active_markets = db.query(Market).filter(
                Market.source == 'blockchain',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).all()
            
            for market in active_markets:
                # Get odds for this market
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.source == 'blockchain'
                ).all()
                
                if not odds:
                    continue
                    
                # Calculate if there's value
                for odd in odds:
                    if odd.decimal_odds > max_odds:
                        continue
                        
                    # Simple edge calculation
                    # In practice, would use signals/models
                    implied_prob = 1 / odd.decimal_odds
                    fair_prob = self._estimate_fair_probability(market, odd)
                    
                    edge = fair_prob - implied_prob
                    
                    if edge >= min_edge:
                        opportunities.append({
                            'market': market,
                            'odd': odd,
                            'edge': edge,
                            'fair_prob': fair_prob,
                            'implied_prob': implied_prob,
                            'recommended_stake': self._calculate_kelly_stake(edge, odd.decimal_odds)
                        })
                        
        return sorted(opportunities, key=lambda x: x['edge'], reverse=True)
        
    def _estimate_fair_probability(self, market: Market, odd: Odd) -> float:
        """Estimate fair probability for a market outcome."""
        # Simplified - in practice would use signals
        # For now, just add a small edge to favorites
        implied = 1 / odd.decimal_odds
        
        if implied > 0.5:  # Favorite
            return implied + 0.02
        else:  # Underdog
            return implied - 0.01
            
    def _calculate_kelly_stake(self, edge: float, odds: float, 
                              kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly stake size."""
        if edge <= 0:
            return 0
            
        p = edge + (1 / odds)  # Fair probability
        q = 1 - p
        b = odds - 1
        
        kelly = (p * b - q) / b
        return max(0, kelly * kelly_fraction)


def demonstrate_blockchain_integration():
    """Demonstrate blockchain integration features."""
    print("=== Blockchain Integration Demo ===\n")
    
    # Initialize
    integration = BlockchainIntegration(network='optimism', paper_trading_mode=True)
    
    # 1. Test connection
    print("1. Testing blockchain connection...")
    if integration.reader.check_connection():
        print("   ✓ Connected to Optimism")
    else:
        print("   ✗ Connection failed")
        return
        
    # 2. Sync recent markets
    print("\n2. Syncing recent markets...")
    try:
        count = integration.sync_blockchain_markets(hours_back=24)
        print(f"   ✓ Synced {count} new markets")
    except Exception as e:
        print(f"   ✗ Sync failed: {e}")
        
    # 3. Update odds
    print("\n3. Updating odds...")
    try:
        count = integration.update_blockchain_odds()
        print(f"   ✓ Updated odds for {count} markets")
    except Exception as e:
        print(f"   ✗ Update failed: {e}")
        
    # 4. Find opportunities
    print("\n4. Finding trading opportunities...")
    opportunities = integration.evaluate_blockchain_opportunities()
    
    if opportunities:
        print(f"   Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:3]):  # Show top 3
            print(f"\n   #{i+1}: {opp['market'].home_team} vs {opp['market'].away_team}")
            print(f"        Outcome: {opp['odd'].outcome}")
            print(f"        Edge: {opp['edge']:.2%}")
            print(f"        Odds: {opp['odd'].decimal_odds}")
            print(f"        Kelly stake: {opp['recommended_stake']:.2%}")
    else:
        print("   No opportunities found")
        
    # 5. Demo trade execution (paper mode)
    if opportunities and integration.paper_trading_mode:
        print("\n5. Demo trade execution (paper mode):")
        opp = opportunities[0]
        
        position = BlockchainPosition(
            market_address=opp['market'].source_id,
            position=opp['odd'].position,
            amount=Decimal('100'),  # $100 bet
            expected_odds=Decimal(str(opp['odd'].decimal_odds))
        )
        
        print(f"   Would execute: ${position.amount} on {opp['odd'].outcome}")
        print(f"   Expected odds: {position.expected_odds}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_blockchain_integration()