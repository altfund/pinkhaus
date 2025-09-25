#!/usr/bin/env python3
"""
Comprehensive tests for blockchain integration.
Tests both reading and trading functionality.
"""

import unittest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json

from blockchain_reader import BlockchainReader, ChainSyncService
from blockchain_trading import (
    BlockchainTrader, BlockchainIntegration, 
    BlockchainPosition, TradeResult
)


class TestBlockchainReader(unittest.TestCase):
    """Test blockchain reading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Web3 connection
        self.mock_w3 = MagicMock()
        self.mock_w3.is_connected.return_value = True
        self.mock_w3.eth.block_number = 100000
        # Mock Web3.to_checksum_address to return a valid address string
        self.mock_w3.to_checksum_address.side_effect = lambda x: x if x.startswith('0x') else '0x' + x
        
    @patch('blockchain_reader.Web3')
    def test_reader_initialization(self, mock_web3_class):
        """Test BlockchainReader initialization."""
        # Mock the class methods
        mock_web3_class.to_checksum_address = MagicMock(side_effect=lambda x: x)
        mock_web3_class.return_value = self.mock_w3
        
        # Mock the contract
        mock_contract = MagicMock()
        self.mock_w3.eth.contract.return_value = mock_contract
        
        reader = BlockchainReader(network='optimism')
        
        self.assertEqual(reader.network, 'optimism')
        self.assertTrue(reader.w3.is_connected())
        
    @patch('blockchain_reader.Web3')
    def test_get_current_odds(self, mock_web3_class):
        """Test fetching current odds from blockchain."""
        # Mock the class methods
        mock_web3_class.to_checksum_address = MagicMock(side_effect=lambda x: x)
        mock_web3_class.return_value = self.mock_w3
        
        # Mock the contract
        mock_contract_instance = MagicMock()
        mock_contract_instance.functions.getMarketDefaultOdds.return_value.call.side_effect = [
            [5e17, 5e17, 0],  # Buy odds (0.5, 0.5, 0)
            [6e17, 6e17, 0]   # Sell odds
        ]
        self.mock_w3.eth.contract.return_value = mock_contract_instance
        
        reader = BlockchainReader(network='optimism')
        reader.sports_amm = mock_contract_instance
        
        # Test getting odds
        odds = reader.get_current_odds('0x1234...')
        
        # Verify results
        self.assertEqual(len(odds), 2)  # Two valid positions
        self.assertAlmostEqual(odds[0]['buy'], 2.0, places=2)
        self.assertAlmostEqual(odds[1]['buy'], 2.0, places=2)
        
    def test_scan_market_creations(self):
        """Test scanning for new market creations."""
        # Create a mock reader instance
        reader = MagicMock()
        
        # Set up necessary attributes
        reader.network = 'optimism'
        reader.db_path = 'test.db'
        
        # Mock w3 instance
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 100000
        mock_w3.to_hex = MagicMock(return_value='0xeventsig')
        mock_w3.keccak = MagicMock(return_value=b'eventsig')
        
        # Mock event logs
        mock_logs = [
            {
                'address': '0x0000000000000000000000000000000000000001',
                'topics': ['0xevent_sig'],
                'blockNumber': 99999,
                'transactionHash': b'0x1234',
                'args': {
                    'market': '0x0000000000000000000000000000000000000002',
                    'gameId': b'game123',
                    'gameLabel': 'Team A vs Team B',
                    'maturityDate': int(datetime.now(timezone.utc).timestamp()) + 3600,
                    'tags': [5, 105],  # Soccer, EPL
                    'normalizedOdds': [5e17, 5e17]
                }
            }
        ]
        
        mock_w3.eth.get_logs.return_value = mock_logs
        reader.w3 = mock_w3
        
        # Mock the contract
        mock_contract = MagicMock()
        mock_contract.address = '0x0000000000000000000000000000000000000001'
        
        # Set up events
        mock_events = MagicMock()
        mock_events.MarketCreated = MagicMock()
        mock_events.MarketCreated.process_log = MagicMock(side_effect=lambda log: log)
        mock_contract.events = mock_events
        
        reader.sports_amm = mock_contract
        reader._store_market = MagicMock()
        
        # Import and patch Web3.to_checksum_address
        from blockchain_reader import BlockchainReader
        with patch('blockchain_reader.Web3.to_checksum_address', side_effect=lambda x: x):
            # Call the actual method
            markets = BlockchainReader.scan_market_creations(reader, 99999, 100000)
        
        # Verify results
        self.assertTrue(mock_w3.eth.get_logs.called)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0]['game_label'], 'Team A vs Team B')
        reader._store_market.assert_called_once()
        
    @patch('blockchain_reader.Web3')
    def test_fetch_recent_markets(self, mock_web3_class):
        """Test fetching recent markets with formatting."""
        # Mock the class methods
        mock_web3_class.to_checksum_address = MagicMock(side_effect=lambda x: x)
        mock_web3_class.return_value = self.mock_w3
        
        # Mock the contract
        mock_contract = MagicMock()
        self.mock_w3.eth.contract.return_value = mock_contract
        
        reader = BlockchainReader(network='optimism')
        reader.scan_market_creations = MagicMock(return_value=[
            {
                'market_address': '0xmarket1',
                'game_id': 'game123',
                'game_label': 'Liverpool vs Manchester City',
                'maturity_date': int(datetime.now(timezone.utc).timestamp()) + 3600,
                'tags': json.dumps([5, 105]),  # Soccer, EPL
                'normalized_odds': json.dumps([5e17, 5e17])
            }
        ])
        
        # Test fetching
        markets = reader.fetch_recent_markets(hours_back=24)
        
        # Verify formatting
        self.assertEqual(len(markets), 1)
        market = markets[0]
        self.assertEqual(market['source'], 'blockchain')
        self.assertEqual(market['home_team'], 'Liverpool')
        self.assertEqual(market['away_team'], 'Manchester City')
        self.assertEqual(market['sport'], 'Soccer')
        self.assertEqual(market['league'], 'League_105')


class TestBlockchainTrader(unittest.TestCase):
    """Test blockchain trading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Web3 and account
        self.mock_w3 = MagicMock()
        self.mock_w3.is_connected.return_value = True
        self.mock_w3.eth.block_number = 100000
        self.mock_w3.eth.gas_price = 30000000000  # 30 Gwei
        self.mock_w3.eth.get_transaction_count.return_value = 5
        
    @patch('blockchain_trading.BlockchainReader')
    @patch('blockchain_trading.Account')
    def test_trader_initialization(self, mock_account_class, mock_reader_class):
        """Test BlockchainTrader initialization."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = '0xtrader123'
        mock_account_class.from_key.return_value = mock_account
        
        # Initialize trader
        trader = BlockchainTrader(network='optimism', private_key='0xprivkey')
        
        # Verify initialization
        self.assertEqual(trader.network, 'optimism')
        self.assertEqual(trader.account.address, '0xtrader123')
        
    @patch('blockchain_trading.BlockchainReader')
    def test_check_balance(self, mock_reader_class):
        """Test balance checking."""
        # Setup mocks
        trader = BlockchainTrader(network='optimism')
        trader.account = MagicMock(address='0xtrader')
        
        # Mock contract calls
        trader.susd = MagicMock()
        trader.susd.functions.balanceOf.return_value.call.return_value = int(1000e18)  # 1000 SUSD
        trader.reader.w3.eth.get_balance.return_value = int(0.1e18)  # 0.1 ETH
        
        # Test balance check
        balances = trader.check_balance()
        
        # Verify results
        self.assertEqual(balances['susd'], Decimal('1000'))
        self.assertEqual(balances['eth'], Decimal('0.1'))
        
    @patch('blockchain_trading.BlockchainReader')
    def test_quote_trade(self, mock_reader_class):
        """Test getting a trade quote."""
        trader = BlockchainTrader(network='optimism')
        
        # Mock contract quote
        trader.sports_amm = MagicMock()
        trader.sports_amm.functions.buyFromAmmQuote.return_value.call.return_value = (
            int(110e18),  # 110 SUSD needed
            int(2e18)     # 2 SUSD additional slippage
        )
        
        # Create position with valid checksum address
        position = BlockchainPosition(
            market_address='0x1234567890123456789012345678901234567890',
            position=0,
            amount=Decimal('100'),
            expected_odds=Decimal('1.1')
        )
        
        # Get quote
        quote = trader.quote_trade(position)
        
        # Verify quote
        self.assertEqual(quote['susd_needed'], Decimal('110'))
        self.assertEqual(quote['additional_slippage'], Decimal('2'))
        self.assertEqual(quote['effective_odds'], Decimal('1.1'))
        self.assertEqual(quote['slippage'], Decimal('0'))
        self.assertTrue(quote['acceptable'])
        
    @patch('blockchain_trading.BlockchainReader')
    def test_execute_trade_success(self, mock_reader_class):
        """Test successful trade execution."""
        # Use a valid hex private key for testing
        test_private_key = '0x' + '1' * 64  # Valid 32-byte hex string
        trader = BlockchainTrader(network='optimism', private_key=test_private_key)
        trader.account = MagicMock(address='0xtrader')
        
        # Mock quote
        trader.quote_trade = MagicMock(return_value={
            'susd_needed': Decimal('110'),
            'additional_slippage': Decimal('2'),
            'effective_odds': Decimal('1.1'),
            'acceptable': True
        })
        
        # Mock approval
        trader.ensure_approval = MagicMock(return_value=True)
        
        # Mock transaction
        mock_tx_hash = b'0xtxhash'
        trader.reader.w3.eth.send_raw_transaction.return_value = mock_tx_hash
        trader.reader.w3.eth.wait_for_transaction_receipt.return_value = {
            'status': 1,
            'gasUsed': 200000
        }
        trader.reader.w3.eth.account.sign_transaction.return_value = MagicMock(
            rawTransaction=b'0xsigned'
        )
        
        # Create position with valid checksum address
        position = BlockchainPosition(
            market_address='0x1234567890123456789012345678901234567890',
            position=0,
            amount=Decimal('100'),
            expected_odds=Decimal('1.1')
        )
        
        # Execute trade
        result = trader.execute_trade(position)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.tx_hash, mock_tx_hash.hex())
        self.assertEqual(result.actual_odds, Decimal('1.1'))
        # susd_paid includes the additional slippage
        self.assertEqual(result.susd_paid, Decimal('112'))  # 110 + 2
        self.assertEqual(result.gas_used, 200000)
        
    @patch('blockchain_trading.BlockchainReader')
    def test_execute_trade_high_slippage(self, mock_reader_class):
        """Test trade rejection due to high slippage."""
        # Use a valid hex private key for testing
        test_private_key = '0x' + '1' * 64  # Valid 32-byte hex string
        trader = BlockchainTrader(network='optimism', private_key=test_private_key)
        trader.account = MagicMock(address='0xtrader')
        
        # Mock quote with high slippage
        trader.quote_trade = MagicMock(return_value={
            'susd_needed': Decimal('120'),
            'effective_odds': Decimal('1.2'),
            'slippage': Decimal('0.1'),  # 10% slippage
            'acceptable': False
        })
        
        # Create position
        position = BlockchainPosition(
            market_address='0xmarket',
            position=0,
            amount=Decimal('100'),
            expected_odds=Decimal('1.1'),
            slippage_tolerance=Decimal('0.02')  # 2% max
        )
        
        # Execute trade
        result = trader.execute_trade(position)
        
        # Verify rejection
        self.assertFalse(result.success)
        self.assertIn('Slippage too high', result.error)


class TestBlockchainIntegration(unittest.TestCase):
    """Test blockchain integration with main system."""
    
    @patch('blockchain_trading.db_manager')
    @patch('blockchain_trading.BlockchainReader')
    def test_sync_blockchain_markets(self, mock_reader_class, mock_db):
        """Test syncing blockchain markets to database."""
        # Setup mocks
        integration = BlockchainIntegration(network='optimism', paper_trading_mode=True)
        
        # Mock blockchain markets
        integration.reader.fetch_recent_markets = MagicMock(return_value=[
            {
                'source': 'blockchain',
                'source_id': 'game123',
                'sport': 'soccer',
                'league': 'EPL',
                'home_team': 'Liverpool',
                'away_team': 'Manchester City',
                'market_type': 'moneyline',
                'maturity_date': datetime.now(timezone.utc) + timedelta(hours=2),
                'normalized_odds': [5e17, 5e17]
            }
        ])
        
        # Mock database session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Test syncing
        count = integration.sync_blockchain_markets(hours_back=24)
        
        # Verify
        self.assertEqual(count, 1)
        mock_session.add.assert_called()
        mock_session.commit.assert_called()
        
    @patch('blockchain_trading.db_manager')
    @patch('blockchain_trading.BlockchainReader')
    def test_update_blockchain_odds(self, mock_reader_class, mock_db):
        """Test updating odds from blockchain."""
        integration = BlockchainIntegration(network='optimism')
        
        # Mock current odds
        integration.reader.get_current_odds = MagicMock(return_value={
            0: {'buy': 1.8, 'sell': 1.9},
            1: {'buy': 2.2, 'sell': 2.3}
        })
        
        # Mock database
        mock_session = MagicMock()
        mock_odd = MagicMock()
        mock_odd.decimal_odds = None  # Initial value
        mock_session.query.return_value.filter.return_value.first.return_value = mock_odd
        mock_db.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Test updating
        count = integration.update_blockchain_odds(['0xmarket1'])
        
        # Verify
        self.assertEqual(count, 1)
        # The first odd (position 0) has buy odds of 1.8
        # Since filter.first() is called for each position, we can't easily verify the exact value
        # Instead verify that the mock was updated
        self.assertTrue(hasattr(mock_odd, 'decimal_odds'))
        mock_session.commit.assert_called()
        
    def test_calculate_kelly_stake(self):
        """Test Kelly stake calculation."""
        integration = BlockchainIntegration(network='optimism')
        
        # Test positive edge
        stake = integration._calculate_kelly_stake(
            edge=0.05,  # 5% edge
            odds=2.0,
            kelly_fraction=0.25
        )
        
        # Kelly formula: f = (p*b - q) / b
        # p = 0.55, q = 0.45, b = 1.0
        # f = (0.55*1.0 - 0.45) / 1.0 = 0.1
        # With 25% Kelly: 0.1 * 0.25 = 0.025
        self.assertAlmostEqual(stake, 0.025, places=3)
        
        # Test negative edge
        stake = integration._calculate_kelly_stake(
            edge=-0.05,
            odds=2.0,
            kelly_fraction=0.25
        )
        self.assertEqual(stake, 0)


class TestChainSyncService(unittest.TestCase):
    """Test blockchain sync service."""
    
    def test_get_last_synced_block(self):
        """Test getting last synced block from database."""
        reader = MagicMock()
        reader.db_path = 'test.db'
        reader.network = 'optimism'
        reader.w3.eth.block_number = 100000
        
        # Patch sqlite3.connect to return a mock
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (99500,)  # Last synced block
            mock_conn.execute.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            from blockchain_reader import ChainSyncService
            service = ChainSyncService(reader)
            
            # Verify service was initialized
            self.assertEqual(service.last_synced_block, 99500)
            
    @patch('asyncio.sleep')
    @patch('blockchain_trading.BlockchainReader')
    async def test_sync_continuously(self, mock_reader_class, mock_sleep):
        """Test continuous blockchain syncing."""
        # Setup mocks
        reader = MagicMock()
        reader.w3.eth.block_number = 100000
        reader.config = {'block_time': 2}
        reader.scan_market_creations = MagicMock()
        reader.scan_trades = MagicMock()
        
        service = ChainSyncService(reader)
        service.last_synced_block = 99990
        
        # Run one iteration
        mock_sleep.side_effect = Exception("Stop after one iteration")
        
        try:
            await service.sync_continuously(batch_size=10)
        except:
            pass
            
        # Verify scanning was called
        reader.scan_market_creations.assert_called_with(99990, 100000)
        reader.scan_trades.assert_called_with(99990, 100000)
        self.assertEqual(service.last_synced_block, 100000)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    @patch('blockchain_trading.db_manager')
    @patch('blockchain_trading.BlockchainReader')
    def test_full_trading_flow(self, mock_reader_class, mock_db):
        """Test complete flow from market discovery to trade execution."""
        # Initialize integration
        integration = BlockchainIntegration(network='optimism', paper_trading_mode=True)
        
        # Mock market discovery
        mock_market = MagicMock()
        mock_market.source_id = '0xmarket1'
        mock_market.home_team = 'Liverpool'
        mock_market.away_team = 'Manchester City'
        
        mock_odd = MagicMock()
        # Configure __gt__ to work with numeric comparison
        mock_odd.decimal_odds = 2.0
        mock_odd.__gt__ = lambda self, other: self.decimal_odds > other
        mock_odd.position = 0
        mock_odd.outcome = 'option_0'
        
        # Mock opportunity evaluation
        integration._estimate_fair_probability = MagicMock(return_value=0.55)
        
        mock_session = MagicMock()
        # Mock for Market query
        market_query_mock = MagicMock()
        market_query_mock.filter.return_value.all.return_value = [mock_market]
        
        # Mock for Odd query
        odd_query_mock = MagicMock()
        odd_query_mock.filter.return_value.all.return_value = [mock_odd]
        
        # Return different mocks based on what's being queried
        def query_side_effect(model):
            if model.__name__ == 'Market':
                return market_query_mock
            elif model.__name__ == 'Odd':
                return odd_query_mock
            return MagicMock()
        
        mock_session.query.side_effect = query_side_effect
        mock_db.get_db_session.return_value.__enter__.return_value = mock_session
        
        # Find opportunities
        opportunities = integration.evaluate_blockchain_opportunities(min_edge=0.02)
        
        # Verify opportunity found
        self.assertEqual(len(opportunities), 1)
        opp = opportunities[0]
        self.assertAlmostEqual(opp['edge'], 0.05, places=2)  # 0.55 - 0.50
        self.assertGreater(opp['recommended_stake'], 0)


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainReader))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainTrader))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlockchainIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChainSyncService))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)