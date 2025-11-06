"""
Web3 integration layer for interacting with Ominari smart contracts
"""
import os
import json
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account.messages import encode_defunct
import logging

logger = logging.getLogger(__name__)


class OminariBlockchainClient:
    """Client for interacting with Ominari smart contracts"""
    
    def __init__(self, chain_network):
        """Initialize with a ChainNetwork model instance"""
        self.chain = chain_network
        self.w3 = self._init_web3()
        self.contracts = self._init_contracts()
    
    def _init_web3(self):
        """Initialize Web3 connection"""
        w3 = Web3(Web3.HTTPProvider(self.chain.rpc_url))
        
        # Add POA middleware for networks like Polygon
        if self.chain.name in ['polygon', 'mumbai']:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.chain.name}")
        
        logger.info(f"Connected to {self.chain.name} at block {w3.eth.block_number}")
        return w3
    
    def _init_contracts(self):
        """Initialize smart contract instances"""
        contracts = {}
        
        # Load ABIs (in production, these would be loaded from files)
        trading_engine_abi = self._load_abi('OminariTradingEngine')
        kelly_optimizer_abi = self._load_abi('KellyOptimizer')
        chunk_manager_abi = self._load_abi('ChunkManager')
        
        # Initialize contract instances
        if self.chain.trading_engine_address:
            contracts['trading_engine'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.chain.trading_engine_address),
                abi=trading_engine_abi
            )
        
        if self.chain.kelly_optimizer_address:
            contracts['kelly_optimizer'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.chain.kelly_optimizer_address),
                abi=kelly_optimizer_abi
            )
        
        if self.chain.chunk_manager_address:
            contracts['chunk_manager'] = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.chain.chunk_manager_address),
                abi=chunk_manager_abi
            )
        
        return contracts
    
    def _load_abi(self, contract_name):
        """Load contract ABI from file"""
        # In production, load from compiled artifacts
        # For now, return minimal ABI
        if contract_name == 'OminariTradingEngine':
            return [
                {
                    "name": "createSession",
                    "type": "function",
                    "inputs": [{"name": "initialBankroll", "type": "uint256"}],
                    "outputs": [{"name": "sessionId", "type": "uint256"}]
                },
                {
                    "name": "getSession",
                    "type": "function",
                    "inputs": [{"name": "sessionId", "type": "uint256"}],
                    "outputs": [{"name": "", "type": "tuple", "components": [
                        {"name": "id", "type": "uint256"},
                        {"name": "trader", "type": "address"},
                        {"name": "initialBankroll", "type": "uint256"},
                        {"name": "currentBankroll", "type": "uint256"},
                        {"name": "startTime", "type": "uint256"},
                        {"name": "lastActivityTime", "type": "uint256"},
                        {"name": "isActive", "type": "bool"},
                        {"name": "totalBetsPlaced", "type": "uint256"},
                        {"name": "totalBetsWon", "type": "uint256"},
                        {"name": "totalProfit", "type": "int256"}
                    ]}]
                },
                {
                    "name": "SessionCreated",
                    "type": "event",
                    "inputs": [
                        {"name": "sessionId", "type": "uint256", "indexed": True},
                        {"name": "trader", "type": "address", "indexed": True},
                        {"name": "bankroll", "type": "uint256", "indexed": False}
                    ]
                }
            ]
        return []
    
    def verify_wallet_ownership(self, wallet_address: str, signature: str, message: str) -> bool:
        """Verify that a signature was created by the wallet owner"""
        try:
            # Encode the message
            encoded_message = encode_defunct(text=message)
            
            # Recover the address from the signature
            recovered_address = self.w3.eth.account.recover_message(
                encoded_message, 
                signature=signature
            )
            
            # Check if it matches
            return recovered_address.lower() == wallet_address.lower()
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return False
    
    def get_session_data(self, session_id: int) -> Optional[Dict]:
        """Fetch session data from blockchain"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Call the contract
            session_data = self.contracts['trading_engine'].functions.getSession(session_id).call()
            
            # Parse the response
            return {
                'id': session_data[0],
                'trader': session_data[1],
                'initial_bankroll': Web3.from_wei(session_data[2], 'ether'),
                'current_bankroll': Web3.from_wei(session_data[3], 'ether'),
                'start_time': session_data[4],
                'last_activity_time': session_data[5],
                'is_active': session_data[6],
                'total_bets_placed': session_data[7],
                'total_bets_won': session_data[8],
                'total_profit': Web3.from_wei(session_data[9], 'ether')
            }
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return None
    
    def get_position_data(self, position_id: int) -> Optional[Dict]:
        """Fetch position data from blockchain"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Call the contract
            position_data = self.contracts['trading_engine'].functions.getPosition(position_id).call()
            
            # Parse the response
            return {
                'id': position_data[0],
                'session_id': position_data[1],
                'market_id': position_data[2].hex(),
                'stake': Web3.from_wei(position_data[3], 'ether'),
                'odds': Web3.from_wei(position_data[4], 'ether'),
                'outcome': position_data[5],
                'timestamp': position_data[6],
                'is_settled': position_data[7],
                'is_won': position_data[8],
                'payout': Web3.from_wei(position_data[9], 'ether')
            }
        except Exception as e:
            logger.error(f"Failed to get position data: {e}")
            return None
    
    def get_events(self, event_name: str, from_block: int, to_block: Optional[int] = None) -> List[Dict]:
        """Fetch events from the blockchain"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Get the event filter
            event_filter = getattr(self.contracts['trading_engine'].events, event_name).create_filter(
                fromBlock=from_block,
                toBlock=to_block or 'latest'
            )
            
            # Get all entries
            events = event_filter.get_all_entries()
            
            # Parse events
            parsed_events = []
            for event in events:
                parsed_event = {
                    'transaction_hash': event['transactionHash'].hex(),
                    'block_number': event['blockNumber'],
                    'log_index': event['logIndex'],
                    'args': dict(event['args'])
                }
                parsed_events.append(parsed_event)
            
            return parsed_events
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
    
    def estimate_gas(self, function_name: str, *args, from_address: str) -> int:
        """Estimate gas for a transaction"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Get the function
            function = getattr(self.contracts['trading_engine'].functions, function_name)(*args)
            
            # Estimate gas
            gas = function.estimate_gas({'from': from_address})
            
            # Add 10% buffer
            return int(gas * 1.1)
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            return 100000  # Default gas limit
    
    def build_transaction(self, function_name: str, *args, from_address: str, value: int = 0) -> Dict:
        """Build a transaction for signing"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Get the function
            function = getattr(self.contracts['trading_engine'].functions, function_name)(*args)
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(from_address)
            
            # Build transaction
            transaction = function.build_transaction({
                'from': from_address,
                'value': value,
                'gas': self.estimate_gas(function_name, *args, from_address=from_address),
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
                'chainId': self.chain.chain_id
            })
            
            return transaction
        except Exception as e:
            logger.error(f"Failed to build transaction: {e}")
            raise
    
    def decode_transaction_input(self, tx_input: str) -> Tuple[str, Dict]:
        """Decode transaction input data"""
        try:
            if 'trading_engine' not in self.contracts:
                raise ValueError("Trading engine contract not initialized")
            
            # Decode the function call
            decoded = self.contracts['trading_engine'].decode_function_input(tx_input)
            
            function_name = decoded[0].fn_name
            parameters = decoded[1]
            
            return function_name, parameters
        except Exception as e:
            logger.error(f"Failed to decode transaction: {e}")
            return None, {}


class MockBlockchainClient(OminariBlockchainClient):
    """Mock blockchain client for testing"""
    
    def __init__(self, chain_network):
        self.chain = chain_network
        self.mock_sessions = {}
        self.mock_positions = {}
        self.next_session_id = 1
        self.next_position_id = 1
    
    def verify_wallet_ownership(self, wallet_address: str, signature: str, message: str) -> bool:
        # Always return True in mock mode
        return True
    
    def get_session_data(self, session_id: int) -> Optional[Dict]:
        return self.mock_sessions.get(session_id)
    
    def create_mock_session(self, trader: str, initial_bankroll: Decimal) -> int:
        session_id = self.next_session_id
        self.next_session_id += 1
        
        self.mock_sessions[session_id] = {
            'id': session_id,
            'trader': trader,
            'initial_bankroll': float(initial_bankroll),
            'current_bankroll': float(initial_bankroll),
            'start_time': 1234567890,
            'last_activity_time': 1234567890,
            'is_active': True,
            'total_bets_placed': 0,
            'total_bets_won': 0,
            'total_profit': 0.0
        }
        
        return session_id