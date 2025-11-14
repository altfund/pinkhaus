#!/usr/bin/env python3
"""
Real trading configuration and wallet management
Handles private keys and trading limits securely
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
from web3 import Web3

logger = logging.getLogger(__name__)


class RealTradingConfig:
    """Manages real trading configuration and wallet security"""
    
    def __init__(self, config_path: str = "config/real_trading_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        self.encryption_key = self._get_or_create_encryption_key()
        self.config = self._load_config()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_path = self.config_path.parent / ".trading_key"
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            return key
            
    def _encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()
        
    def _decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "mode": "testnet",  # testnet or mainnet
                "networks": {
                    "arbitrum": {
                        "rpc_url": "https://arb1.arbitrum.io/rpc",
                        "chain_id": 42161,
                        "gas_limit": 3000000,
                        "max_gas_price_gwei": 5.0
                    },
                    "optimism": {
                        "rpc_url": "https://mainnet.optimism.io",
                        "chain_id": 10,
                        "gas_limit": 3000000,
                        "max_gas_price_gwei": 5.0
                    },
                    "base": {
                        "rpc_url": "https://mainnet.base.org",
                        "chain_id": 8453,
                        "gas_limit": 3000000,
                        "max_gas_price_gwei": 5.0
                    }
                },
                "default_network": "arbitrum",
                "safety_limits": {
                    "max_bet_size_usd": 100.0,
                    "max_daily_loss_usd": 500.0,
                    "max_concurrent_bets": 10,
                    "max_exposure_pct": 25.0,
                    "min_edge_required": 3.0,
                    "require_2fa": True,
                    "emergency_stop": False
                },
                "wallet": {
                    "address": None,
                    "encrypted_private_key": None
                },
                "approved_contracts": {
                    "arbitrum": {
                        "sports_amm_v2": "0xfb64E79A562F7250131cf528242CEB10fDC82395",
                        "collateral": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"  # USDC
                    },
                    "optimism": {
                        "sports_amm_v2": "0x170a5714112daEfF20E798B6e92e25B86Ea603C1",
                        "collateral": "0x8c6f28f2F1A3C87F0f938b96d27520d9751ec8d9"  # sUSD
                    },
                    "base": {
                        "sports_amm_v2": "0xC3E7f5a2548c446555bb3D99EdE57e73b02fEb58",
                        "collateral": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC
                    }
                }
            }
            self._save_config(default_config)
            return default_config
            
    def _save_config(self, config: Optional[Dict] = None):
        """Save configuration to file"""
        if config:
            self.config = config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        # Set restrictive permissions
        os.chmod(self.config_path, 0o600)
        
    def set_wallet(self, private_key: str) -> str:
        """Set wallet private key (encrypted)"""
        # Validate private key
        try:
            account = Web3().eth.account.from_key(private_key)
            address = account.address
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")
            
        # Encrypt and store
        encrypted_key = self._encrypt(private_key)
        self.config['wallet']['address'] = address
        self.config['wallet']['encrypted_private_key'] = encrypted_key
        self._save_config()
        
        logger.info(f"Wallet configured: {address}")
        return address
        
    def get_wallet_address(self) -> Optional[str]:
        """Get configured wallet address"""
        return self.config['wallet'].get('address')
        
    def get_private_key(self) -> Optional[str]:
        """Get decrypted private key (use with caution)"""
        encrypted = self.config['wallet'].get('encrypted_private_key')
        if encrypted:
            return self._decrypt(encrypted)
        return None
        
    def is_configured(self) -> bool:
        """Check if wallet is configured"""
        return bool(self.config['wallet'].get('encrypted_private_key'))
        
    def get_network_config(self, network: Optional[str] = None) -> Dict:
        """Get network configuration"""
        network = network or self.config['default_network']
        return self.config['networks'].get(network, {})
        
    def get_safety_limits(self) -> Dict:
        """Get current safety limits"""
        return self.config['safety_limits'].copy()
        
    def update_safety_limits(self, limits: Dict):
        """Update safety limits"""
        self.config['safety_limits'].update(limits)
        self._save_config()
        
    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is activated"""
        return self.config['safety_limits'].get('emergency_stop', False)
        
    def set_emergency_stop(self, stop: bool):
        """Set emergency stop status"""
        self.config['safety_limits']['emergency_stop'] = stop
        self._save_config()
        logger.warning(f"Emergency stop {'ACTIVATED' if stop else 'DEACTIVATED'}")
        
    def get_approved_contracts(self, network: Optional[str] = None) -> Dict:
        """Get approved contract addresses for network"""
        network = network or self.config['default_network']
        return self.config['approved_contracts'].get(network, {})
        
    def switch_mode(self, mode: str):
        """Switch between testnet and mainnet"""
        if mode not in ['testnet', 'mainnet']:
            raise ValueError("Mode must be 'testnet' or 'mainnet'")
        self.config['mode'] = mode
        self._save_config()
        logger.info(f"Switched to {mode} mode")
        
    def get_mode(self) -> str:
        """Get current mode (testnet/mainnet)"""
        return self.config.get('mode', 'testnet')
        
    def validate_trade(self, amount_usd: float, current_exposure_usd: float,
                      daily_loss_usd: float, edge: float) -> tuple[bool, str]:
        """Validate if a trade meets safety criteria"""
        limits = self.get_safety_limits()
        
        # Check emergency stop
        if self.is_emergency_stopped():
            return False, "Emergency stop is active"
            
        # Check bet size
        if amount_usd > limits['max_bet_size_usd']:
            return False, f"Bet size ${amount_usd} exceeds max ${limits['max_bet_size_usd']}"
            
        # Check daily loss
        if daily_loss_usd >= limits['max_daily_loss_usd']:
            return False, f"Daily loss limit reached: ${daily_loss_usd}"
            
        # Check edge requirement
        if edge < limits['min_edge_required']:
            return False, f"Edge {edge}% below minimum {limits['min_edge_required']}%"
            
        # Check exposure
        total_exposure = current_exposure_usd + amount_usd
        if total_exposure > (limits['max_exposure_pct'] / 100) * self.get_total_bankroll():
            return False, f"Would exceed exposure limit"
            
        return True, "Trade approved"
        
    def get_total_bankroll(self) -> float:
        """Get total bankroll from wallet balance"""
        # This would check actual wallet balance
        # For now return a configured amount
        return self.config.get('bankroll', 10000.0)


def main():
    """Test configuration"""
    config = RealTradingConfig()
    
    print(f"Mode: {config.get_mode()}")
    print(f"Configured: {config.is_configured()}")
    print(f"Safety Limits: {json.dumps(config.get_safety_limits(), indent=2)}")
    
    # Test trade validation
    valid, reason = config.validate_trade(50, 100, 50, 5.0)
    print(f"Trade validation: {valid} - {reason}")
    

if __name__ == "__main__":
    main()