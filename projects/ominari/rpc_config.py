#!/usr/bin/env python3
"""
RPC Configuration Manager for Blockchain Access

Manages RPC endpoints with fallback options and automatic failover.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from web3 import Web3
from web3.providers import HTTPProvider
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class RPCEndpoint:
    """RPC endpoint configuration."""
    name: str
    url: str
    priority: int = 0  # Lower is higher priority
    rate_limit: Optional[int] = None  # Requests per second
    last_used: float = 0
    failures: int = 0
    last_failure: float = 0
    is_healthy: bool = True


class RPCManager:
    """Manages multiple RPC endpoints with automatic failover."""
    
    # Free tier endpoints (use as fallback)
    FREE_ENDPOINTS = {
        'optimism': [
            'https://mainnet.optimism.io',
            'https://rpc.ankr.com/optimism',
            'https://optimism-mainnet.public.blastapi.io',
            'https://1rpc.io/op',
        ],
        'arbitrum': [
            'https://arb1.arbitrum.io/rpc',
            'https://rpc.ankr.com/arbitrum',
            'https://arbitrum-mainnet.public.blastapi.io',
            'https://1rpc.io/arb',
        ],
        'optimism_sepolia': [
            'https://sepolia.optimism.io',
            'https://optimism-sepolia.public.blastapi.io',
        ],
        'arbitrum_sepolia': [
            'https://sepolia-rollup.arbitrum.io/rpc',
            'https://arbitrum-sepolia.public.blastapi.io',
        ]
    }
    
    # Premium providers (need API keys)
    PREMIUM_PROVIDERS = {
        'alchemy': {
            'optimism': 'https://opt-mainnet.g.alchemy.com/v2/{api_key}',
            'arbitrum': 'https://arb-mainnet.g.alchemy.com/v2/{api_key}',
            'optimism_sepolia': 'https://opt-sepolia.g.alchemy.com/v2/{api_key}',
            'arbitrum_sepolia': 'https://arb-sepolia.g.alchemy.com/v2/{api_key}',
        },
        'infura': {
            'optimism': 'https://optimism-mainnet.infura.io/v3/{api_key}',
            'arbitrum': 'https://arbitrum-mainnet.infura.io/v3/{api_key}',
            'optimism_sepolia': 'https://optimism-sepolia.infura.io/v3/{api_key}',
            'arbitrum_sepolia': 'https://arbitrum-sepolia.infura.io/v3/{api_key}',
        },
        'quicknode': {
            'optimism': 'https://opt-mainnet.quicknode.pro/{api_key}',
            'arbitrum': 'https://arb-mainnet.quicknode.pro/{api_key}',
        }
    }
    
    def __init__(self, network: str = 'optimism', config_file: Optional[str] = None):
        self.network = network
        self.endpoints: List[RPCEndpoint] = []
        self.current_index = 0
        self.config_file = config_file or '.rpc_config.json'
        
        # Load configuration
        self._load_config()
        
        # Initialize endpoints
        self._init_endpoints()
        
    def _load_config(self):
        """Load RPC configuration from file or environment."""
        config = {}
        
        # Try loading from file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
        
        # Check environment variables
        env_mappings = {
            'ALCHEMY_API_KEY': 'alchemy',
            'INFURA_API_KEY': 'infura',
            'QUICKNODE_API_KEY': 'quicknode',
            f'{self.network.upper()}_RPC_URL': 'custom',
        }
        
        for env_var, provider in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if provider == 'custom':
                    config['custom_endpoints'] = config.get('custom_endpoints', [])
                    config['custom_endpoints'].append(value)
                else:
                    config[f'{provider}_api_key'] = value
        
        self.config = config
        
    def _init_endpoints(self):
        """Initialize RPC endpoints based on configuration."""
        priority = 0
        
        # Add custom endpoints first (highest priority)
        for url in self.config.get('custom_endpoints', []):
            self.endpoints.append(RPCEndpoint(
                name='custom',
                url=url,
                priority=priority
            ))
            priority += 1
        
        # Add premium endpoints
        for provider, templates in self.PREMIUM_PROVIDERS.items():
            api_key = self.config.get(f'{provider}_api_key')
            if api_key and self.network in templates:
                url = templates[self.network].format(api_key=api_key)
                self.endpoints.append(RPCEndpoint(
                    name=provider,
                    url=url,
                    priority=priority,
                    rate_limit=self._get_rate_limit(provider)
                ))
                priority += 1
        
        # Add free endpoints as fallback
        if self.network in self.FREE_ENDPOINTS:
            for url in self.FREE_ENDPOINTS[self.network]:
                self.endpoints.append(RPCEndpoint(
                    name='free',
                    url=url,
                    priority=priority,
                    rate_limit=1  # Conservative rate limit for free endpoints
                ))
                priority += 1
        
        if not self.endpoints:
            raise ValueError(f"No RPC endpoints configured for {self.network}")
        
        logger.info(f"Initialized {len(self.endpoints)} RPC endpoints for {self.network}")
        
    def _get_rate_limit(self, provider: str) -> Optional[int]:
        """Get rate limit for provider."""
        # Conservative defaults for free tiers
        limits = {
            'alchemy': 25,  # Free tier
            'infura': 10,   # Free tier
            'quicknode': 25,  # Free tier
        }
        return limits.get(provider)
        
    def get_web3(self) -> Tuple[Web3, RPCEndpoint]:
        """Get a working Web3 instance with the best available endpoint."""
        # Sort by priority and health
        available = sorted(
            [ep for ep in self.endpoints if ep.is_healthy],
            key=lambda x: (x.priority, x.failures)
        )
        
        if not available:
            # All endpoints failed, reset and try again
            logger.warning("All endpoints failed, resetting...")
            for ep in self.endpoints:
                ep.is_healthy = True
                ep.failures = 0
            available = self.endpoints
        
        # Try each endpoint
        for endpoint in available:
            try:
                # Check rate limit
                if endpoint.rate_limit:
                    time_since_last = time.time() - endpoint.last_used
                    min_interval = 1.0 / endpoint.rate_limit
                    if time_since_last < min_interval:
                        continue
                
                # Create Web3 instance
                w3 = Web3(HTTPProvider(endpoint.url))
                
                # Test connection
                if w3.is_connected():
                    block = w3.eth.block_number
                    if block > 0:
                        endpoint.last_used = time.time()
                        endpoint.failures = 0
                        return w3, endpoint
                        
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint.name}: {e}")
                endpoint.failures += 1
                endpoint.last_failure = time.time()
                
                # Mark as unhealthy after 3 failures
                if endpoint.failures >= 3:
                    endpoint.is_healthy = False
        
        raise ConnectionError(f"No working RPC endpoints available for {self.network}")
        
    def test_all_endpoints(self) -> Dict[str, bool]:
        """Test all configured endpoints."""
        results = {}
        
        for endpoint in self.endpoints:
            try:
                w3 = Web3(HTTPProvider(endpoint.url))
                if w3.is_connected():
                    block = w3.eth.block_number
                    results[f"{endpoint.name} ({endpoint.url[:30]}...)"] = True
                    logger.info(f"✅ {endpoint.name}: Connected (block {block})")
                else:
                    results[f"{endpoint.name} ({endpoint.url[:30]}...)"] = False
                    logger.warning(f"❌ {endpoint.name}: Not connected")
            except Exception as e:
                results[f"{endpoint.name} ({endpoint.url[:30]}...)"] = False
                logger.error(f"❌ {endpoint.name}: {e}")
                
        return results
        
    def save_config(self):
        """Save current configuration to file."""
        config = {
            'network': self.network,
            'custom_endpoints': [
                ep.url for ep in self.endpoints if ep.name == 'custom'
            ],
            'api_keys': {
                provider: self.config.get(f'{provider}_api_key', '')
                for provider in ['alchemy', 'infura', 'quicknode']
                if self.config.get(f'{provider}_api_key')
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved RPC configuration to {self.config_file}")


def create_rpc_config():
    """Create initial RPC configuration."""
    print("\n🔧 RPC Configuration Setup")
    print("=" * 50)
    print("\nThis will help you configure blockchain RPC endpoints.")
    print("\nOptions:")
    print("1. Use free public endpoints (rate limited)")
    print("2. Configure premium provider (Alchemy/Infura/QuickNode)")
    print("3. Use custom RPC endpoint")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    config = {}
    
    if choice == '2':
        print("\n📝 Premium Provider Setup")
        print("\nGet free API keys from:")
        print("- Alchemy: https://alchemy.com")
        print("- Infura: https://infura.io")
        print("- QuickNode: https://quicknode.com")
        
        provider = input("\nProvider (alchemy/infura/quicknode): ").strip().lower()
        api_key = input(f"Enter {provider} API key: ").strip()
        
        if provider in ['alchemy', 'infura', 'quicknode'] and api_key:
            config[f'{provider}_api_key'] = api_key
            print(f"✅ Configured {provider}")
            
    elif choice == '3':
        print("\n🔗 Custom RPC Setup")
        url = input("Enter RPC URL: ").strip()
        if url:
            config['custom_endpoints'] = [url]
            print("✅ Added custom endpoint")
            
    # Save configuration
    with open('.rpc_config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    print("\n✅ Configuration saved to .rpc_config.json")
    print("\nYou can also set environment variables:")
    print("export ALCHEMY_API_KEY=your-key")
    print("export INFURA_API_KEY=your-key")
    print("export OPTIMISM_RPC_URL=https://...")
    

def test_rpc_config():
    """Test RPC configuration for all networks."""
    print("\n🧪 Testing RPC Configuration")
    print("=" * 50)
    
    for network in ['optimism', 'arbitrum']:
        print(f"\n📡 Testing {network.upper()}")
        try:
            manager = RPCManager(network)
            results = manager.test_all_endpoints()
            
            working = sum(1 for v in results.values() if v)
            total = len(results)
            
            print(f"\nResults: {working}/{total} endpoints working")
            for endpoint, status in results.items():
                print(f"  {'✅' if status else '❌'} {endpoint}")
                
        except Exception as e:
            print(f"❌ Error testing {network}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        create_rpc_config()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_rpc_config()
    else:
        # Run tests
        test_rpc_config()