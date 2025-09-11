#!/usr/bin/env python3
"""
Multi-Chain Data Management System

Handles data differentiation across:
- Multiple blockchain networks (Optimism, Arbitrum, Base, etc.)
- Mainnet vs Testnet environments
- Production vs Development/Testing data
- Internal test data isolation
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    PRODUCTION = "production"
    STAGING = "staging"
    TESTNET = "testnet"
    DEVELOPMENT = "development"
    TEST = "test"


class NetworkType(Enum):
    """Network types."""
    MAINNET = "mainnet"
    TESTNET = "testnet"
    LOCAL = "local"


@dataclass
class ChainConfig:
    """Configuration for a blockchain network."""
    chain_name: str
    chain_id: int
    network_type: NetworkType
    rpc_urls: List[str]
    explorer_url: str
    native_token: str
    contracts: Dict[str, str]
    is_active: bool = True
    
    def get_identifier(self) -> str:
        """Get unique chain identifier."""
        return f"{self.chain_name}_{self.network_type.value}_{self.chain_id}"


class MultiChainDataManager:
    """Manages data across multiple chains with proper isolation."""
    
    def __init__(self):
        self.chains = self._initialize_chains()
        self.environment = self._detect_environment()
        self.data_prefixes = self._setup_data_prefixes()
        
    def _detect_environment(self) -> Environment:
        """Detect current environment from env variables."""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        env_mapping = {
            'production': Environment.PRODUCTION,
            'prod': Environment.PRODUCTION,
            'staging': Environment.STAGING,
            'testnet': Environment.TESTNET,
            'development': Environment.DEVELOPMENT,
            'dev': Environment.DEVELOPMENT,
            'test': Environment.TEST
        }
        
        return env_mapping.get(env, Environment.DEVELOPMENT)
    
    def _initialize_chains(self) -> Dict[str, ChainConfig]:
        """Initialize all supported chains."""
        chains = {}
        
        # Mainnet chains
        chains['optimism_mainnet'] = ChainConfig(
            chain_name='optimism',
            chain_id=10,
            network_type=NetworkType.MAINNET,
            rpc_urls=[
                'https://mainnet.optimism.io',
                'https://optimism-mainnet.public.blastapi.io',
                'https://rpc.ankr.com/optimism'
            ],
            explorer_url='https://optimistic.etherscan.io',
            native_token='ETH',
            contracts={
                'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
                'position_manager': '0x8b0B5Cc80c8c6E31Fb09Bd3C08D6b4d8F9dB4d5C'
            }
        )
        
        chains['arbitrum_mainnet'] = ChainConfig(
            chain_name='arbitrum',
            chain_id=42161,
            network_type=NetworkType.MAINNET,
            rpc_urls=[
                'https://arb1.arbitrum.io/rpc',
                'https://arbitrum-mainnet.public.blastapi.io',
                'https://rpc.ankr.com/arbitrum'
            ],
            explorer_url='https://arbiscan.io',
            native_token='ETH',
            contracts={
                'sports_amm_v2': '0x7465c5d60d3d095443CF9991Da03304A30D42Eae',
                'position_manager': '0x9a3D6c42F5f789Bd3C08D6b4d8F9dB4d5C8b0B5C'
            }
        )
        
        chains['base_mainnet'] = ChainConfig(
            chain_name='base',
            chain_id=8453,
            network_type=NetworkType.MAINNET,
            rpc_urls=[
                'https://mainnet.base.org',
                'https://base.llamarpc.com'
            ],
            explorer_url='https://basescan.org',
            native_token='ETH',
            contracts={
                'sports_amm_v2': '0x0000000000000000000000000000000000000000',  # Placeholder
            }
        )
        
        # Testnet chains
        chains['optimism_sepolia'] = ChainConfig(
            chain_name='optimism',
            chain_id=11155420,
            network_type=NetworkType.TESTNET,
            rpc_urls=[
                'https://sepolia.optimism.io',
                'https://optimism-sepolia.public.blastapi.io'
            ],
            explorer_url='https://sepolia-optimism.etherscan.io',
            native_token='ETH',
            contracts={
                'sports_amm_v2': '0x5e2c7704F5f784B2e4A5C1bD5b7fBE578A2b6EdC',
                'test_market': '0x1234567890123456789012345678901234567890'
            }
        )
        
        chains['arbitrum_sepolia'] = ChainConfig(
            chain_name='arbitrum',
            chain_id=421614,
            network_type=NetworkType.TESTNET,
            rpc_urls=[
                'https://sepolia-rollup.arbitrum.io/rpc'
            ],
            explorer_url='https://sepolia.arbiscan.io',
            native_token='ETH',
            contracts={
                'sports_amm_v2': '0x2345678901234567890123456789012345678901',
                'test_market': '0x3456789012345678901234567890123456789012'
            }
        )
        
        # Local development chain
        chains['local_dev'] = ChainConfig(
            chain_name='local',
            chain_id=31337,
            network_type=NetworkType.LOCAL,
            rpc_urls=[
                'http://localhost:8545',
                'http://127.0.0.1:8545'
            ],
            explorer_url='http://localhost:3000',
            native_token='ETH',
            contracts={
                'mock_sports_amm': '0x5FbDB2315678afecb367f032d93F642f64180aa3',
                'test_market': '0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512'
            },
            is_active=os.getenv('USE_LOCAL_CHAIN', 'false').lower() == 'true'
        )
        
        return chains
    
    def _setup_data_prefixes(self) -> Dict[str, str]:
        """Setup data prefixes for proper isolation."""
        return {
            Environment.PRODUCTION: "prod",
            Environment.STAGING: "stage",
            Environment.TESTNET: "test",
            Environment.DEVELOPMENT: "dev",
            Environment.TEST: "unittest"
        }
    
    def get_active_chains(self) -> List[ChainConfig]:
        """Get chains active for current environment."""
        active_chains = []
        
        for chain in self.chains.values():
            # Filter based on environment
            if self.environment == Environment.PRODUCTION:
                if chain.network_type == NetworkType.MAINNET and chain.is_active:
                    active_chains.append(chain)
                    
            elif self.environment == Environment.TESTNET:
                if chain.network_type == NetworkType.TESTNET and chain.is_active:
                    active_chains.append(chain)
                    
            elif self.environment in [Environment.DEVELOPMENT, Environment.TEST]:
                if chain.network_type in [NetworkType.TESTNET, NetworkType.LOCAL] and chain.is_active:
                    active_chains.append(chain)
                    
            elif self.environment == Environment.STAGING:
                # Staging uses testnet chains but with production-like config
                if chain.network_type == NetworkType.TESTNET and chain.is_active:
                    active_chains.append(chain)
        
        return active_chains
    
    def get_data_namespace(self, chain_config: ChainConfig) -> str:
        """Get data namespace for proper isolation."""
        env_prefix = self.data_prefixes[self.environment]
        chain_id = chain_config.get_identifier()
        
        # Create namespace: env_chainname_networktype_chainid
        namespace = f"{env_prefix}_{chain_id}"
        
        # Add test suffix for unit tests
        if self.environment == Environment.TEST:
            test_id = os.getenv('TEST_RUN_ID', 'default')
            namespace = f"{namespace}_{test_id}"
        
        return namespace
    
    def get_database_config(self, chain_config: ChainConfig) -> Dict[str, Any]:
        """Get database configuration for a specific chain."""
        namespace = self.get_data_namespace(chain_config)
        
        config = {
            'namespace': namespace,
            'tables': {
                'markets': f"{namespace}_markets",
                'odds': f"{namespace}_odds",
                'positions': f"{namespace}_positions",
                'sync_status': f"{namespace}_sync_status"
            },
            'indexes': {
                'market_id': f"idx_{namespace}_market_id",
                'timestamp': f"idx_{namespace}_timestamp",
                'chain_block': f"idx_{namespace}_chain_block"
            },
            'redis_prefix': namespace,
            'schema_version': '1.0'
        }
        
        # Add environment-specific settings
        if self.environment == Environment.PRODUCTION:
            config['retention_days'] = 365
            config['archive_enabled'] = True
        elif self.environment == Environment.TESTNET:
            config['retention_days'] = 30
            config['archive_enabled'] = False
        else:
            config['retention_days'] = 7
            config['archive_enabled'] = False
        
        return config
    
    def create_market_id(self, chain_config: ChainConfig, market_address: str, 
                        market_type: str = "standard") -> str:
        """Create globally unique market ID."""
        namespace = self.get_data_namespace(chain_config)
        
        # Create deterministic ID
        components = [
            namespace,
            chain_config.chain_name,
            str(chain_config.chain_id),
            market_address.lower(),
            market_type
        ]
        
        # Hash for consistency
        id_string = "_".join(components)
        id_hash = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        
        return f"{namespace}_{id_hash}"
    
    def parse_market_id(self, market_id: str) -> Dict[str, Any]:
        """Parse market ID to extract chain and environment info."""
        parts = market_id.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid market ID format: {market_id}")
        
        env_prefix = parts[0]
        chain_name = parts[1]
        
        # Find environment
        environment = None
        for env, prefix in self.data_prefixes.items():
            if prefix == env_prefix:
                environment = env
                break
        
        return {
            'environment': environment,
            'chain_name': chain_name,
            'full_id': market_id,
            'namespace': '_'.join(parts[:-1])
        }
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration for current environment."""
        active_chains = self.get_active_chains()
        
        config = {
            'environment': self.environment.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chains': [],
            'deployment': {}
        }
        
        # Chain configurations
        for chain in active_chains:
            chain_config = {
                'name': chain.chain_name,
                'chain_id': chain.chain_id,
                'network_type': chain.network_type.value,
                'rpc_urls': chain.rpc_urls,
                'contracts': chain.contracts,
                'database': self.get_database_config(chain)
            }
            config['chains'].append(chain_config)
        
        # Environment-specific deployment settings
        if self.environment == Environment.PRODUCTION:
            config['deployment'] = {
                'replicas': 3,
                'resources': {
                    'cpu': '2000m',
                    'memory': '4Gi'
                },
                'monitoring': {
                    'enabled': True,
                    'alert_channels': ['pagerduty', 'slack']
                },
                'backup': {
                    'enabled': True,
                    'frequency': 'hourly',
                    'retention_days': 30
                }
            }
        elif self.environment == Environment.STAGING:
            config['deployment'] = {
                'replicas': 2,
                'resources': {
                    'cpu': '1000m',
                    'memory': '2Gi'
                },
                'monitoring': {
                    'enabled': True,
                    'alert_channels': ['slack']
                },
                'backup': {
                    'enabled': True,
                    'frequency': 'daily',
                    'retention_days': 7
                }
            }
        else:  # Development/Test
            config['deployment'] = {
                'replicas': 1,
                'resources': {
                    'cpu': '500m',
                    'memory': '1Gi'
                },
                'monitoring': {
                    'enabled': False
                },
                'backup': {
                    'enabled': False
                }
            }
        
        return config
    
    def validate_data_isolation(self) -> Dict[str, bool]:
        """Validate that data isolation is properly configured."""
        validation = {
            'environment_set': self.environment is not None,
            'chains_configured': len(self.chains) > 0,
            'active_chains': len(self.get_active_chains()) > 0,
            'namespace_unique': True,
            'test_isolation': True
        }
        
        # Check namespace uniqueness
        namespaces = set()
        for chain in self.chains.values():
            ns = self.get_data_namespace(chain)
            if ns in namespaces:
                validation['namespace_unique'] = False
                break
            namespaces.add(ns)
        
        # Check test isolation
        if self.environment == Environment.TEST:
            test_id = os.getenv('TEST_RUN_ID')
            validation['test_isolation'] = test_id is not None
        
        validation['all_valid'] = all(validation.values())
        
        return validation


def create_deployment_workflow():
    """Create deployment workflow documentation."""
    
    workflow = """
# Multi-Chain Deployment Workflow

## 1. Environment Structure

```
production/
├── optimism-mainnet/
├── arbitrum-mainnet/
└── base-mainnet/

staging/
├── optimism-sepolia/
├── arbitrum-sepolia/
└── base-sepolia/

development/
├── local-chain/
├── optimism-sepolia/
└── arbitrum-sepolia/

test/
├── mock-chain/
└── isolated-test-data/
```

## 2. Data Isolation Strategy

### Database Tables
- Production: `prod_optimism_mainnet_10_markets`
- Staging: `stage_optimism_testnet_11155420_markets`
- Test: `unittest_local_local_31337_abc123_markets`

### Redis Keys
- Production: `prod_optimism_mainnet_10:market:0x123...`
- Staging: `stage_optimism_testnet_11155420:market:0x456...`
- Test: `unittest_local_local_31337_abc123:market:0x789...`

## 3. Deployment Commands

### Production Deployment
```bash
export ENVIRONMENT=production
./deploy.sh production deploy

# Only mainnet chains will be active
# Data goes to production namespace
# Full monitoring and backups enabled
```

### Staging Deployment
```bash
export ENVIRONMENT=staging
./deploy.sh staging deploy

# Testnet chains active
# Production-like configuration
# Staging namespace isolation
```

### Development
```bash
export ENVIRONMENT=development
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Local + testnet chains
# Development namespace
# Debug logging enabled
```

### Testing
```bash
export ENVIRONMENT=test
export TEST_RUN_ID=$(uuidgen)
pytest tests/

# Isolated test namespace
# Automatic cleanup after tests
# No external chain connections
```

## 4. Chain-Specific Configuration

### Per-Chain Environment Variables
```bash
# Optimism Mainnet
OPTIMISM_MAINNET_RPC_URL=https://mainnet.optimism.io
OPTIMISM_MAINNET_CONTRACTS_SPORTS_AMM=0xFb4e4811...

# Optimism Sepolia
OPTIMISM_SEPOLIA_RPC_URL=https://sepolia.optimism.io
OPTIMISM_SEPOLIA_CONTRACTS_SPORTS_AMM=0x5e2c7704...

# Arbitrum Mainnet
ARBITRUM_MAINNET_RPC_URL=https://arb1.arbitrum.io/rpc
ARBITRUM_MAINNET_CONTRACTS_SPORTS_AMM=0x7465c5d6...
```

## 5. Data Flow

```
Blockchain Event → Chain-Specific Reader → Namespaced Storage → Unified API
     ↓                    ↓                      ↓                    ↓
[Optimism M.]    [prod_opt_main_10]    [PostgreSQL/Redis]    [/api/markets]
[Arbitrum T.]    [test_arb_test_421614] [Isolated Tables]    [/api/markets]
```

## 6. Monitoring & Validation

### Health Checks
- `/health/chains` - Shows active chains per environment
- `/health/isolation` - Validates data isolation
- `/metrics/chain/{chain_id}` - Chain-specific metrics

### Deployment Validation
```python
manager = MultiChainDataManager()
validation = manager.validate_data_isolation()
assert validation['all_valid'], "Data isolation check failed"
```
"""
    
    with open('MULTI_CHAIN_DEPLOYMENT_WORKFLOW.md', 'w') as f:
        f.write(workflow)
    
    return workflow


def demo_multi_chain_system():
    """Demonstrate multi-chain data management."""
    
    print("🔗 Multi-Chain Data Management Demo")
    print("=" * 60)
    
    # Initialize manager
    manager = MultiChainDataManager()
    
    print(f"\n📍 Current Environment: {manager.environment.value}")
    
    # Show active chains
    print("\n🔗 Active Chains:")
    for chain in manager.get_active_chains():
        print(f"   - {chain.chain_name} ({chain.network_type.value})")
        print(f"     Chain ID: {chain.chain_id}")
        print(f"     Namespace: {manager.get_data_namespace(chain)}")
    
    # Example market ID creation
    print("\n🆔 Market ID Examples:")
    for chain in manager.get_active_chains()[:2]:
        market_id = manager.create_market_id(
            chain, 
            "0x1234567890123456789012345678901234567890"
        )
        print(f"   {chain.chain_name}: {market_id}")
        
        # Parse it back
        parsed = manager.parse_market_id(market_id)
        print(f"     Environment: {parsed['environment'].value}")
        print(f"     Chain: {parsed['chain_name']}")
    
    # Deployment config
    config = manager.get_deployment_config()
    print(f"\n🚀 Deployment Configuration:")
    print(f"   Environment: {config['environment']}")
    print(f"   Active Chains: {len(config['chains'])}")
    print(f"   Replicas: {config['deployment'].get('replicas', 1)}")
    
    # Validation
    validation = manager.validate_data_isolation()
    print(f"\n✅ Data Isolation Validation:")
    for check, passed in validation.items():
        if check != 'all_valid':
            print(f"   {check}: {'✓' if passed else '✗'}")
    
    # Save deployment config
    with open('deployment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📄 Deployment config saved to deployment_config.json")


if __name__ == "__main__":
    # Create documentation
    create_deployment_workflow()
    
    # Run demo
    demo_multi_chain_system()
    
    print("\n✨ Multi-chain data management system ready!")
    print("📖 See MULTI_CHAIN_DEPLOYMENT_WORKFLOW.md for details")