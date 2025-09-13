#!/usr/bin/env python3
"""
Testnet Configuration for Ominari Blockchain Trading

Provides safe testnet environments for testing blockchain trading
without risking real funds.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for a blockchain network."""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    native_token: str
    is_testnet: bool
    faucet_url: Optional[str] = None
    
    
@dataclass
class ContractConfig:
    """Configuration for smart contracts."""
    address: str
    abi_path: str
    deployment_block: int
    network: str


class TestnetConfig:
    """Comprehensive testnet configuration manager."""
    
    def __init__(self):
        self.networks = self._setup_networks()
        self.contracts = self._setup_contracts()
        self.trading_config = self._setup_trading_config()
        
    def _setup_networks(self) -> Dict[str, NetworkConfig]:
        """Configure testnet networks."""
        return {
            # Optimism Testnet (Sepolia)
            'optimism_sepolia': NetworkConfig(
                name='Optimism Sepolia',
                chain_id=11155420,
                rpc_url='https://sepolia.optimism.io',
                explorer_url='https://sepolia-optimism.etherscan.io',
                native_token='ETH',
                is_testnet=True,
                faucet_url='https://faucet.paradigm.xyz'
            ),
            
            # Arbitrum Testnet (Sepolia)
            'arbitrum_sepolia': NetworkConfig(
                name='Arbitrum Sepolia',
                chain_id=421614,
                rpc_url='https://sepolia-rollup.arbitrum.io/rpc',
                explorer_url='https://sepolia.arbiscan.io',
                native_token='ETH',
                is_testnet=True,
                faucet_url='https://faucet.paradigm.xyz'
            ),
            
            # Polygon Mumbai (deprecated but may still have contracts)
            'polygon_mumbai': NetworkConfig(
                name='Polygon Mumbai',
                chain_id=80001,
                rpc_url='https://rpc-mumbai.maticvigil.com',
                explorer_url='https://mumbai.polygonscan.com',
                native_token='MATIC',
                is_testnet=True,
                faucet_url='https://faucet.polygon.technology'
            ),
            
            # Base Sepolia Testnet
            'base_sepolia': NetworkConfig(
                name='Base Sepolia',
                chain_id=84532,
                rpc_url='https://sepolia.base.org',
                explorer_url='https://sepolia.basescan.org',
                native_token='ETH',
                is_testnet=True,
                faucet_url='https://www.coinbase.com/faucets/base-ethereum-sepolia-faucet'
            ),
            
            # Ethereum Sepolia (for reference)
            'ethereum_sepolia': NetworkConfig(
                name='Ethereum Sepolia',
                chain_id=11155111,
                rpc_url='https://sepolia.infura.io/v3/YOUR_PROJECT_ID',
                explorer_url='https://sepolia.etherscan.io',
                native_token='ETH',
                is_testnet=True,
                faucet_url='https://faucet.paradigm.xyz'
            )
        }
    
    def _setup_contracts(self) -> Dict[str, ContractConfig]:
        """Configure known testnet contracts."""
        return {
            # Overtime Markets on Optimism Sepolia
            'overtime_sepolia': ContractConfig(
                address='0x1234567890123456789012345678901234567890',  # Placeholder
                abi_path='contracts/overtime_market.json',
                deployment_block=1000000,
                network='optimism_sepolia'
            ),
            
            # Example prediction market on Arbitrum Sepolia
            'prediction_arbitrum': ContractConfig(
                address='0x2345678901234567890123456789012345678901',  # Placeholder
                abi_path='contracts/prediction_market.json',
                deployment_block=1000000,
                network='arbitrum_sepolia'
            )
        }
    
    def _setup_trading_config(self) -> Dict:
        """Configure safe trading parameters for testnet."""
        return {
            'max_position_size': 0.01,  # 0.01 ETH max per position (testnet)
            'max_daily_volume': 0.1,    # 0.1 ETH max daily trading
            'min_odds': 1.1,            # Minimum odds to consider
            'max_odds': 10.0,           # Maximum odds to consider
            'kelly_multiplier': 0.1,    # Conservative Kelly sizing (10% of calculated)
            'stop_loss_threshold': 0.5, # Stop if down 50%
            'profit_target': 2.0,       # Take profit at 100% gain
            
            # Risk management
            'max_concurrent_positions': 5,
            'position_timeout_hours': 24,
            'emergency_stop_loss': 0.8,  # Emergency stop at 80% loss
            
            # Testing parameters
            'simulation_mode': True,
            'paper_trading_only': True,
            'log_all_decisions': True,
            'detailed_reporting': True
        }
    
    def get_network(self, network_name: str) -> NetworkConfig:
        """Get network configuration by name."""
        if network_name not in self.networks:
            raise ValueError(f"Unknown network: {network_name}")
        return self.networks[network_name]
    
    def get_testnet_networks(self) -> List[NetworkConfig]:
        """Get all testnet network configurations."""
        return [config for config in self.networks.values() if config.is_testnet]
    
    def get_contract(self, contract_name: str) -> ContractConfig:
        """Get contract configuration by name."""
        if contract_name not in self.contracts:
            raise ValueError(f"Unknown contract: {contract_name}")
        return self.contracts[contract_name]
    
    def create_testnet_environment_vars(self) -> Dict[str, str]:
        """Create environment variables for testnet deployment."""
        env_vars = {
            'ENVIRONMENT': 'testnet',
            'TESTNET_MODE': 'true',
            'PAPER_TRADING_ONLY': 'true',
            'LOG_LEVEL': 'DEBUG',
            'MAX_POSITION_SIZE': '0.01',
            'MAX_DAILY_VOLUME': '0.1',
            'KELLY_MULTIPLIER': '0.1',
            'SIMULATION_MODE': 'true'
        }
        
        # Add RPC URLs for testnets
        for name, config in self.networks.items():
            if config.is_testnet:
                env_key = f"{name.upper()}_RPC_URL"
                env_vars[env_key] = config.rpc_url
        
        return env_vars
    
    def validate_testnet_setup(self) -> Dict[str, bool]:
        """Validate that testnet configuration is safe."""
        validation_results = {}
        
        # Check that we're not using mainnet
        for name, config in self.networks.items():
            if not config.is_testnet:
                validation_results[f"{name}_mainnet_check"] = False
                logger.warning(f"⚠️ {name} is not marked as testnet!")
            else:
                validation_results[f"{name}_testnet_check"] = True
        
        # Check trading limits are conservative
        trading_config = self.trading_config
        
        validation_results['max_position_safe'] = trading_config['max_position_size'] <= 0.1
        validation_results['max_volume_safe'] = trading_config['max_daily_volume'] <= 1.0
        validation_results['kelly_conservative'] = trading_config['kelly_multiplier'] <= 0.25
        validation_results['paper_trading_enabled'] = trading_config['paper_trading_only']
        validation_results['simulation_enabled'] = trading_config['simulation_mode']
        
        # Overall safety check
        all_safe = all(validation_results.values())
        validation_results['overall_safe'] = all_safe
        
        return validation_results
    
    def generate_testnet_report(self) -> str:
        """Generate comprehensive testnet configuration report."""
        report = []
        report.append("🧪 OMINARI TESTNET CONFIGURATION REPORT")
        report.append("=" * 60)
        
        # Network configurations
        report.append("\n📡 TESTNET NETWORKS:")
        for name, config in self.networks.items():
            if config.is_testnet:
                report.append(f"   ✅ {config.name}")
                report.append(f"      Chain ID: {config.chain_id}")
                report.append(f"      RPC: {config.rpc_url}")
                report.append(f"      Faucet: {config.faucet_url or 'N/A'}")
        
        # Contract configurations  
        report.append("\n📋 SMART CONTRACTS:")
        for name, config in self.contracts.items():
            network_config = self.networks[config.network]
            report.append(f"   📄 {name}")
            report.append(f"      Network: {network_config.name}")
            report.append(f"      Address: {config.address}")
            report.append(f"      Block: {config.deployment_block}")
        
        # Trading configuration
        report.append("\n⚙️ TRADING CONFIGURATION:")
        for key, value in self.trading_config.items():
            report.append(f"   {key}: {value}")
        
        # Safety validation
        report.append("\n🔒 SAFETY VALIDATION:")
        validation = self.validate_testnet_setup()
        for check, passed in validation.items():
            status = "✅" if passed else "❌"
            report.append(f"   {status} {check}: {passed}")
        
        # Environment variables
        report.append("\n🌍 ENVIRONMENT VARIABLES:")
        env_vars = self.create_testnet_environment_vars()
        for key, value in env_vars.items():
            report.append(f"   {key}={value}")
        
        return "\n".join(report)


def create_testnet_docker_compose():
    """Create testnet-specific docker-compose configuration."""
    
    testnet_compose = """version: '3.8'

services:
  # Testnet trading system
  ominari-testnet:
    build: .
    container_name: ominari-testnet
    environment:
      - SERVICE=trading
      - ENVIRONMENT=testnet
      - TESTNET_MODE=true
      - PAPER_TRADING_ONLY=true
      - LOG_LEVEL=DEBUG
      - MAX_POSITION_SIZE=0.01
      - MAX_DAILY_VOLUME=0.1
      - KELLY_MULTIPLIER=0.1
      - SIMULATION_MODE=true
      - DATABASE_URL=postgresql://ominari_user:ominari_testnet@postgres-testnet:5432/ominari_testnet
      - REDIS_URL=redis://redis-testnet:6379/0
      # Testnet RPC URLs
      - OPTIMISM_SEPOLIA_RPC_URL=https://sepolia.optimism.io
      - ARBITRUM_SEPOLIA_RPC_URL=https://sepolia-rollup.arbitrum.io/rpc
      - BASE_SEPOLIA_RPC_URL=https://sepolia.base.org
    ports:
      - "8001:8000"  # Different port from mainnet
    volumes:
      - testnet_logs:/app/logs
      - testnet_reports:/app/betting_reports
    networks:
      - ominari-testnet
    depends_on:
      postgres-testnet:
        condition: service_healthy
      redis-testnet:
        condition: service_healthy
    restart: unless-stopped

  # Testnet blockchain sync
  ominari-blockchain-testnet:
    build: .
    container_name: ominari-blockchain-testnet
    environment:
      - SERVICE=blockchain-sync
      - ENVIRONMENT=testnet
      - TESTNET_MODE=true
      - DATABASE_URL=postgresql://ominari_user:ominari_testnet@postgres-testnet:5432/ominari_testnet
      - REDIS_URL=redis://redis-testnet:6379/0
      - OPTIMISM_SEPOLIA_RPC_URL=https://sepolia.optimism.io
      - ARBITRUM_SEPOLIA_RPC_URL=https://sepolia-rollup.arbitrum.io/rpc
      - SYNC_INTERVAL=30  # Faster sync for testing
    volumes:
      - testnet_blockchain_logs:/app/logs
    networks:
      - ominari-testnet
    depends_on:
      postgres-testnet:
        condition: service_healthy
    restart: unless-stopped

  # Testnet PostgreSQL
  postgres-testnet:
    image: postgres:15-alpine
    container_name: ominari-postgres-testnet
    environment:
      POSTGRES_DB: ominari_testnet
      POSTGRES_USER: ominari_user
      POSTGRES_PASSWORD: ominari_testnet
    ports:
      - "5433:5432"  # Different port from mainnet
    volumes:
      - postgres_testnet_data:/var/lib/postgresql/data
      - ./postgres-init:/docker-entrypoint-initdb.d
    networks:
      - ominari-testnet
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ominari_user -d ominari_testnet"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Testnet Redis
  redis-testnet:
    image: redis:7-alpine
    container_name: ominari-redis-testnet
    ports:
      - "6380:6379"  # Different port from mainnet
    command: redis-server --appendonly yes --maxmemory 256mb
    volumes:
      - redis_testnet_data:/data
    networks:
      - ominari-testnet
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Testnet monitoring
  grafana-testnet:
    image: grafana/grafana:latest
    container_name: ominari-grafana-testnet
    ports:
      - "3001:3000"  # Different port from mainnet
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=testnet_admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana_testnet_data:/var/lib/grafana
    networks:
      - ominari-testnet
    restart: unless-stopped

volumes:
  postgres_testnet_data:
  redis_testnet_data:
  testnet_logs:
  testnet_reports:
  testnet_blockchain_logs:
  grafana_testnet_data:

networks:
  ominari-testnet:
    driver: bridge
"""
    
    with open('docker-compose.testnet.yml', 'w') as f:
        f.write(testnet_compose)
    
    logger.info("✅ Created docker-compose.testnet.yml")


def main():
    """Main testnet configuration setup."""
    logger.info("🧪 Setting up Ominari Testnet Configuration")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = TestnetConfig()
    
    # Generate and display report
    report = config.generate_testnet_report()
    print(report)
    
    # Validate safety
    validation = config.validate_testnet_setup()
    if validation['overall_safe']:
        logger.info("\n✅ TESTNET CONFIGURATION IS SAFE")
    else:
        logger.error("\n❌ TESTNET CONFIGURATION HAS SAFETY ISSUES")
        logger.error("Please review the validation results above")
        return False
    
    # Create docker compose for testnet
    create_testnet_docker_compose()
    
    # Create environment file
    env_vars = config.create_testnet_environment_vars()
    with open('.env.testnet', 'w') as f:
        f.write("# Ominari Testnet Environment Variables\n")
        f.write(f"# Generated on {datetime.now(timezone.utc).isoformat()}\n\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info("✅ Created .env.testnet")
    
    # Instructions
    logger.info("\n🚀 TESTNET SETUP COMPLETE!")
    logger.info("Next steps:")
    logger.info("1. Get testnet ETH from faucets:")
    for name, network in config.networks.items():
        if network.is_testnet and network.faucet_url:
            logger.info(f"   - {network.name}: {network.faucet_url}")
    
    logger.info("\n2. Start testnet environment:")
    logger.info("   docker-compose -f docker-compose.testnet.yml up -d")
    
    logger.info("\n3. Monitor testnet trading:")
    logger.info("   - Trading API: http://localhost:8001")
    logger.info("   - Grafana: http://localhost:3001")
    logger.info("   - PostgreSQL: localhost:5433")
    
    logger.info("\n⚠️ REMEMBER: This is testnet only - no real money!")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)