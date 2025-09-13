#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Environment Deployment Configuration
Manages development, testing, staging, and production environments.
"""

import os
import json
import logging
from typing import Dict, Any
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for a single environment."""
    name: str
    network: str  # blockchain network
    rpc_url: str
    chain_id: int
    contracts: Dict[str, str]
    api_endpoints: Dict[str, str]
    database: Dict[str, str]
    redis: Dict[str, str]
    logging: Dict[str, Any]
    monitoring: Dict[str, Any]
    secrets_provider: str  # 'env', 'aws_secrets', 'vault'
    feature_flags: Dict[str, bool]
    

class DeploymentManager:
    """Manages multi-environment deployments."""
    
    # Environment definitions
    ENVIRONMENTS = {
        'development': {
            'network': 'optimism_sepolia',
            'chain_id': 11155420,
            'rpc_url': os.getenv('DEV_RPC_URL', 'https://sepolia.optimism.io'),
            'contracts': {
                'sports_amm_v2': '0x5e2c7704F5f784B2e4A5C1bD5b7fBE578A2b6EdC',
                'usdc': '0x5fd84259d66Cd46123540766Be93DFE6D43130D7'
            },
            'api_endpoints': {
                'overtime': 'https://api-sepolia.overtime.markets',
                'graphql': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-sepolia'
            },
            'database': {
                'url': os.getenv('DEV_DB_URL', 'sqlite:///dev_sport_odds.db'),
                'pool_size': 5,
                'echo': True
            },
            'redis': {
                'url': os.getenv('DEV_REDIS_URL', 'redis://localhost:6379/0'),
                'ttl': 300
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'detailed',
                'destinations': ['console', 'file']
            },
            'monitoring': {
                'enabled': False,
                'apm_enabled': False,
                'metrics_port': 9090
            },
            'secrets_provider': 'env',
            'feature_flags': {
                'paper_trading': True,
                'live_trading': False,
                'experimental_signals': True,
                'debug_mode': True
            }
        },
        
        'testing': {
            'network': 'optimism_sepolia',
            'chain_id': 11155420,
            'rpc_url': os.getenv('TEST_RPC_URL', 'https://sepolia.optimism.io'),
            'contracts': {
                'sports_amm_v2': '0x5e2c7704F5f784B2e4A5C1bD5b7fBE578A2b6EdC',
                'usdc': '0x5fd84259d66Cd46123540766Be93DFE6D43130D7'
            },
            'api_endpoints': {
                'overtime': 'https://api-sepolia.overtime.markets',
                'graphql': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-sepolia'
            },
            'database': {
                'url': 'sqlite:///test_sport_odds.db',
                'pool_size': 1,
                'echo': False
            },
            'redis': {
                'url': 'redis://localhost:6379/1',
                'ttl': 60
            },
            'logging': {
                'level': 'INFO',
                'format': 'simple',
                'destinations': ['console']
            },
            'monitoring': {
                'enabled': False,
                'apm_enabled': False,
                'metrics_port': 9091
            },
            'secrets_provider': 'env',
            'feature_flags': {
                'paper_trading': True,
                'live_trading': False,
                'experimental_signals': True,
                'debug_mode': False
            }
        },
        
        'staging': {
            'network': 'optimism',
            'chain_id': 10,
            'rpc_url': os.getenv('STAGING_RPC_URL', 'https://mainnet.optimism.io'),
            'contracts': {
                'sports_amm_v2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
                'usdc': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607'
            },
            'api_endpoints': {
                'overtime': 'https://overtimemarketsv2.xyz',
                'graphql': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism'
            },
            'database': {
                'url': os.getenv('STAGING_DB_URL', 'postgresql://user:pass@localhost/staging_odds'),
                'pool_size': 10,
                'echo': False
            },
            'redis': {
                'url': os.getenv('STAGING_REDIS_URL', 'redis://staging-redis:6379/0'),
                'ttl': 600
            },
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'destinations': ['console', 'elasticsearch']
            },
            'monitoring': {
                'enabled': True,
                'apm_enabled': True,
                'metrics_port': 9090,
                'datadog_api_key': os.getenv('DATADOG_API_KEY')
            },
            'secrets_provider': 'aws_secrets',
            'feature_flags': {
                'paper_trading': True,
                'live_trading': True,
                'experimental_signals': False,
                'debug_mode': False
            }
        },
        
        'production': {
            'network': 'optimism',
            'chain_id': 10,
            'rpc_url': os.getenv('PROD_RPC_URL', 'https://mainnet.optimism.io'),
            'contracts': {
                'sports_amm_v2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
                'usdc': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607'
            },
            'api_endpoints': {
                'overtime': 'https://overtimemarketsv2.xyz',
                'graphql': 'https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism'
            },
            'database': {
                'url': os.getenv('PROD_DB_URL', 'postgresql://user:pass@prod-db/sport_odds'),
                'pool_size': 20,
                'echo': False,
                'ssl_mode': 'require'
            },
            'redis': {
                'url': os.getenv('PROD_REDIS_URL', 'redis://prod-redis:6379/0'),
                'ttl': 3600,
                'cluster_mode': True
            },
            'logging': {
                'level': 'WARNING',
                'format': 'json',
                'destinations': ['elasticsearch', 'cloudwatch']
            },
            'monitoring': {
                'enabled': True,
                'apm_enabled': True,
                'metrics_port': 9090,
                'datadog_api_key': os.getenv('DATADOG_API_KEY'),
                'pagerduty_key': os.getenv('PAGERDUTY_KEY')
            },
            'secrets_provider': 'aws_secrets',
            'feature_flags': {
                'paper_trading': False,
                'live_trading': True,
                'experimental_signals': False,
                'debug_mode': False
            }
        }
    }
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.config = self._load_config(environment)
        self._setup_logging()
        
    def _load_config(self, environment: str) -> EnvironmentConfig:
        """Load configuration for specified environment."""
        if environment not in self.ENVIRONMENTS:
            raise ValueError(f"Unknown environment: {environment}")
            
        env_dict = self.ENVIRONMENTS[environment]
        
        return EnvironmentConfig(
            name=environment,
            network=env_dict['network'],
            rpc_url=env_dict['rpc_url'],
            chain_id=env_dict['chain_id'],
            contracts=env_dict['contracts'],
            api_endpoints=env_dict['api_endpoints'],
            database=env_dict['database'],
            redis=env_dict['redis'],
            logging=env_dict['logging'],
            monitoring=env_dict['monitoring'],
            secrets_provider=env_dict['secrets_provider'],
            feature_flags=env_dict['feature_flags']
        )
        
    def _setup_logging(self):
        """Configure logging based on environment."""
        log_config = self.config.logging
        
        # Set log level
        log_level = getattr(logging, log_config['level'])
        logging.getLogger().setLevel(log_level)
        
        # Configure format
        if log_config['format'] == 'json':
            # Would use python-json-logger here
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"message": "%(message)s", "module": "%(module)s"}'
            )
        elif log_config['format'] == 'detailed':
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            
        # Configure handlers
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            
        if 'console' in log_config['destinations']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
            
        if 'file' in log_config['destinations']:
            file_handler = logging.FileHandler(f'{self.environment}.log')
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            
    def get_database_url(self) -> str:
        """Get database URL with secrets resolved."""
        db_url = self.config.database['url']
        
        if self.config.secrets_provider == 'aws_secrets':
            # Would fetch from AWS Secrets Manager
            pass
        elif self.config.secrets_provider == 'vault':
            # Would fetch from HashiCorp Vault
            pass
            
        return db_url
        
    def get_contract_address(self, contract_name: str) -> str:
        """Get contract address for current environment."""
        return self.config.contracts.get(contract_name, '')
        
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.config.feature_flags.get(feature, False)
        
    def get_rpc_provider(self):
        """Get Web3 RPC provider for current environment."""
        from web3 import Web3
        
        return Web3(Web3.HTTPProvider(self.config.rpc_url))
        
    def export_config(self, format: str = 'json') -> str:
        """Export configuration in specified format."""
        config_dict = asdict(self.config)
        
        if format == 'json':
            return json.dumps(config_dict, indent=2)
        elif format == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False)
        elif format == 'env':
            lines = []
            for key, value in self._flatten_dict(config_dict).items():
                env_key = f"{self.environment.upper()}_{key.upper()}"
                lines.append(f"{env_key}={value}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary for env vars."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def validate_config(self) -> bool:
        """Validate current configuration."""
        required_fields = [
            'rpc_url', 'chain_id', 'contracts', 'database'
        ]
        
        for field in required_fields:
            if not hasattr(self.config, field):
                logger.error(f"Missing required field: {field}")
                return False
                
        # Test RPC connection
        try:
            w3 = self.get_rpc_provider()
            if not w3.is_connected():
                logger.error("Cannot connect to RPC provider")
                return False
        except Exception as e:
            logger.error(f"RPC validation failed: {e}")
            return False
            
        return True


def create_docker_compose(env_config: EnvironmentConfig) -> str:
    """Generate docker-compose.yml for environment."""
    compose = f"""version: '3.8'

services:
  app:
    build: .
    environment:
      - ENVIRONMENT={env_config.name}
      - RPC_URL={env_config.rpc_url}
      - CHAIN_ID={env_config.chain_id}
      - DATABASE_URL={env_config.database['url']}
      - REDIS_URL={env_config.redis['url']}
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=sport_odds
      - POSTGRES_USER=ominari
      - POSTGRES_PASSWORD=changeme
    volumes:
      - db_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  prometheus:
    image: prom/prometheus
    ports:
      - "{env_config.monitoring['metrics_port']}:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
volumes:
  db_data:
"""
    return compose


def create_kubernetes_manifests(env_config: EnvironmentConfig) -> Dict[str, str]:
    """Generate Kubernetes manifests for environment."""
    
    deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: ominari-{env_config.name}
  labels:
    app: ominari
    environment: {env_config.name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ominari
      environment: {env_config.name}
  template:
    metadata:
      labels:
        app: ominari
        environment: {env_config.name}
    spec:
      containers:
      - name: app
        image: ominari:{env_config.name}
        env:
        - name: ENVIRONMENT
          value: "{env_config.name}"
        - name: RPC_URL
          valueFrom:
            secretKeyRef:
              name: ominari-secrets
              key: rpc-url
        ports:
        - containerPort: 8000
"""
    
    service = f"""apiVersion: v1
kind: Service
metadata:
  name: ominari-{env_config.name}
spec:
  selector:
    app: ominari
    environment: {env_config.name}
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
    
    return {
        'deployment.yaml': deployment,
        'service.yaml': service
    }


def main():
    """Example usage of deployment configuration."""
    # Create configs for all environments
    for env_name in ['development', 'testing', 'staging', 'production']:
        print(f"\n{'='*50}")
        print(f"Environment: {env_name}")
        print('='*50)
        
        # Create deployment manager
        deploy = DeploymentManager(env_name)
        
        # Validate configuration
        if deploy.validate_config():
            print("✅ Configuration valid")
        else:
            print("❌ Configuration invalid")
            
        # Show key settings
        print(f"Network: {deploy.config.network}")
        print(f"Chain ID: {deploy.config.chain_id}")
        print(f"Database: {deploy.config.database['url']}")
        print(f"Live Trading: {deploy.is_feature_enabled('live_trading')}")
        
        # Export configuration
        env_file = f".env.{env_name}"
        with open(env_file, 'w') as f:
            f.write(deploy.export_config('env'))
        print(f"Exported config to {env_file}")
        
        # Generate Docker Compose
        if env_name == 'development':
            compose = create_docker_compose(deploy.config)
            with open(f"docker-compose.{env_name}.yml", 'w') as f:
                f.write(compose)
            print(f"Generated docker-compose.{env_name}.yml")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()