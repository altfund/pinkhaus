#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deploy Graph Node Infrastructure
Sets up local Graph Node with PostgreSQL and IPFS
"""

import os
import subprocess
import time
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_docker():
    """Check if Docker and Docker Compose are installed."""
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        logger.info("✅ Docker is installed")
    except Exception:
        logger.error("❌ Docker is not installed. Please install Docker first.")
        return False
    
    try:
        subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        logger.info("✅ Docker Compose is installed")
    except Exception:
        # Try docker compose (newer version)
        try:
            subprocess.run(['docker', 'compose', 'version'], check=True, capture_output=True)
            logger.info("✅ Docker Compose (plugin) is installed")
        except Exception:
            logger.error("❌ Docker Compose is not installed.")
            return False
    
    return True


def check_env_vars():
    """Check and set up environment variables."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from example...")
        env_example.rename(env_file)
    
    # Check for RPC URLs
    optimism_rpc = os.getenv('OPTIMISM_RPC_URL')
    arbitrum_rpc = os.getenv('ARBITRUM_RPC_URL')
    
    if not optimism_rpc or 'YOUR_API_KEY' in optimism_rpc:
        logger.warning("⚠️  OPTIMISM_RPC_URL not configured. Using public endpoint (rate limited)")
        os.environ['OPTIMISM_RPC_URL'] = 'https://mainnet.optimism.io'
    
    if not arbitrum_rpc or 'YOUR_API_KEY' in arbitrum_rpc:
        logger.warning("⚠️  ARBITRUM_RPC_URL not configured. Using public endpoint (rate limited)")
        os.environ['ARBITRUM_RPC_URL'] = 'https://arb1.arbitrum.io/rpc'
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        'data/graph-node',
        'data/postgres',
        'data/ipfs',
        'subgraphs',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Created data directories")


def start_services():
    """Start Graph Node services."""
    logger.info("Starting Graph Node infrastructure...")
    
    try:
        # Stop any existing containers
        subprocess.run(['docker-compose', '-f', 'docker-compose.graph-node-alt.yml', 'down'], 
                      capture_output=True)
    except Exception:
        pass
    
    # Start services
    result = subprocess.run(
        ['docker-compose', '-f', 'docker-compose.graph-node-alt.yml', 'up', '-d'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to start services: {result.stderr}")
        return False
    
    logger.info("✅ Started Docker containers")
    return True


def wait_for_services():
    """Wait for services to be ready."""
    services = {
        'PostgreSQL': ('localhost', 15432),
        'IPFS': ('localhost', 15001),
        'Graph Node': ('localhost', 18030)
    }
    
    logger.info("Waiting for services to start...")
    
    for service, (host, port) in services.items():
        for i in range(30):  # 30 second timeout
            try:
                if service == 'Graph Node':
                    # Check Graph Node health
                    response = requests.get(f'http://{host}:{port}/', timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ {service} is ready")
                        break
                else:
                    # Simple TCP check for other services
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        logger.info(f"✅ {service} is ready")
                        break
            except Exception:
                pass
            
            time.sleep(1)
        else:
            logger.warning(f"⚠️  {service} not ready after 30 seconds")


def show_status():
    """Show status of Graph Node services."""
    logger.info("\n" + "="*60)
    logger.info("Graph Node Infrastructure Status")
    logger.info("="*60)
    
    # Check container status
    result = subprocess.run(
        ['docker-compose', '-f', 'docker-compose.graph-node-alt.yml', 'ps'],
        capture_output=True,
        text=True
    )
    
    if result.stdout:
        print(result.stdout)
    
    logger.info("\nEndpoints:")
    logger.info("  - GraphQL: http://localhost:18000/")
    logger.info("  - Admin: http://localhost:18030/")
    logger.info("  - Metrics: http://localhost:18040/metrics")
    logger.info("  - IPFS: http://localhost:18080/")
    logger.info("  - PostgreSQL: localhost:15432")
    
    logger.info("\nNext steps:")
    logger.info("1. Clone Thales subgraphs:")
    logger.info("   git clone https://github.com/thales-markets/thales-subgraph.git subgraphs/thales-subgraph")
    logger.info("2. Deploy subgraphs using deploy_subgraphs.sh")


def main():
    """Deploy Graph Node infrastructure."""
    logger.info("🚀 Deploying Graph Node Infrastructure")
    
    # Check prerequisites
    if not check_docker():
        return
    
    # Set up environment
    check_env_vars()
    create_directories()
    
    # Start services
    if not start_services():
        logger.error("Failed to start services")
        return
    
    # Wait for services
    wait_for_services()
    
    # Show status
    show_status()
    
    logger.info("\n✅ Graph Node deployment complete!")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()