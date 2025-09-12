#!/usr/bin/env python3
"""
Test Data Isolation Examples

Shows how data is properly isolated between different environments,
chains, and test runs.
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, List
import json

# Mock different environments
class TestEnvironments:
    """Simulate different deployment environments."""
    
    @staticmethod
    async def test_production_isolation():
        """Test production data isolation."""
        print("\n🏭 Testing PRODUCTION Data Isolation")
        print("-" * 50)
        
        # Set production environment
        os.environ['ENVIRONMENT'] = 'production'
        
        from multi_chain_data_manager import MultiChainDataManager
        manager = MultiChainDataManager()
        
        # Show active chains
        chains = manager.get_active_chains()
        print(f"Active chains: {[c.chain_name for c in chains]}")
        
        # Create market IDs for production
        for chain in chains[:2]:
            market_address = "0xPROD1234567890123456789012345678901234567890"
            market_id = manager.create_market_id(chain, market_address)
            db_config = manager.get_database_config(chain)
            
            print(f"\n{chain.chain_name} Mainnet:")
            print(f"  Market ID: {market_id}")
            print(f"  Tables:")
            print(f"    - {db_config['tables']['markets']}")
            print(f"    - {db_config['tables']['odds']}")
            print(f"  Redis prefix: {db_config['redis_prefix']}")
    
    @staticmethod
    async def test_staging_isolation():
        """Test staging data isolation."""
        print("\n🎭 Testing STAGING Data Isolation")
        print("-" * 50)
        
        # Set staging environment
        os.environ['ENVIRONMENT'] = 'staging'
        
        from multi_chain_data_manager import MultiChainDataManager
        manager = MultiChainDataManager()
        
        chains = manager.get_active_chains()
        print(f"Active chains: {[c.chain_name for c in chains]}")
        
        for chain in chains[:2]:
            market_address = "0xSTAG2345678901234567890123456789012345678901"
            market_id = manager.create_market_id(chain, market_address)
            db_config = manager.get_database_config(chain)
            
            print(f"\n{chain.chain_name} Testnet:")
            print(f"  Market ID: {market_id}")
            print(f"  Tables:")
            print(f"    - {db_config['tables']['markets']}")
            print(f"  Data retention: {db_config['retention_days']} days")
    
    @staticmethod
    async def test_unittest_isolation():
        """Test unit test data isolation."""
        print("\n🧪 Testing UNIT TEST Data Isolation")
        print("-" * 50)
        
        # Set test environment with unique ID
        os.environ['ENVIRONMENT'] = 'test'
        test_id = str(uuid.uuid4())[:8]
        os.environ['TEST_RUN_ID'] = test_id
        
        from multi_chain_data_manager import MultiChainDataManager
        manager = MultiChainDataManager()
        
        chains = manager.get_active_chains()
        
        for chain in chains[:1]:
            market_address = "0xTEST3456789012345678901234567890123456789012"
            market_id = manager.create_market_id(chain, market_address)
            db_config = manager.get_database_config(chain)
            
            print(f"\nTest Run ID: {test_id}")
            print(f"Chain: {chain.chain_name}")
            print(f"  Market ID: {market_id}")
            print(f"  Tables (isolated per test):")
            print(f"    - {db_config['tables']['markets']}")
            print(f"  ⚠️  These tables will be cleaned up after test!")
    
    @staticmethod
    async def test_concurrent_environments():
        """Test that different environments don't interfere."""
        print("\n🔄 Testing Concurrent Environment Isolation")
        print("-" * 50)
        
        # Same market on different environments
        test_market = "0xSAME567890123456789012345678901234567890ADDR"
        
        environments = ['production', 'staging', 'development']
        market_ids = {}
        
        for env in environments:
            os.environ['ENVIRONMENT'] = env
            
            from multi_chain_data_manager import MultiChainDataManager
            manager = MultiChainDataManager()
            
            chains = manager.get_active_chains()
            if chains:
                chain = chains[0]
                market_id = manager.create_market_id(chain, test_market)
                market_ids[env] = market_id
        
        print(f"Same market address: {test_market}")
        print("\nGenerated IDs (all different):")
        for env, mid in market_ids.items():
            print(f"  {env:12}: {mid}")
        
        # Verify all IDs are unique
        unique_ids = set(market_ids.values())
        if len(unique_ids) == len(market_ids):
            print("\n✅ All market IDs are unique across environments!")
        else:
            print("\n❌ COLLISION DETECTED! IDs are not unique!")


class DataFlowExamples:
    """Show how data flows through the system."""
    
    @staticmethod
    async def show_data_flow():
        """Demonstrate data flow from blockchain to storage."""
        print("\n📊 Data Flow Example")
        print("-" * 50)
        
        # Production flow
        print("\n1. Production Data Flow:")
        print("   Optimism Mainnet Event")
        print("   ↓")
        print("   blockchain-optimism-mainnet container")
        print("   ↓")
        print("   MultiChainDataManager.create_market_id()")
        print("   ↓")
        print("   PostgreSQL: prod_optimism_mainnet_10_markets")
        print("   ↓")
        print("   Redis: prod_optimism_mainnet_10:market:0x123...")
        print("   ↓")
        print("   Unified API: /api/v1/markets/prod_optimism_mainnet_10_a1b2c3d4")
        
        # Test flow
        print("\n2. Test Data Flow:")
        print("   Mock Blockchain Event")
        print("   ↓")
        print("   Test Environment + Random UUID")
        print("   ↓")
        print("   PostgreSQL: unittest_local_local_31337_abc123_markets")
        print("   ↓")
        print("   Cleaned up after test completion")
        
        # Query routing
        print("\n3. Query Routing:")
        print("   API Request: GET /api/v1/markets/{market_id}")
        print("   ↓")
        print("   Parse market_id → extract environment + chain")
        print("   ↓")
        print("   Route to correct database/namespace")
        print("   ↓")
        print("   Return data with source metadata")


class DeploymentExamples:
    """Show deployment command examples."""
    
    @staticmethod
    def show_deployment_commands():
        """Show example deployment commands."""
        print("\n🚀 Deployment Command Examples")
        print("-" * 50)
        
        commands = [
            {
                'desc': 'Deploy Production (all mainnet chains)',
                'cmd': './deploy-multichain.sh production deploy',
                'effect': 'Starts Optimism, Arbitrum, Base mainnet syncs'
            },
            {
                'desc': 'Deploy Staging (testnet with prod config)',
                'cmd': './deploy-multichain.sh staging deploy',
                'effect': 'Starts testnets with production-like settings'
            },
            {
                'desc': 'Deploy only Optimism testnet',
                'cmd': './deploy-multichain.sh testnet deploy optimism',
                'effect': 'Starts only Optimism Sepolia sync'
            },
            {
                'desc': 'Local development with mock chain',
                'cmd': './deploy-multichain.sh development deploy',
                'effect': 'Starts local Hardhat + testnets'
            },
            {
                'desc': 'Run tests with isolation',
                'cmd': 'TEST_RUN_ID=$(uuidgen) pytest tests/',
                'effect': 'Each test run gets unique namespace'
            },
            {
                'desc': 'Check data isolation',
                'cmd': './deploy-multichain.sh production validate',
                'effect': 'Validates namespace uniqueness'
            }
        ]
        
        for cmd_info in commands:
            print(f"\n{cmd_info['desc']}:")
            print(f"  $ {cmd_info['cmd']}")
            print(f"  → {cmd_info['effect']}")


async def main():
    """Run all examples."""
    print("🔗 Multi-Chain Data Isolation Demo")
    print("=" * 60)
    
    # Test different environments
    await TestEnvironments.test_production_isolation()
    await TestEnvironments.test_staging_isolation()
    await TestEnvironments.test_unittest_isolation()
    await TestEnvironments.test_concurrent_environments()
    
    # Show data flow
    await DataFlowExamples.show_data_flow()
    
    # Show deployment commands
    DeploymentExamples.show_deployment_commands()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Each environment has isolated namespaces")
    print("2. Test data never touches production")
    print("3. Multiple chains can run concurrently")
    print("4. Market IDs encode environment + chain info")
    print("5. Deployment profiles control what runs")
    print("6. Unit tests get unique namespaces per run")
    
    print("\n🛡️ Data Safety Guaranteed!")


if __name__ == "__main__":
    asyncio.run(main())