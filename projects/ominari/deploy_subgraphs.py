#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deploy Thales Subgraphs to Local Graph Node
"""

import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Graph Node endpoints
GRAPH_NODE_HTTP = "http://localhost:18000"
GRAPH_NODE_ADMIN = "http://localhost:18020"
IPFS_URL = "http://localhost:15001"


def check_dependencies():
    """Check if required tools are installed."""
    tools = ['node', 'npm', 'graph']
    
    for tool in tools:
        try:
            subprocess.run([tool, '--version'], check=True, capture_output=True)
            logger.info(f"✅ {tool} is installed")
        except Exception:
            if tool == 'graph':
                logger.info("Installing Graph CLI...")
                try:
                    subprocess.run(['npm', 'install', '-g', '@graphprotocol/graph-cli'], check=True)
                    logger.info("✅ Graph CLI installed")
                except Exception as e:
                    logger.error(f"Failed to install Graph CLI: {e}")
                    return False
            else:
                logger.error(f"❌ {tool} is not installed")
                return False
    
    return True


def build_subgraph(subgraph_dir):
    """Build a subgraph."""
    logger.info(f"Building subgraph in {subgraph_dir}")
    
    # Install dependencies
    result = subprocess.run(
        ['npm', 'install'],
        cwd=subgraph_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"npm install failed: {result.stderr}")
        return False
    
    # Generate code (OvertimeV2 specific)
    if 'OvertimeV2' in str(subgraph_dir):
        result = subprocess.run(
            ['npm', 'run', 'codegen:optimism'],
            cwd=subgraph_dir,
            capture_output=True,
            text=True
        )
    else:
        result = subprocess.run(
            ['npm', 'run', 'codegen'],
            cwd=subgraph_dir,
            capture_output=True,
            text=True
        )
    
    if result.returncode != 0:
        logger.error(f"codegen failed: {result.stderr}")
        return False
    
    # Build - for OvertimeV2, build happens with codegen
    # For other subgraphs, we might need a separate build step
    if 'OvertimeV2' not in str(subgraph_dir):
        result = subprocess.run(
            ['npm', 'run', 'build'],
            cwd=subgraph_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"build failed: {result.stderr}")
            return False
    
    logger.info("✅ Subgraph built successfully")
    return True


def create_subgraph(name):
    """Create a subgraph on the local node."""
    logger.info(f"Creating subgraph: {name}")
    
    result = subprocess.run(
        ['graph', 'create', '--node', GRAPH_NODE_ADMIN, name],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and 'already exists' not in result.stderr:
        logger.error(f"Failed to create subgraph: {result.stderr}")
        return False
    
    logger.info(f"✅ Subgraph {name} ready")
    return True


def deploy_subgraph(subgraph_dir, manifest, name, version="0.0.1"):
    """Deploy a subgraph to the local node."""
    logger.info(f"Deploying {name} from {manifest}")
    
    # First create the subgraph
    if not create_subgraph(name):
        return False
    
    # Deploy
    result = subprocess.run(
        [
            'graph', 'deploy',
            '--node', GRAPH_NODE_ADMIN,
            '--ipfs', IPFS_URL,
            '--version-label', version,
            name,
            manifest
        ],
        cwd=subgraph_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Deployment failed: {result.stderr}")
        return False
    
    logger.info(f"✅ Deployed {name} successfully")
    return True


def update_manifest_for_local(manifest_path):
    """Update manifest file for local deployment."""
    logger.info(f"Updating manifest {manifest_path} for local deployment")
    
    # Read manifest
    with open(manifest_path, 'r') as f:
        content = f.read()
    
    # For local deployment, we might need to update the network
    # Graph Node uses "mainnet" for all EVM chains in dev mode
    updated_content = content.replace('network: optimism', 'network: mainnet')
    updated_content = updated_content.replace('network: arbitrum', 'network: mainnet')
    updated_content = updated_content.replace('network: base', 'network: mainnet')
    
    # Write temporary manifest
    temp_manifest = manifest_path.with_suffix('.local.yaml')
    with open(temp_manifest, 'w') as f:
        f.write(updated_content)
    
    return temp_manifest


def deploy_overtime_subgraphs():
    """Deploy Overtime subgraphs."""
    base_dir = Path('subgraphs/thales-subgraph')
    
    # Subgraphs to deploy
    subgraphs = [
        {
            'dir': base_dir / 'OvertimeV2',
            'manifest': 'subgraphs/subgraph-op.yaml',
            'name': 'overtime/optimism',
            'network': 'optimism'
        },
        # We can only deploy one network at a time with current Graph Node setup
        # {
        #     'dir': base_dir / 'OvertimeV2',
        #     'manifest': 'subgraphs/subgraph-arb.yaml',
        #     'name': 'overtime/arbitrum',
        #     'network': 'arbitrum'
        # }
    ]
    
    for subgraph in subgraphs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Deploying {subgraph['name']}")
        logger.info(f"{'='*60}")
        
        subgraph_dir = subgraph['dir']
        manifest_path = subgraph_dir / subgraph['manifest']
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            continue
        
        # Update manifest for local deployment
        local_manifest = update_manifest_for_local(manifest_path)
        
        # Build subgraph
        if not build_subgraph(subgraph_dir):
            logger.error(f"Failed to build {subgraph['name']}")
            continue
        
        # Deploy
        if not deploy_subgraph(
            subgraph_dir,
            local_manifest.name,
            subgraph['name']
        ):
            logger.error(f"Failed to deploy {subgraph['name']}")
            continue
        
        # Clean up temporary manifest
        local_manifest.unlink()
        
        logger.info(f"✅ {subgraph['name']} deployed successfully")
        
        # Give it time to index
        logger.info("Waiting for initial indexing...")
        time.sleep(10)


def check_deployment_status():
    """Check the status of deployed subgraphs."""
    logger.info("\n" + "="*60)
    logger.info("Deployment Status")
    logger.info("="*60)
    
    try:
        # Query the GraphQL endpoint
        import requests
        
        # List subgraphs
        response = requests.post(
            f"{GRAPH_NODE_HTTP}/graphql",
            json={
                "query": """
                {
                    subgraphDeployments {
                        subgraph
                        synced
                        health
                        node
                    }
                }
                """
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'subgraphDeployments' in data['data']:
                for deployment in data['data']['subgraphDeployments']:
                    logger.info(f"\nSubgraph: {deployment['subgraph']}")
                    logger.info(f"  Synced: {deployment['synced']}")
                    logger.info(f"  Health: {deployment['health']}")
                    logger.info(f"  Node: {deployment['node']}")
            else:
                logger.info("No deployments found")
        else:
            logger.warning(f"Failed to query deployments: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
    
    logger.info("\nQuery endpoints:")
    logger.info("  - http://localhost:18000/subgraphs/name/overtime/optimism")
    logger.info("  - http://localhost:18000/subgraphs/name/overtime/arbitrum")


def main():
    """Deploy subgraphs to local Graph Node."""
    logger.info("🚀 Deploying Thales Subgraphs")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies")
        return
    
    # Check if subgraphs are cloned
    if not Path('subgraphs/thales-subgraph').exists():
        logger.error("Thales subgraphs not found. Please clone first:")
        logger.error("git clone https://github.com/thales-markets/thales-subgraph.git subgraphs/thales-subgraph")
        return
    
    # Deploy subgraphs
    deploy_overtime_subgraphs()
    
    # Check status
    check_deployment_status()
    
    logger.info("\n✅ Subgraph deployment complete!")
    logger.info("\nNext steps:")
    logger.info("1. Update GraphQL client to use local endpoints")
    logger.info("2. Test queries against the local subgraph")
    logger.info("3. Monitor indexing progress")


if __name__ == "__main__":
    main()