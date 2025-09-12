#!/usr/bin/env python3
"""Fetch verified contract ABIs from Etherscan."""

import requests
import json
import os
from typing import Dict, Any

ETHERSCAN_APIS = {
    'optimism': 'https://api-optimistic.etherscan.io/api',
    'arbitrum': 'https://api.arbiscan.io/api'
}

# You'll need API keys from Etherscan
API_KEYS = {
    'optimism': os.getenv('OPTIMISM_ETHERSCAN_KEY', ''),
    'arbitrum': os.getenv('ARBITRUM_ETHERSCAN_KEY', '')
}

# Known contract addresses for SportsAMMV2
CONTRACTS = {
    'optimism': {
        'SportsAMMV2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'SportsAMMV2Manager': '0x5a17bD98Aaa4C1B6B0Fa5e36cE6dF97E83ED4c29'
    },
    'arbitrum': {
        'SportsAMMV2': '0x7465c5d60d3d095443CF9991Da03304A30D42Eae',
        'SportsAMMV2Manager': '0x324071dE63Ca8fe21885F7fb3a60f8b140C183BC'
    }
}


def fetch_abi(network: str, address: str) -> Dict[str, Any]:
    """Fetch contract ABI from Etherscan."""
    api_url = ETHERSCAN_APIS[network]
    api_key = API_KEYS[network]
    
    if not api_key:
        print(f"Warning: No API key for {network}. Using demo ABI.")
        return None
    
    params = {
        'module': 'contract',
        'action': 'getabi',
        'address': address,
        'apikey': api_key
    }
    
    response = requests.get(api_url, params=params)
    data = response.json()
    
    if data['status'] == '1':
        return json.loads(data['result'])
    else:
        print(f"Error fetching ABI for {address}: {data.get('message', 'Unknown error')}")
        return None


def save_abis():
    """Fetch and save all contract ABIs."""
    abis = {}
    
    for network, contracts in CONTRACTS.items():
        abis[network] = {}
        for name, address in contracts.items():
            print(f"Fetching {name} ABI from {network}...")
            abi = fetch_abi(network, address)
            if abi:
                abis[network][name] = abi
                print(f"✓ Got {len(abi)} ABI entries for {name}")
            else:
                print(f"✗ Failed to fetch {name} ABI")
    
    # Save to file
    with open('contract_abis.json', 'w') as f:
        json.dump(abis, f, indent=2)
    
    print("\nABIs saved to contract_abis.json")
    return abis


if __name__ == '__main__':
    # For now, use a simplified ABI that should work
    # In production, you'd fetch from Etherscan
    
    sports_amm_v2_abi = [
        {
            "name": "MarketCreated",
            "type": "event",
            "anonymous": False,
            "inputs": [
                {"name": "market", "type": "address", "indexed": True},
                {"name": "gameId", "type": "bytes32", "indexed": True},
                {"name": "gameLabel", "type": "string", "indexed": False},
                {"name": "maturityDate", "type": "uint256", "indexed": False},
                {"name": "tags", "type": "uint256[]", "indexed": False},
                {"name": "normalizedOdds", "type": "uint256[]", "indexed": False}
            ]
        },
        {
            "name": "BoughtFromAmm",
            "type": "event",
            "anonymous": False,
            "inputs": [
                {"name": "buyer", "type": "address", "indexed": True},
                {"name": "market", "type": "address", "indexed": True},
                {"name": "position", "type": "uint8", "indexed": False},
                {"name": "amount", "type": "uint256", "indexed": False},
                {"name": "sUSDPaid", "type": "uint256", "indexed": False},
                {"name": "susd", "type": "address", "indexed": False},
                {"name": "asset", "type": "address", "indexed": False}
            ]
        },
        {
            "name": "buyFromAMM",
            "type": "function",
            "inputs": [
                {"name": "market", "type": "address"},
                {"name": "position", "type": "uint8"},
                {"name": "amount", "type": "uint256"},
                {"name": "expectedPayout", "type": "uint256"},
                {"name": "additionalSlippage", "type": "uint256"}
            ],
            "outputs": [{"name": "", "type": "uint256"}]
        }
    ]
    
    # Save simplified ABI
    abis = {
        'optimism': {'SportsAMMV2': sports_amm_v2_abi},
        'arbitrum': {'SportsAMMV2': sports_amm_v2_abi}
    }
    
    with open('contract_abis.json', 'w') as f:
        json.dump(abis, f, indent=2)
    
    print("Created simplified contract ABIs")
    print(f"Optimism SportsAMMV2: {CONTRACTS['optimism']['SportsAMMV2']}")
    print(f"Arbitrum SportsAMMV2: {CONTRACTS['arbitrum']['SportsAMMV2']}")