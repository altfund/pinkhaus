#!/usr/bin/env python3
"""Test blockchain reading with the correct SportsAMMV2 contract."""

from web3 import Web3

# Connect to Optimism
RPC_URL = "https://mainnet.optimism.io"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Correct contract address
SPORTS_AMM_V2 = "0xFb4e4811C7A811E098A556bD79B64c20b479E431"

# Load ABI (simplified for events)
ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "buyer", "type": "address"},
            {"indexed": False, "name": "marketId", "type": "bytes32"},
            {"indexed": False, "name": "position", "type": "uint8"},
            {"indexed": False, "name": "amount", "type": "uint256"},
            {"indexed": False, "name": "sUSDPaid", "type": "uint256"},
            {"indexed": False, "name": "susd", "type": "address"},
            {"indexed": False, "name": "asset", "type": "address"}
        ],
        "name": "BoughtFromAmm",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "marketId", "type": "bytes32"},
            {"indexed": False, "name": "different", "type": "bool"}
        ],
        "name": "ReferrerPaid",
        "type": "event"
    }
]

# Initialize contract
contract = w3.eth.contract(address=Web3.to_checksum_address(SPORTS_AMM_V2), abi=ABI)

# Get latest block
latest_block = w3.eth.block_number
print(f"Latest block: {latest_block}")

# Check last 1000 blocks for any trades
from_block = latest_block - 1000
print(f"Checking blocks {from_block} to {latest_block} for BoughtFromAmm events...")

try:
    # Get BoughtFromAmm events
    events = contract.events.BoughtFromAmm().get_logs(
        from_block=from_block,
        to_block=latest_block
    )
    
    print(f"\nFound {len(events)} trade events")
    
    for event in events[:5]:  # Show first 5
        print("\nTrade Event:")
        print(f"  Block: {event['blockNumber']}")
        print(f"  Buyer: {event['args']['buyer']}")
        print(f"  Market ID: {event['args']['marketId'].hex()}")
        print(f"  Position: {event['args']['position']}")
        print(f"  Amount: {event['args']['amount'] / 1e18:.4f}")
        print(f"  USD Paid: {event['args']['sUSDPaid'] / 1e18:.2f}")
        
except Exception as e:
    print(f"Error fetching events: {e}")

# Also check recent transactions
print("\nChecking recent transactions to contract...")
try:
    # This is a simplified check - in production would need proper filtering
    block = w3.eth.get_block(latest_block, full_transactions=True)
    
    txs_to_contract = [tx for tx in block.transactions if tx.get('to') and tx['to'].lower() == SPORTS_AMM_V2.lower()]
    
    print(f"Found {len(txs_to_contract)} transactions to SportsAMMV2 in latest block")
    
except Exception as e:
    print(f"Error checking transactions: {e}")