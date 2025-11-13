#!/usr/bin/env python3
"""
Test v2_ address theory
"""

# Example v2_ IDs from our data
test_ids = [
    "v2_0x3230323530393230393431313834373000000000000000000000000000000000",
    "v2_0x3230323530393230333730313031334300000000000000000000000000000000"
]

for source_id in test_ids:
    print(f"\nOriginal ID: {source_id}")
    print(f"Length: {len(source_id)}")
    
    if source_id.startswith('v2_0x'):
        # Extract address after v2_
        address = source_id[3:]  # Remove 'v2_'
        print(f"Extracted address: {address}")
        print(f"Address length: {len(address)} (Ethereum addresses are 42 chars)")
        print(f"Blockchain link: https://optimistic.etherscan.io/address/{address}")
        print(f"Overtime link: https://overtimemarkets.xyz/markets/{source_id}")