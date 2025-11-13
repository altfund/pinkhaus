#!/usr/bin/env python3
"""
Decode hex IDs properly
"""

test_ids = [
    "0x3230323530393230393431313834373000000000000000000000000000000000",
    "0x3230323530393230333730313031334300000000000000000000000000000000"
]

for hex_id in test_ids:
    # Remove 0x prefix and trailing zeros
    hex_data = hex_id[2:].rstrip('0')
    # Add a 0 if odd length
    if len(hex_data) % 2 == 1:
        hex_data += '0'
    
    # Decode hex to ASCII
    decoded = bytes.fromhex(hex_data).decode('ascii')
    print(f"\nOriginal: {hex_id}")
    print(f"Decoded: {decoded}")
    print(f"This appears to be an internal ID, NOT a blockchain address!")
    
    # Check if it starts with a date
    if decoded.startswith('2025'):
        print(f"Likely format: YYYYMMDD + other ID components")