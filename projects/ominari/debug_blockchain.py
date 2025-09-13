#!/usr/bin/env python3
"""Debug blockchain reader issues."""

from web3 import Web3
import json

# Test event signature calculation
w3 = Web3()

# Calculate event signature
event_sig = w3.keccak(text='MarketCreated(address,bytes32,string,uint256,uint256[],uint256[])')
print(f"Event signature (bytes): {event_sig}")
print(f"Event signature (hex): {event_sig.hex()}")
print(f"Event signature (0x hex): {w3.to_hex(event_sig)}")

# Test address formatting
address = '0x170a5714112daEfF20E798B6e92e25B86Ea603C1'
print(f"\nOriginal address: {address}")
print(f"Checksum address: {Web3.to_checksum_address(address)}")

# Test log filter format
log_filter = {
    'address': Web3.to_checksum_address(address),
    'fromBlock': 140574562,
    'toBlock': 140574662,
    'topics': [w3.to_hex(event_sig)]
}

print(f"\nLog filter: {json.dumps(log_filter, indent=2)}")

# Check if topics need 0x prefix
print(f"\nTopics[0] starts with 0x: {log_filter['topics'][0].startswith('0x')}")