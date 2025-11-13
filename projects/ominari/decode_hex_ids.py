#!/usr/bin/env python3
"""
Try to decode the hex IDs
"""
import binascii

test_id = "0x3230323530393230393431313834373000000000000000000000000000000000"

print(f"Original: {test_id}")
print(f"Length: {len(test_id)}")

# Remove 0x prefix and trailing zeros
hex_data = test_id[2:].rstrip('0')
print(f"\nHex data without 0x and trailing zeros: {hex_data}")

try:
    # Try to decode as hex to ASCII
    decoded = binascii.unhexlify(hex_data).decode('ascii')
    print(f"Decoded as ASCII: {decoded}")
except Exception as e:
    print(f"Could not decode as ASCII: {e}")

# Check if it's a timestamp or other encoded data
print(f"\nFirst few hex pairs: {hex_data[:20]}")
for i in range(0, min(20, len(hex_data)), 2):
    hex_pair = hex_data[i:i+2]
    decimal = int(hex_pair, 16)
    char = chr(decimal) if 32 <= decimal < 127 else f'\\x{hex_pair}'
    print(f"  {hex_pair} = {decimal} = {char}")