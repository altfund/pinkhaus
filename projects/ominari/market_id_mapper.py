#!/usr/bin/env python3
"""Market ID mapper to connect blockchain and API markets"""

import re
from typing import Dict, Optional, List, Tuple

class MarketIDMapper:
    """Maps between different market ID formats used by blockchain and API"""
    
    def __init__(self):
        # Known ID format patterns
        self.patterns = {
            'overtime_real': re.compile(r'^overtime_real_0x([0-9a-fA-F]+)$'),
            'blockchain_v2': re.compile(r'^(?:blockchain_)?v2_0x([0-9a-fA-F]+)$'),
            'api_format': re.compile(r'^(?:api_|live_api_|overtime_soccer_)(.+)$'),
            'hex_address': re.compile(r'^0x([0-9a-fA-F]+)$')
        }
    
    def extract_core_id(self, market_id: str) -> Optional[str]:
        """Extract the core hex ID from various formats"""
        if not market_id:
            return None
        
        # Try each pattern
        for format_name, pattern in self.patterns.items():
            match = pattern.match(market_id)
            if match:
                core_id = match.group(1)
                
                # Remove padding (trailing zeros)
                if format_name in ['overtime_real', 'blockchain_v2', 'hex_address']:
                    # Remove groups of 8+ zeros at the end
                    core_id = re.sub(r'(00){8,}$', '', core_id)
                    
                    # Decode hex if it looks like encoded data
                    if len(core_id) > 20 and all(c in '0123456789' for c in core_id[:8]):
                        try:
                            # Try to decode as hex-encoded string
                            decoded = bytes.fromhex(core_id).decode('utf-8', errors='ignore')
                            # Clean non-printable chars
                            decoded = ''.join(c for c in decoded if c.isprintable())
                            if decoded:
                                return decoded
                        except:
                            pass
                
                return core_id
        
        # If no pattern matches, return cleaned version
        return market_id.lower().strip()
    
    def match_markets(self, market1_id: str, market2_id: str) -> bool:
        """Check if two market IDs refer to the same market"""
        core1 = self.extract_core_id(market1_id)
        core2 = self.extract_core_id(market2_id)
        
        if not core1 or not core2:
            return False
        
        # Direct match
        if core1 == core2:
            return True
        
        # Check if one is substring of other (for partial matches)
        if len(core1) > 10 and len(core2) > 10:
            if core1 in core2 or core2 in core1:
                return True
        
        return False
    
    def get_blockchain_address(self, market_id: str) -> Optional[str]:
        """Convert any market ID to blockchain address format"""
        core_id = self.extract_core_id(market_id)
        
        if not core_id:
            return None
        
        # If already looks like an address
        if core_id.startswith('0x') and len(core_id) == 42:
            return core_id
        
        # If it's a hex string without 0x
        if all(c in '0123456789abcdefABCDEF' for c in core_id):
            # Ensure proper length for address
            if len(core_id) < 40:
                # Pad with zeros if needed
                core_id = core_id.ljust(40, '0')
            elif len(core_id) > 40:
                # Truncate to address length
                core_id = core_id[:40]
            
            return f"0x{core_id}"
        
        return None
    
    def group_related_ids(self, market_ids: List[str]) -> List[List[str]]:
        """Group market IDs that refer to the same market"""
        groups = []
        processed = set()
        
        for i, id1 in enumerate(market_ids):
            if id1 in processed:
                continue
            
            group = [id1]
            processed.add(id1)
            
            for j, id2 in enumerate(market_ids[i+1:], i+1):
                if id2 not in processed and self.match_markets(id1, id2):
                    group.append(id2)
                    processed.add(id2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups


def test_mapper():
    """Test the market ID mapper"""
    mapper = MarketIDMapper()
    
    test_cases = [
        # Same market, different formats
        ('overtime_real_0x3230323531313033444441354644463000000000000000000000000000000000', 
         'blockchain_v2_0x3230323531313033444441354644463000000000000000000000000000000000'),
        
        # Hex encoded data
        ('overtime_real_0x3230323531313033383446324339334200000000000000000000000000000000',
         'v2_0x3230323531313033383446324339334200000000000000000000000000000000'),
        
        # Direct address
        ('0x1b06d99321576e57ee431a8f3cAa88b1c6409e86',
         '0x1b06d99321576e57ee431a8f3cAa88b1c6409e86')
    ]
    
    print("🧪 Testing Market ID Mapper\n")
    
    for id1, id2 in test_cases:
        core1 = mapper.extract_core_id(id1)
        core2 = mapper.extract_core_id(id2)
        matches = mapper.match_markets(id1, id2)
        
        print(f"ID1: {id1[:50]}...")
        print(f"Core1: {core1}")
        print(f"ID2: {id2[:50]}...")
        print(f"Core2: {core2}")
        print(f"Match: {matches}")
        print()


if __name__ == "__main__":
    test_mapper()