#!/usr/bin/env python3
"""Create blockchain connection mapping for Overtime markets"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import and_, func

class BlockchainConnector:
    """Connects API markets to their blockchain counterparts"""
    
    def __init__(self):
        self.connection_map = {}
    
    def create_market_connections(self):
        """Create connections between API and blockchain markets"""
        
        with db_manager.get_db_session() as db:
            # For Overtime V2, the market IDs in the API should correspond to blockchain addresses
            # The issue is that the API returns hex-encoded market IDs that need to be decoded
            
            print("🔗 Creating blockchain connections for Overtime markets...\n")
            
            # Get ALL overtime_v2 markets (these are from API imports)
            api_markets = db.query(Market).filter(
                and_(
                    Market.source == 'overtime_v2',
                    Market.sport.ilike('%soccer%')
                )
            ).all()
            
            print(f"Found {len(api_markets)} API markets to connect\n")
            
            connections_created = 0
            
            for market in api_markets:
                # Extract the game ID from the API market ID
                # Format: overtime_real_0x[encoded_game_id][padding]
                market_id = market.source_id
                
                if market_id.startswith('overtime_real_0x'):
                    # Extract hex portion
                    hex_part = market_id[16:]  # Remove 'overtime_real_0x'
                    
                    # The hex part contains the encoded game ID
                    # Try to decode it
                    try:
                        # Remove padding zeros
                        hex_clean = hex_part.rstrip('0')
                        if len(hex_clean) % 2 != 0:
                            hex_clean = hex_clean + '0'
                        
                        # This is the actual game ID that should match blockchain
                        game_id = hex_clean
                        
                        # The blockchain market address would be derived from this game ID
                        # In Overtime V2, markets are created with deterministic addresses
                        blockchain_address = self.derive_blockchain_address(game_id)
                        
                        self.connection_map[market_id] = {
                            'api_id': market_id,
                            'game_id': game_id,
                            'blockchain_address': blockchain_address,
                            'home_team': market.home_team,
                            'away_team': market.away_team,
                            'maturity_date': market.maturity_date.isoformat(),
                            'sport': market.sport
                        }
                        
                        connections_created += 1
                        
                        if connections_created <= 5:
                            print(f"Connected market {connections_created}:")
                            print(f"  Teams: {market.home_team} vs {market.away_team}")
                            print(f"  API ID: {market_id[:50]}...")
                            print(f"  Game ID: {game_id}")
                            print(f"  Blockchain: {blockchain_address}")
                            print()
                    
                    except Exception as e:
                        print(f"Error processing {market_id}: {e}")
            
            print(f"\n✅ Created {connections_created} blockchain connections")
            
            # Save connections to file
            output_file = 'blockchain_connections.json'
            with open(output_file, 'w') as f:
                json.dump(self.connection_map, f, indent=2)
            
            print(f"💾 Saved connections to {output_file}")
            
            return self.connection_map
    
    def derive_blockchain_address(self, game_id: str) -> str:
        """Derive blockchain market address from game ID"""
        # In Overtime V2, market addresses are deterministic
        # This is a simplified version - real implementation would use
        # the actual contract logic to derive addresses
        
        # For now, we'll use a placeholder format
        # Real implementation would use CREATE2 or similar deterministic addressing
        if game_id.startswith('0x'):
            game_id = game_id[2:]
        
        # Ensure proper length for an address (40 chars)
        if len(game_id) < 40:
            game_id = game_id.ljust(40, '0')
        elif len(game_id) > 40:
            game_id = game_id[:40]
        
        return f"0x{game_id}"
    
    def get_blockchain_info(self, api_market_id: str) -> Optional[Dict]:
        """Get blockchain connection info for an API market"""
        return self.connection_map.get(api_market_id)


def main():
    """Create and test blockchain connections"""
    connector = BlockchainConnector()
    connections = connector.create_market_connections()
    
    print(f"\n\n📊 Summary:")
    print(f"Total connections created: {len(connections)}")
    
    if connections:
        # Test retrieval
        test_id = list(connections.keys())[0]
        info = connector.get_blockchain_info(test_id)
        print(f"\n🧪 Test retrieval for first market:")
        print(f"  API ID: {test_id[:50]}...")
        print(f"  Blockchain: {info['blockchain_address']}")
        print(f"  Teams: {info['home_team']} vs {info['away_team']}")


if __name__ == "__main__":
    main()