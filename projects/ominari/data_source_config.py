#!/usr/bin/env python3
"""
Data Source Configuration for easy switching between environments.
"""

import os
from enum import Enum
from typing import List, Dict, Any

class DataSourceMode(Enum):
    """Data source modes for different environments."""
    DEVELOPMENT = "development"  # Use cached/historical data
    TESTING = "testing"         # Use REST API only
    STAGING = "staging"         # Use REST API + Blockchain
    PRODUCTION = "production"   # All sources with live trading

class DataSourceConfig:
    """Configuration for data sources based on environment."""
    
    def __init__(self, mode: str = None):
        self.mode = DataSourceMode(mode or os.getenv('DATA_SOURCE_MODE', 'testing'))
        
    @property
    def sources(self) -> Dict[str, bool]:
        """Get enabled data sources for current mode."""
        configs = {
            DataSourceMode.DEVELOPMENT: {
                'rest_api': True,
                'blockchain': False,
                'graphql': False,
                'use_cache': True,
                'paper_trading': True,
                'live_trading': False
            },
            DataSourceMode.TESTING: {
                'rest_api': True,
                'blockchain': False,
                'graphql': False,
                'use_cache': False,
                'paper_trading': True,
                'live_trading': False
            },
            DataSourceMode.STAGING: {
                'rest_api': True,
                'blockchain': True,
                'graphql': False,  # Until we find new endpoints
                'use_cache': False,
                'paper_trading': True,
                'live_trading': False
            },
            DataSourceMode.PRODUCTION: {
                'rest_api': True,
                'blockchain': True,
                'graphql': False,  # Until we find new endpoints
                'use_cache': False,
                'paper_trading': False,
                'live_trading': True
            }
        }
        return configs[self.mode]
    
    @property
    def update_frequency(self) -> Dict[str, int]:
        """Get update frequencies in minutes."""
        if self.mode == DataSourceMode.DEVELOPMENT:
            return {
                'rest_api': 60,      # Less frequent in dev
                'blockchain': 0,     # Disabled
                'signals': 60
            }
        elif self.mode == DataSourceMode.PRODUCTION:
            return {
                'rest_api': 5,       # Every 5 minutes
                'blockchain': 2,     # Every 2 minutes
                'signals': 1         # Every minute
            }
        else:  # Testing/Staging
            return {
                'rest_api': 5,
                'blockchain': 5,
                'signals': 5
            }
    
    @property
    def filters(self) -> Dict[str, Any]:
        """Get data filters for current mode."""
        if self.mode == DataSourceMode.DEVELOPMENT:
            return {
                'sports': ['Soccer'],  # Limited sports in dev
                'min_liquidity': 0,
                'max_markets': 100
            }
        elif self.mode == DataSourceMode.PRODUCTION:
            return {
                'sports': ['Soccer', 'Football', 'Basketball', 'Baseball'],
                'min_liquidity': 1000,  # Only liquid markets
                'max_markets': None
            }
        else:
            return {
                'sports': ['Soccer', 'Football'],
                'min_liquidity': 0,
                'max_markets': 500
            }
    
    def get_active_sources(self) -> List[str]:
        """Get list of active data sources."""
        return [source for source, enabled in self.sources.items() 
                if enabled and source not in ['paper_trading', 'live_trading', 'use_cache']]
    
    def should_use_source(self, source: str) -> bool:
        """Check if a specific source should be used."""
        return self.sources.get(source, False)
    
    def __str__(self):
        return f"DataSourceConfig(mode={self.mode.value}, sources={self.get_active_sources()})"


# Usage example
if __name__ == "__main__":
    # Show configurations for each mode
    for mode in DataSourceMode:
        config = DataSourceConfig(mode.value)
        print(f"\n{mode.value.upper()} Configuration:")
        print(f"  Active sources: {config.get_active_sources()}")
        print(f"  Paper trading: {config.sources['paper_trading']}")
        print(f"  Live trading: {config.sources['live_trading']}")
        print(f"  Update frequencies: {config.update_frequency}")
        print(f"  Sports filter: {config.filters['sports']}")