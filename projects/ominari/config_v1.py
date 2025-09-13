#!/usr/bin/env python3
"""
Ominari V1 Configuration
Simple, locked configuration for v1: API for data/quotes, blockchain for trades only.
"""

from typing import Dict, Any, Literal
from pydantic import Field
from pydantic_settings import BaseSettings

class V1Config(BaseSettings):
    """Locked v1 configuration."""
    
    # Data Sources (v1 uses API only for market data)
    use_overtime_api: Literal[True] = True
    use_blockchain: Literal[True] = True  # Only for trade monitoring
    use_graphql: Literal[False] = False   # Disabled in v1
    
    # API Configuration
    overtime_api_url: str = Field(
        default="https://api.overtime.io/overtime-v2",
        env="OVERTIME_API_URL"
    )
    overtime_api_key: str = Field(env="OVERTIME_API_KEY")
    
    # Blockchain Configuration (for trade monitoring only)
    blockchain_network: Literal["optimism"] = "optimism"
    sports_amm_v2_address: Literal["0xFb4e4811C7A811E098A556bD79B64c20b479E431"] = "0xFb4e4811C7A811E098A556bD79B64c20b479E431"
    rpc_url: str = Field(
        default="https://mainnet.optimism.io",
        env="OPTIMISM_RPC_URL"
    )
    
    # Update Frequencies (in minutes)
    api_update_frequency: int = Field(default=5, env="API_UPDATE_FREQ")
    blockchain_scan_frequency: int = Field(default=5, env="BLOCKCHAIN_SCAN_FREQ")
    
    # Trading Configuration
    paper_trading_enabled: bool = Field(default=True, env="PAPER_TRADING")
    live_trading_enabled: bool = Field(default=False, env="LIVE_TRADING")
    
    # Data Filters
    sports_filter: list = Field(default=["Soccer", "Football"])
    min_market_liquidity: float = Field(default=0.0)  # No liquidity filter in v1
    
    # Signal Configuration
    use_external_signals: bool = Field(default=False)  # Start with internal only
    signal_update_frequency: int = Field(default=5)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env
    
    def get_data_flow(self) -> Dict[str, Any]:
        """Get v1 data flow configuration."""
        return {
            "market_data": {
                "source": "overtime_api",
                "endpoint": f"{self.overtime_api_url}/networks/10/markets",
                "frequency_minutes": self.api_update_frequency
            },
            "quotes": {
                "source": "overtime_api",
                "endpoint": f"{self.overtime_api_url}/networks/10/quotes",
                "real_time": True
            },
            "trades": {
                "source": "blockchain",
                "network": self.blockchain_network,
                "contract": self.sports_amm_v2_address,
                "events": ["BoughtFromAmm", "SoldToAmm"],
                "frequency_minutes": self.blockchain_scan_frequency
            },
            "execution": {
                "paper_trading": self.paper_trading_enabled,
                "live_trading": self.live_trading_enabled
            }
        }
    
    def validate_v1_setup(self) -> bool:
        """Validate v1 configuration is correct."""
        checks = {
            "api_enabled": self.use_overtime_api,
            "api_key_set": bool(self.overtime_api_key),
            "blockchain_trades_only": self.use_blockchain and not self.use_graphql,
            "correct_contract": self.sports_amm_v2_address == "0xFb4e4811C7A811E098A556bD79B64c20b479E431",
            "paper_or_live": self.paper_trading_enabled or self.live_trading_enabled
        }
        
        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"❌ V1 validation failed: {failed}")
            return False
        
        print("✅ V1 configuration validated")
        return True


# Singleton instance
v1_config = V1Config()

if __name__ == "__main__":
    print("=== Ominari V1 Configuration ===")
    print("\nData Sources:")
    print("  Market Data: Overtime API")
    print("  Quotes: Overtime API") 
    print(f"  Trade Monitoring: Blockchain ({v1_config.blockchain_network})")
    print("  GraphQL: Disabled")
    
    print("\nUpdate Frequencies:")
    print(f"  API: Every {v1_config.api_update_frequency} minutes")
    print(f"  Blockchain: Every {v1_config.blockchain_scan_frequency} minutes")
    
    print("\nTrading Mode:")
    print(f"  Paper Trading: {v1_config.paper_trading_enabled}")
    print(f"  Live Trading: {v1_config.live_trading_enabled}")
    
    print("\nValidation:")
    v1_config.validate_v1_setup()
    
    print("\nData Flow:")
    import json
    print(json.dumps(v1_config.get_data_flow(), indent=2))