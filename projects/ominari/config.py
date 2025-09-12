#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Configuration Management
Uses Pydantic for validation and environment variable loading.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    url: str = Field(default="sqlite:///sport_odds.db", env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DB_ECHO")
    
    model_config = ConfigDict(env_prefix="DB_")


class RedisConfig(BaseSettings):
    """Redis configuration."""
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    ttl: int = Field(default=3600, env="REDIS_TTL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    model_config = ConfigDict(env_prefix="REDIS_")


class BlockchainConfig(BaseSettings):
    """Blockchain configuration."""
    network: str = Field(default="optimism", env="BLOCKCHAIN_NETWORK")
    rpc_url: str = Field(default="https://mainnet.optimism.io", env="RPC_URL")
    chain_id: int = Field(default=10, env="CHAIN_ID")
    block_confirmations: int = Field(default=3, env="BLOCK_CONFIRMATIONS")
    gas_multiplier: float = Field(default=1.2, env="GAS_MULTIPLIER")
    
    # Contract addresses
    sports_amm_v2: str = Field(
        default="0xFb4e4811C7A811E098A556bD79B64c20b479E431",  # Updated to correct SportsAMMV2 address
        env="SPORTS_AMM_V2_ADDRESS"
    )
    usdc_address: str = Field(
        default="0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
        env="USDC_ADDRESS"
    )
    
    model_config = ConfigDict(env_prefix="BLOCKCHAIN_")


class SignalConfig(BaseSettings):
    """Signal system configuration."""
    default_lookback_days: int = Field(default=30, env="SIGNAL_LOOKBACK_DAYS")
    min_edge_threshold: float = Field(default=0.01, env="MIN_EDGE_THRESHOLD")
    max_position_size: float = Field(default=0.25, env="MAX_POSITION_SIZE")
    signal_timeout_seconds: int = Field(default=30, env="SIGNAL_TIMEOUT")
    
    # Weight calculation
    weight_method: str = Field(default="bayesian", env="WEIGHT_METHOD")
    weight_update_frequency: int = Field(default=3600, env="WEIGHT_UPDATE_FREQ")
    
    model_config = ConfigDict(env_prefix="SIGNAL_")


class TradingConfig(BaseSettings):
    """Trading configuration."""
    initial_capital: float = Field(default=10000.0, env="INITIAL_CAPITAL")
    volatility_target: float = Field(default=0.16, env="VOLATILITY_TARGET")
    max_leverage: float = Field(default=1.0, env="MAX_LEVERAGE")
    
    # Risk limits
    max_drawdown: float = Field(default=0.20, env="MAX_DRAWDOWN")
    position_limit: int = Field(default=10, env="POSITION_LIMIT")
    daily_loss_limit: float = Field(default=0.05, env="DAILY_LOSS_LIMIT")
    
    # Execution
    commission_rate: float = Field(default=0.002, env="COMMISSION_RATE")
    slippage_factor: float = Field(default=0.001, env="SLIPPAGE_FACTOR")
    min_bet_size: float = Field(default=10.0, env="MIN_BET_SIZE")
    
    # Sports filter - focused on soccer for now
    allowed_sports: List[str] = Field(
        default=["Soccer", "Football", "EPL", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "UEFA", "FIFA"],
        env="ALLOWED_SPORTS"
    )
    sport_filter_enabled: bool = Field(default=True, env="SPORT_FILTER_ENABLED")
    
    @field_validator("allowed_sports", mode='before')
    @classmethod
    def parse_allowed_sports(cls, v):
        if isinstance(v, str):
            return [sport.strip() for sport in v.split(",")]
        return v
    
    model_config = ConfigDict(env_prefix="TRADING_")


class AlphaResearchConfig(BaseSettings):
    """Alpha research configuration."""
    min_samples_raw_rd: int = Field(default=1000, env="ALPHA_MIN_SAMPLES_RAW")
    min_samples_in_sample: int = Field(default=5000, env="ALPHA_MIN_SAMPLES_IS")
    min_samples_out_sample: int = Field(default=2000, env="ALPHA_MIN_SAMPLES_OOS")
    
    # Success thresholds
    min_sharpe_raw: float = Field(default=0.5, env="ALPHA_MIN_SHARPE_RAW")
    min_sharpe_production: float = Field(default=0.7, env="ALPHA_MIN_SHARPE_PROD")
    max_p_value: float = Field(default=0.05, env="ALPHA_MAX_P_VALUE")
    
    model_config = ConfigDict(env_prefix="ALPHA_")


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # APM
    apm_enabled: bool = Field(default=False, env="APM_ENABLED")
    datadog_api_key: Optional[str] = Field(default=None, env="DATADOG_API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Alerting
    pagerduty_key: Optional[str] = Field(default=None, env="PAGERDUTY_KEY")
    slack_webhook: Optional[str] = Field(default=None, env="SLACK_WEBHOOK")
    
    model_config = ConfigDict(env_prefix="MONITORING_")


class APIConfig(BaseSettings):
    """API configuration."""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # API keys
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    require_api_key: bool = Field(default=False, env="REQUIRE_API_KEY")
    
    @field_validator("cors_origins", mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    model_config = ConfigDict(env_prefix="API_")


class ExternalAPIConfig(BaseSettings):
    """External API configuration."""
    overtime_api_url: str = Field(
        default="https://overtimemarketsv2.xyz",
        env="OVERTIME_API_URL"
    )
    overtime_api_key: Optional[str] = Field(default=None, env="OVERTIME_API_KEY")
    
    graphql_endpoints: Dict[str, str] = Field(
        default={
            "optimism": "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism",
            "arbitrum": "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum"
        }
    )
    
    # Rate limits
    api_rate_limit: int = Field(default=10, env="EXTERNAL_API_RATE_LIMIT")
    api_timeout: int = Field(default=30, env="EXTERNAL_API_TIMEOUT")
    
    model_config = ConfigDict(env_prefix="EXTERNAL_")


class FeatureFlags(BaseSettings):
    """Feature flags configuration."""
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    live_trading: bool = Field(default=False, env="LIVE_TRADING")
    experimental_signals: bool = Field(default=False, env="EXPERIMENTAL")
    debug_mode: bool = Field(default=False, env="DEBUG")
    
    # Advanced features
    multi_account: bool = Field(default=False, env="MULTI_ACCOUNT")
    blockchain_sync: bool = Field(default=True, env="BLOCKCHAIN_SYNC")
    graphql_streaming: bool = Field(default=False)  # Disabled - The Graph endpoints have moved
    
    model_config = ConfigDict(
        env_prefix="FEATURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


class Settings(BaseSettings):
    """Main settings class aggregating all configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Component configs
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    blockchain: BlockchainConfig = BlockchainConfig()
    signals: SignalConfig = SignalConfig()
    trading: TradingConfig = TradingConfig()
    alpha: AlphaResearchConfig = AlphaResearchConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()
    external_apis: ExternalAPIConfig = ExternalAPIConfig()
    features: FeatureFlags = FeatureFlags()
    
    # Paths
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    log_dir: Path = Field(default=Path("logs"), env="LOG_DIR")
    report_dir: Path = Field(default=Path("reports"), env="REPORT_DIR")
    
    @field_validator("data_dir", "log_dir", "report_dir")
    @classmethod
    def create_directories(cls, v):
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        db_url = self.database.url
        
        # Add options for production
        if self.is_production and "postgresql" in db_url:
            if "?" not in db_url:
                db_url += "?"
            else:
                db_url += "&"
            db_url += "sslmode=require"
            
        return db_url
    
    def get_redis_url(self) -> str:
        """Get Redis URL with auth if needed."""
        return self.redis.url
    
    def validate_config(self) -> bool:
        """Validate configuration consistency."""
        errors = []
        
        # Check feature dependencies
        if self.features.live_trading and not self.features.paper_trading:
            errors.append("Live trading requires paper trading to be enabled")
            
        if self.features.live_trading and self.trading.initial_capital < 1000:
            errors.append("Live trading requires at least $1000 initial capital")
            
        if self.is_production and self.debug:
            errors.append("Debug mode should not be enabled in production")
            
        if self.blockchain.chain_id not in [10, 42161, 11155420]:
            errors.append(f"Unknown chain ID: {self.blockchain.chain_id}")
            
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "environment": self.environment,
            "database": self.database.model_dump(),
            "redis": self.redis.model_dump(),
            "blockchain": self.blockchain.model_dump(),
            "signals": self.signals.model_dump(),
            "trading": self.trading.model_dump(),
            "alpha": self.alpha.model_dump(),
            "monitoring": self.monitoring.model_dump(),
            "api": self.api.model_dump(),
            "features": self.features.model_dump()
        }
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"  # Allow extra fields from environment
    )


# Global settings instance
settings = Settings()

# Validate on import
if not settings.validate_config():
    logger.warning("Configuration validation failed - check logs")


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings


def reload_settings():
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings