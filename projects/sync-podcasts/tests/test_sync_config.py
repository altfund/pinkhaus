"""Test sync configuration."""

import pytest
from sync_podcasts.sync import SyncConfig


class TestSyncConfig:
    """Test SyncConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SyncConfig()
        
        assert config.feeds_path == "./feeds.toml"
        assert config.db_path == "./podcasts.db"
        assert config.vector_store_path == "./grant_chroma_db"
        assert config.model == "tiny"
        assert config.round_robin is True
        assert config.daemon is False
        assert config.date_threshold is None
        assert config.days_back is None
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SyncConfig(
            feeds_path="/custom/feeds.toml",
            db_path="/custom/db.sqlite",
            vector_store_path="/custom/chroma",
            model="large",
            round_robin=False,
            daemon=True,
            date_threshold="2024-01-01T00:00:00Z",
            days_back=30,
            ollama_base_url="http://custom:11434",
            verbose=True
        )
        
        assert config.feeds_path == "/custom/feeds.toml"
        assert config.db_path == "/custom/db.sqlite"
        assert config.vector_store_path == "/custom/chroma"
        assert config.model == "large"
        assert config.round_robin is False
        assert config.daemon is True
        assert config.date_threshold == "2024-01-01T00:00:00Z"
        assert config.days_back == 30
        assert config.ollama_base_url == "http://custom:11434"
        assert config.verbose is True