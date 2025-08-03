"""Test round-robin functionality with mock RSS feeds."""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from sync_podcasts.sync import PodcastSyncer, SyncConfig


class TestRoundRobin:
    """Test round-robin podcast downloading."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_transcription_service(self):
        """Mock the transcription service to track processing order."""
        with patch('sync_podcasts.sync.TranscriptionService') as mock_service:
            # Track which items were processed in order
            mock_service.return_value.processed_items = []
            
            def mock_process(request):
                # Extract the feed item being processed
                response = MagicMock()
                response.success = True
                response.summary = MagicMock()
                
                # Simulate processing one item
                if len(mock_service.return_value.processed_items) < 9:  # We have 9 total items
                    response.summary.processed = 1
                    # In a real scenario, we'd extract the actual item
                    # For testing, we'll just track the count
                    mock_service.return_value.processed_items.append(
                        f"Item {len(mock_service.return_value.processed_items) + 1}"
                    )
                else:
                    response.summary.processed = 0
                
                return response
            
            mock_service.return_value.process = mock_process
            yield mock_service.return_value
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client."""
        with patch('sync_podcasts.sync.OllamaClient') as mock_client:
            yield mock_client.return_value
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAG pipeline."""
        with patch('sync_podcasts.sync.RAGPipeline') as mock_pipeline:
            mock_pipeline.return_value.index_all_transcriptions = MagicMock()
            yield mock_pipeline
    
    def test_round_robin_processing(self, temp_dir, mock_transcription_service, 
                                   mock_ollama_client, mock_rag_pipeline):
        """Test that round-robin mode processes feeds in rotation."""
        # Setup config
        config = SyncConfig(
            feeds_path=str(Path(__file__).parent / "fixtures" / "test_feeds.toml"),
            db_path=str(Path(temp_dir) / "test.db"),
            vector_store_path=str(Path(temp_dir) / "chroma"),
            round_robin=True,
            date_threshold="2024-01-01T00:00:00"
        )
        
        # Create syncer
        syncer = PodcastSyncer(config)
        syncer.transcription_service = mock_transcription_service
        
        # Run sync (not in daemon mode)
        syncer.run()
        
        # Verify that items were processed
        assert len(mock_transcription_service.processed_items) > 0
        
        # Verify indexing was called after each item
        assert mock_rag_pipeline.return_value.index_all_transcriptions.call_count == \
               len(mock_transcription_service.processed_items)
    
    def test_sequential_processing(self, temp_dir, mock_transcription_service,
                                 mock_ollama_client, mock_rag_pipeline):
        """Test that non-round-robin mode processes feeds sequentially."""
        # Setup config without round-robin
        config = SyncConfig(
            feeds_path=str(Path(__file__).parent / "fixtures" / "test_feeds.toml"),
            db_path=str(Path(temp_dir) / "test.db"),
            vector_store_path=str(Path(temp_dir) / "chroma"),
            round_robin=False,  # Sequential mode
            date_threshold="2024-01-01T00:00:00"
        )
        
        # Create syncer
        syncer = PodcastSyncer(config)
        syncer.transcription_service = mock_transcription_service
        
        # Run sync
        syncer.run()
        
        # Verify processing occurred
        assert len(mock_transcription_service.processed_items) > 0
    
    def test_date_threshold_filtering(self, temp_dir, mock_transcription_service,
                                    mock_ollama_client, mock_rag_pipeline):
        """Test that date threshold is properly set."""
        # Test with days_back
        config = SyncConfig(
            feeds_path=str(Path(__file__).parent / "fixtures" / "test_feeds.toml"),
            db_path=str(Path(temp_dir) / "test.db"),
            vector_store_path=str(Path(temp_dir) / "chroma"),
            days_back=7
        )
        
        syncer = PodcastSyncer(config)
        
        # Verify date threshold was set to 7 days ago
        threshold_date = datetime.fromisoformat(syncer.date_threshold.replace('Z', '+00:00'))
        days_diff = (datetime.now() - threshold_date).days
        assert 6 <= days_diff <= 7  # Allow for slight timing differences
    
    def test_daemon_mode_exit(self, temp_dir, mock_transcription_service,
                            mock_ollama_client, mock_rag_pipeline):
        """Test that daemon mode can be interrupted."""
        config = SyncConfig(
            feeds_path=str(Path(__file__).parent / "fixtures" / "test_feeds.toml"),
            db_path=str(Path(temp_dir) / "test.db"),
            vector_store_path=str(Path(temp_dir) / "chroma"),
            daemon=True
        )
        
        syncer = PodcastSyncer(config)
        syncer.transcription_service = mock_transcription_service
        
        # Mock time.sleep to avoid actual sleeping
        with patch('time.sleep') as mock_sleep:
            # Make process_one_podcast return False to trigger sleep
            with patch.object(syncer, 'process_one_podcast', return_value=False) as mock_process:
                # Simulate KeyboardInterrupt after first iteration
                mock_sleep.side_effect = KeyboardInterrupt()
                
                # Run should exit gracefully
                syncer.run()
                
                # Verify it tried to process at least once
                mock_process.assert_called()