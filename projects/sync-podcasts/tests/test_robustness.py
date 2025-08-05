"""Test robust processing features for sync-podcasts."""

import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import pytest

from sync_podcasts.sync import PodcastSyncer, SyncConfig, ProcessLock
from pinkhaus_models import TranscriptionDatabase


class TestProcessLock:
    """Test the process lock mechanism."""
    
    def test_single_lock_acquisition(self):
        """Test that a single process can acquire the lock."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tf:
            lock = ProcessLock(tf.name)
            
            # Should be able to acquire lock
            assert lock.acquire() is True
            
            # Release and cleanup
            lock.release()
            
            # Should be able to acquire again after release
            assert lock.acquire() is True
            lock.release()
    
    def test_multiple_lock_attempts(self):
        """Test that multiple processes cannot acquire the same lock."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tf:
            lock1 = ProcessLock(tf.name)
            lock2 = ProcessLock(tf.name)
            
            # First lock should succeed
            assert lock1.acquire() is True
            
            # Second lock should fail
            assert lock2.acquire() is False
            
            # Release first lock
            lock1.release()
            
            # Now second lock should succeed
            assert lock2.acquire() is True
            lock2.release()
    
    def test_different_feeds_different_locks(self):
        """Test that different feeds files get different locks."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tf1:
            with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tf2:
                lock1 = ProcessLock(tf1.name)
                lock2 = ProcessLock(tf2.name)
                
                # Both locks should succeed since they're for different feeds
                assert lock1.acquire() is True
                assert lock2.acquire() is True
                
                # Lock files should be different
                assert lock1.lock_file != lock2.lock_file
                
                # Cleanup
                lock1.release()
                lock2.release()


class TestRobustProcessing:
    """Test robust processing features."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def mock_transcription_service(self):
        """Mock the transcription service."""
        service = Mock()
        # Create a mock response
        response = Mock()
        response.success = True
        response.summary = Mock(processed=1, failed=0, skipped=0)
        response.errors = []
        response.results = [Mock()]
        service.process.return_value = response
        return service
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock the Ollama client."""
        return Mock()
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock the RAG pipeline."""
        pipeline = Mock()
        pipeline.index_all_transcriptions.return_value = None
        return pipeline
    
    def test_stale_processing_cleanup(self, temp_db):
        """Test that stale processing items are cleaned up."""
        config = SyncConfig(
            db_path=temp_db,
            stale_minutes=1  # Very short for testing
        )
        
        # Create database and add a stale processing item
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Add a processing item
        db.set_processing_state(
            feed_url="http://example.com/feed.rss",
            feed_item_id="episode-1",
            state="transcribing",
            feed_item_title="Test Episode",
            pid=99999,  # Non-existent PID
            hostname="old-host"
        )
        
        # Wait a bit to make it stale
        time.sleep(0.1)
        
        # Create syncer and check cleanup
        with patch('sync_podcasts.sync.TranscriptionService') as mock_service:
            with patch('sync_podcasts.sync.OllamaClient') as mock_ollama:
                syncer = PodcastSyncer(config)
                syncer._check_and_clean_stale_processing()
        
        # Verify the stale item was cleaned up
        stale_items = db.get_stale_processing_items(1)
        assert len(stale_items) == 0
    
    def test_failed_item_tracking(self, temp_db):
        """Test that failed items are tracked correctly."""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Record a failure
        db.record_failed_item(
            feed_url="http://example.com/feed.rss",
            feed_item_id="episode-1",
            error_type="DownloadError",
            error_message="Failed to download audio",
            feed_item_title="Test Episode"
        )
        
        # Check it was recorded
        failed = db.get_failed_item("http://example.com/feed.rss", "episode-1")
        assert failed is not None
        assert failed['failure_count'] == 1
        assert failed['error_type'] == "DownloadError"
        
        # Record another failure for the same item
        db.record_failed_item(
            feed_url="http://example.com/feed.rss",
            feed_item_id="episode-1",
            error_type="TranscriptionError",
            error_message="Failed to transcribe"
        )
        
        # Check count increased
        failed = db.get_failed_item("http://example.com/feed.rss", "episode-1")
        assert failed['failure_count'] == 2
        assert failed['error_type'] == "TranscriptionError"
    
    def test_skip_permanently_failed_items(self, temp_db):
        """Test that items with too many failures are skipped."""
        config = SyncConfig(
            db_path=temp_db,
            max_failures=3
        )
        
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Record 3 failures for an item
        for i in range(3):
            db.record_failed_item(
                feed_url="http://example.com/feed.rss",
                feed_item_id="episode-1",
                error_type="Error",
                error_message=f"Failure {i+1}"
            )
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                syncer = PodcastSyncer(config)
                should_skip, reason = syncer._should_skip_item(
                    "http://example.com/feed.rss", 
                    "episode-1"
                )
        
        assert should_skip is True
        assert "permanently failed" in reason
    
    def test_process_lock_prevents_multiple_instances(self, temp_db):
        """Test that process lock prevents multiple instances."""
        config = SyncConfig(db_path=temp_db)
        
        with patch('sync_podcasts.sync.TranscriptionService'):
            with patch('sync_podcasts.sync.OllamaClient'):
                syncer1 = PodcastSyncer(config)
                syncer2 = PodcastSyncer(config)
                
                # First syncer should acquire lock
                assert syncer1.process_lock.acquire() is True
                
                # Second syncer should fail to acquire lock
                assert syncer2.process_lock.acquire() is False
                
                # Release first lock
                syncer1.process_lock.release()
                
                # Now second should succeed
                assert syncer2.process_lock.acquire() is True
                syncer2.process_lock.release()
    
    def test_processing_state_tracking(self, temp_db):
        """Test that processing states are tracked."""
        db = TranscriptionDatabase(temp_db)
        db.create_tables()
        
        # Set processing state
        db.set_processing_state(
            feed_url="http://example.com/feed.rss",
            feed_item_id="episode-1",
            state="downloading",
            feed_item_title="Test Episode",
            pid=os.getpid(),
            hostname="test-host"
        )
        
        # Verify it's set
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM processing_state WHERE feed_item_id = ?",
                ("episode-1",)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["state"] == "downloading"
        
        # Clear state
        db.clear_processing_state("http://example.com/feed.rss", "episode-1")
        
        # Verify it's cleared
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM processing_state WHERE feed_item_id = ?",
                ("episode-1",)
            )
            row = cursor.fetchone()
            assert row is None