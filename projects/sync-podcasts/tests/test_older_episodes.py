"""Test sync-podcasts ability to explore older episodes when no new releases."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from sync_podcasts.sync import PodcastSyncer, SyncConfig
from beige_book.models import TranscriptionResponse, ProcessingSummary


class TestOlderEpisodes:
    """Test that sync-podcasts continues processing older episodes when no new releases exist."""

    @pytest.fixture
    def mock_transcription_responses(self):
        """Create a sequence of responses simulating no new episodes but older ones available."""
        responses = []

        # First call: Process newest episode (from before date threshold)
        response1 = MagicMock(spec=TranscriptionResponse)
        response1.success = True
        response1.summary = ProcessingSummary(
            total_items=1, processed=1, skipped=0, failed=0, elapsed_time=1.0
        )
        responses.append(response1)

        # Second call: Process second oldest episode
        response2 = MagicMock(spec=TranscriptionResponse)
        response2.success = True
        response2.summary = ProcessingSummary(
            total_items=1, processed=1, skipped=0, failed=0, elapsed_time=1.0
        )
        responses.append(response2)

        # Third call: No more episodes to process
        response3 = MagicMock(spec=TranscriptionResponse)
        response3.success = True
        response3.summary = ProcessingSummary(
            total_items=0, processed=0, skipped=0, failed=0, elapsed_time=0.5
        )
        responses.append(response3)

        return responses

    @patch("sync_podcasts.sync.RAGPipeline")
    @patch("sync_podcasts.sync.OllamaClient")
    @patch("sync_podcasts.sync.TranscriptionService")
    def test_processes_older_episodes_with_date_threshold(
        self,
        mock_service_class,
        mock_ollama_class,
        mock_rag_class,
        mock_transcription_responses,
    ):
        """Test that older episodes are processed even with a recent date threshold."""
        # Setup mock service to return our sequence of responses
        mock_service = mock_service_class.return_value
        mock_service.process.side_effect = mock_transcription_responses

        # Setup mock RAG pipeline
        mock_rag = mock_rag_class.return_value
        mock_rag.index_all_transcriptions = MagicMock()

        # Configure with a date threshold of 1 day ago
        # This simulates a scenario where all episodes are older than 1 day
        config = SyncConfig(
            feeds_path="test_feeds.toml",
            db_path="test.db",
            vector_store_path="test_chroma",
            days_back=1,  # Only look for episodes from last day
            round_robin=True,
            skip_validation=True,
        )

        # Create and run syncer
        syncer = PodcastSyncer(config)
        syncer.run()

        # Verify that process was called multiple times
        assert mock_service.process.call_count == 3

        # Verify that indexing happened for the 2 successful processes
        assert mock_rag_class.call_count == 2
        assert mock_rag.index_all_transcriptions.call_count == 2

        # Verify the date threshold was set correctly
        threshold_date = datetime.fromisoformat(
            syncer.date_threshold.replace("Z", "+00:00")
        )
        assert (datetime.now() - threshold_date).days <= 1

    @patch("sync_podcasts.sync.RAGPipeline")
    @patch("sync_podcasts.sync.OllamaClient")
    @patch("sync_podcasts.sync.TranscriptionService")
    def test_continues_processing_until_no_episodes_left(
        self, mock_service_class, mock_ollama_class, mock_rag_class
    ):
        """Test that sync continues processing episodes until none are left."""
        # Create a longer sequence of responses
        responses = []
        for i in range(5):
            response = MagicMock(spec=TranscriptionResponse)
            response.success = True
            response.summary = ProcessingSummary(
                total_items=1, processed=1, skipped=0, failed=0, elapsed_time=1.0
            )
            responses.append(response)

        # Final response with no episodes
        final_response = MagicMock(spec=TranscriptionResponse)
        final_response.success = True
        final_response.summary = ProcessingSummary(
            total_items=0, processed=0, skipped=0, failed=0, elapsed_time=0.5
        )
        responses.append(final_response)

        mock_service = mock_service_class.return_value
        mock_service.process.side_effect = responses

        mock_rag = mock_rag_class.return_value
        mock_rag.index_all_transcriptions = MagicMock()

        config = SyncConfig(
            feeds_path="test_feeds.toml",
            db_path="test.db",
            vector_store_path="test_chroma",
            date_threshold="2024-01-01T00:00:00",  # Old date to ensure all episodes are "new"
            skip_validation=True,
        )

        syncer = PodcastSyncer(config)
        syncer.run()

        # Should have processed 5 episodes + 1 final check
        assert mock_service.process.call_count == 6

        # Should have indexed 5 times (not the final empty response)
        assert mock_rag_class.call_count == 5

    @patch("sync_podcasts.sync.RAGPipeline")
    @patch("sync_podcasts.sync.OllamaClient")
    @patch("sync_podcasts.sync.TranscriptionService")
    def test_daemon_mode_continues_checking_older_episodes(
        self, mock_service_class, mock_ollama_class, mock_rag_class
    ):
        """Test that daemon mode continues to check for older episodes after processing new ones."""
        # Simulate: new episode -> no new episodes -> older episode found -> no episodes
        responses = [
            # First check: new episode
            MagicMock(
                success=True,
                summary=ProcessingSummary(
                    total_items=1, processed=1, skipped=0, failed=0, elapsed_time=1.0
                ),
            ),
            # Second check: no new episodes
            MagicMock(
                success=True,
                summary=ProcessingSummary(
                    total_items=0, processed=0, skipped=0, failed=0, elapsed_time=0.5
                ),
            ),
            # Third check (after sleep): older episode found
            MagicMock(
                success=True,
                summary=ProcessingSummary(
                    total_items=1, processed=1, skipped=0, failed=0, elapsed_time=1.0
                ),
            ),
            # Fourth check: no episodes left
            MagicMock(
                success=True,
                summary=ProcessingSummary(
                    total_items=0, processed=0, skipped=0, failed=0, elapsed_time=0.5
                ),
            ),
        ]

        mock_service = mock_service_class.return_value
        mock_service.process.side_effect = responses

        mock_rag = mock_rag_class.return_value
        mock_rag.index_all_transcriptions = MagicMock()

        config = SyncConfig(
            feeds_path="test_feeds.toml",
            db_path="test.db",
            vector_store_path="test_chroma",
            daemon=True,
            days_back=7,
            skip_validation=True,
        )

        syncer = PodcastSyncer(config)

        # Mock sleep and interrupt after 4 iterations
        call_count = 0

        def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Allow 2 sleep cycles
                raise KeyboardInterrupt()

        with patch("time.sleep", side_effect=mock_sleep):
            syncer.run()

        # Should have made 4 process calls before interruption
        assert mock_service.process.call_count == 4

        # Should have indexed 2 times (for the 2 successful processes)
        assert mock_rag_class.call_count == 2

    def test_date_threshold_behavior_documented(self):
        """Test that verifies the date threshold is used correctly.

        The date threshold should be used to filter episodes, but the system
        should continue processing older episodes that exist before the threshold
        if no new episodes are available.
        """
        # This is more of a documentation test to ensure understanding
        config = SyncConfig(days_back=7, skip_validation=True)
        syncer = PodcastSyncer(config)

        # Verify date threshold is set to 7 days ago
        threshold = datetime.fromisoformat(syncer.date_threshold.replace("Z", "+00:00"))
        days_diff = (datetime.now() - threshold).days
        assert 6 <= days_diff <= 7

        # The actual behavior verification is in the integration tests above
