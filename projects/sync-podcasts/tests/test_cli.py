"""Test CLI interface."""

import pytest
from unittest.mock import patch, MagicMock

from sync_podcasts.cli import main, setup_logging
from sync_podcasts.validate_feed import FeedValidationResult


class TestCLI:
    """Test command-line interface."""

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose flag."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(verbose=True)
            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs["level"] == 10  # logging.DEBUG

    def test_setup_logging_normal(self):
        """Test logging setup without verbose flag."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(verbose=False)
            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs["level"] == 20  # logging.INFO

    def test_main_help(self):
        """Test help output."""
        with patch("sys.argv", ["sync-podcasts", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_missing_feeds_file(self):
        """Test error when feeds file doesn't exist."""
        with patch("sys.argv", ["sync-podcasts", "--feeds", "/nonexistent/feeds.toml"]):
            with patch("sys.stderr.write"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    @patch("sync_podcasts.cli.Path")
    @patch("sync_podcasts.cli.PodcastSyncer")
    def test_main_successful_run(self, mock_syncer_class, mock_path):
        """Test successful run with valid arguments."""
        # Mock that feeds file exists
        mock_path.return_value.exists.return_value = True

        # Mock syncer
        mock_syncer = MagicMock()
        mock_syncer_class.return_value = mock_syncer

        with patch(
            "sys.argv", ["sync-podcasts", "--feeds", "feeds.toml", "--days", "7"]
        ):
            main()

        # Verify syncer was created and run
        mock_syncer_class.assert_called_once()
        mock_syncer.run.assert_called_once()

        # Check config
        config = mock_syncer_class.call_args[0][0]
        assert config.feeds_path == "feeds.toml"
        assert config.days_back == 7

    @patch("sync_podcasts.cli.Path")
    def test_main_with_invalid_feeds(self, mock_path):
        """Test that sync-podcasts exits when feeds are invalid."""
        # Mock that feeds file exists
        mock_path.return_value.exists.return_value = True

        # Mock validation to return invalid feeds
        with patch("sync_podcasts.sync.validate_feeds_toml") as mock_validate:
            mock_validate.return_value = {
                "https://example.com/invalid.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="No audio content found",
                    feed_title="Blog Feed",
                ),
                "https://example.com/broken.rss": FeedValidationResult(
                    is_valid=False, error_message="Network error: Connection refused"
                ),
            }

            # Mock other dependencies
            with patch("sync_podcasts.sync.TranscriptionService"):
                with patch("sync_podcasts.sync.OllamaClient"):
                    with patch("sync_podcasts.sync.TranscriptionDatabase"):
                        with patch(
                            "sys.argv", ["sync-podcasts", "--feeds", "feeds.toml"]
                        ):
                            with patch("sys.stderr.write"):
                                # Should exit with error code 1
                                with pytest.raises(SystemExit) as exc_info:
                                    main()
                                assert exc_info.value.code == 1
