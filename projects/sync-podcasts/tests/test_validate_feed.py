"""Tests for feed validation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import toml
from pathlib import Path

from sync_podcasts.validate_feed import (
    FeedValidator,
    FeedValidationResult,
    validate_feed_url,
    validate_feeds_toml,
    update_toml_feeds,
    create_invalid_feeds_toml,
)


class TestFeedValidator:
    """Test the FeedValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a FeedValidator instance."""
        return FeedValidator()

    def test_valid_rss_feed_with_audio(self, validator):
        """Test validation of a valid RSS feed with audio content."""
        # Mock feedparser response
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.version = "rss20"
        mock_feed.get.return_value = {
            "title": "Test Podcast",
            "description": "A test podcast",
        }

        # Create mock entries with audio
        entry1 = MagicMock()
        entry1.enclosures = [
            {"type": "audio/mpeg", "href": "http://example.com/ep1.mp3"}
        ]
        entry1.get.return_value = "Episode 1"

        entry2 = MagicMock()
        entry2.enclosures = [
            {"type": "audio/mp3", "href": "http://example.com/ep2.mp3"}
        ]
        entry2.get.return_value = "Episode 2"

        mock_feed.entries = [entry1, entry2]

        # Mock FeedParser.parse_feed to return items
        with patch("feedparser.parse", return_value=mock_feed):
            with patch.object(validator.feed_parser, "parse_feed") as mock_parse:
                mock_parse.return_value = [Mock(), Mock()]  # 2 audio items

                result = validator.validate_feed("http://example.com/feed.rss")

        assert result.is_valid is True
        assert result.feed_title == "Test Podcast"
        assert result.feed_description == "A test podcast"
        assert result.total_entries == 2
        assert result.audio_entries == 2
        assert result.feed_type == "RSS"
        assert result.error_message is None

    def test_valid_atom_feed_with_audio(self, validator):
        """Test validation of a valid Atom feed."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.version = "atom10"
        mock_feed.get.return_value = {
            "title": "Atom Podcast",
            "description": "An atom feed",
        }
        mock_feed.entries = [MagicMock()]

        with patch("feedparser.parse", return_value=mock_feed):
            with patch.object(validator.feed_parser, "parse_feed") as mock_parse:
                mock_parse.return_value = [Mock()]  # 1 audio item

                result = validator.validate_feed("http://example.com/feed.atom")

        assert result.is_valid is True
        assert result.feed_type == "Atom"

    def test_feed_without_audio_content(self, validator):
        """Test validation of a feed without audio content."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.get.return_value = {"title": "Blog Feed"}
        mock_feed.entries = [MagicMock(), MagicMock()]

        with patch("feedparser.parse", return_value=mock_feed):
            with patch.object(validator.feed_parser, "parse_feed") as mock_parse:
                mock_parse.return_value = []  # No audio items

                result = validator.validate_feed("http://example.com/blog.rss")

        assert result.is_valid is False
        assert result.total_entries == 2
        assert result.audio_entries == 0
        assert "no audio content found" in result.error_message

    def test_empty_feed(self, validator):
        """Test validation of an empty feed."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.get.return_value = {"title": "Empty Feed"}
        mock_feed.entries = []

        with patch("feedparser.parse", return_value=mock_feed):
            result = validator.validate_feed("http://example.com/empty.rss")

        assert result.is_valid is False
        assert result.total_entries == 0
        assert "Feed contains no entries" in result.error_message

    def test_network_error(self, validator):
        """Test handling of network errors."""
        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception(
            "URLError: <urlopen error [Errno -2] Name or service not known>"
        )

        with patch("feedparser.parse", return_value=mock_feed):
            result = validator.validate_feed("http://invalid.example.com/feed.rss")

        assert result.is_valid is False
        assert "Network error" in result.error_message

    def test_invalid_xml(self, validator):
        """Test handling of invalid XML/non-RSS content."""
        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("not well-formed (invalid token)")

        with patch("feedparser.parse", return_value=mock_feed):
            result = validator.validate_feed("http://example.com/notxml.html")

        assert result.is_valid is False
        assert "not appear to be a valid RSS/XML feed" in result.error_message

    def test_no_feed_data(self, validator):
        """Test handling when no feed data is returned."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        # Remove feed and entries attributes
        del mock_feed.feed
        del mock_feed.entries

        with patch("feedparser.parse", return_value=mock_feed):
            result = validator.validate_feed("http://example.com/empty")

        assert result.is_valid is False
        assert "No feed data found" in result.error_message

    def test_feed_parser_exception(self, validator):
        """Test handling of exceptions from FeedParser."""
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.get.return_value = {"title": "Test Feed"}
        mock_feed.entries = [MagicMock()]

        with patch("feedparser.parse", return_value=mock_feed):
            with patch.object(validator.feed_parser, "parse_feed") as mock_parse:
                mock_parse.side_effect = Exception("Audio parsing error")

                result = validator.validate_feed("http://example.com/feed.rss")

        assert result.is_valid is False
        assert "Error checking for audio content" in result.error_message

    def test_unexpected_exception(self, validator):
        """Test handling of unexpected exceptions."""
        with patch("feedparser.parse", side_effect=Exception("Unexpected error")):
            result = validator.validate_feed("http://example.com/feed.rss")

        assert result.is_valid is False
        assert "Unexpected error" in result.error_message

    def test_validate_feed_url_function_valid(self):
        """Test the validate_feed_url convenience function with valid feed."""
        with patch(
            "sync_podcasts.validate_feed.FeedValidator.validate_feed"
        ) as mock_validate:
            mock_validate.return_value = FeedValidationResult(
                is_valid=True,
                feed_title="Test Podcast",
                feed_type="RSS",
                audio_entries=10,
            )

            is_valid, message = validate_feed_url("http://example.com/feed.rss")

        assert is_valid is True
        assert "Valid RSS feed: Test Podcast (10 audio episodes)" in message

    def test_validate_feed_url_function_invalid(self):
        """Test the validate_feed_url convenience function with invalid feed."""
        with patch(
            "sync_podcasts.validate_feed.FeedValidator.validate_feed"
        ) as mock_validate:
            mock_validate.return_value = FeedValidationResult(
                is_valid=False, error_message="Network error"
            )

            is_valid, message = validate_feed_url("http://example.com/feed.rss")

        assert is_valid is False
        assert message == "Network error"

    def test_long_description_truncation(self, validator):
        """Test that long descriptions are truncated in results."""
        long_description = "A" * 200
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.get.return_value = {
            "title": "Test Feed",
            "description": long_description,
        }
        mock_feed.entries = [MagicMock()]

        with patch("feedparser.parse", return_value=mock_feed):
            with patch.object(validator.feed_parser, "parse_feed") as mock_parse:
                mock_parse.return_value = [Mock()]

                result = validator.validate_feed("http://example.com/feed.rss")

        assert result.is_valid is True
        assert result.feed_description == long_description
        assert len(result.feed_description) == 200


class TestValidateFeedsToml:
    """Test the validate_feeds_toml function."""

    def test_validate_multiple_feeds_from_toml(self):
        """Test validating multiple feeds from a TOML file."""
        # Create temporary TOML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/feed1.rss",
    "https://example.com/feed2.rss",
    "https://example.com/feed3.rss"
]
""")
            toml_path = f.name

        try:
            # Mock the validator
            with patch(
                "sync_podcasts.validate_feed.FeedValidator.validate_feed"
            ) as mock_validate:
                # Setup different results for each feed
                mock_validate.side_effect = [
                    FeedValidationResult(
                        is_valid=True,
                        feed_title="Podcast 1",
                        feed_type="RSS",
                        audio_entries=10,
                    ),
                    FeedValidationResult(
                        is_valid=False, error_message="No audio content"
                    ),
                    FeedValidationResult(
                        is_valid=True,
                        feed_title="Podcast 3",
                        feed_type="Atom",
                        audio_entries=5,
                    ),
                ]

                results = validate_feeds_toml(toml_path)

            # Verify results
            assert len(results) == 3
            assert results["https://example.com/feed1.rss"].is_valid is True
            assert results["https://example.com/feed2.rss"].is_valid is False
            assert results["https://example.com/feed3.rss"].is_valid is True

        finally:
            os.unlink(toml_path)

    def test_invalid_toml_file(self):
        """Test handling of invalid TOML file."""
        # Create temporary invalid TOML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml [[ content")
            toml_path = f.name

        try:
            results = validate_feeds_toml(toml_path)

            # Should have error result
            assert "_error" in results
            assert results["_error"].is_valid is False
            assert "Error parsing TOML file" in results["_error"].error_message

        finally:
            os.unlink(toml_path)

    def test_missing_feeds_in_toml(self):
        """Test handling of TOML file without feeds."""
        # Create TOML without feeds
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[other]
data = "value"
""")
            toml_path = f.name

        try:
            results = validate_feeds_toml(toml_path)

            # Should have error about missing feeds
            assert "_error" in results
            assert results["_error"].is_valid is False
            assert "No RSS feeds found" in results["_error"].error_message

        finally:
            os.unlink(toml_path)

    def test_empty_feeds_list(self):
        """Test handling of empty feeds list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[feeds]
rss = []
""")
            toml_path = f.name

        try:
            results = validate_feeds_toml(toml_path)

            # Should return empty results dict
            assert len(results) == 0

        finally:
            os.unlink(toml_path)

    def test_move_invalid_feeds_to_separate_file(self):
        """Test moving invalid feeds to invalid-feeds.toml."""
        # Create temporary TOML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/valid1.rss",
    "https://example.com/invalid.rss",
    "https://example.com/valid2.rss",
    "https://example.com/broken.rss"
]
""")
            toml_path = f.name

        try:
            # Mock the validator
            with patch(
                "sync_podcasts.validate_feed.FeedValidator.validate_feed"
            ) as mock_validate:
                # Setup different results for each feed
                mock_validate.side_effect = [
                    FeedValidationResult(
                        is_valid=True,
                        feed_title="Valid Podcast 1",
                        feed_type="RSS",
                        audio_entries=10,
                    ),
                    FeedValidationResult(
                        is_valid=False,
                        error_message="No audio content found",
                        feed_title="Blog Feed",
                    ),
                    FeedValidationResult(
                        is_valid=True,
                        feed_title="Valid Podcast 2",
                        feed_type="RSS",
                        audio_entries=5,
                    ),
                    FeedValidationResult(
                        is_valid=False,
                        error_message="Network error: Connection refused",
                    ),
                ]

                # Call with move_invalid=True
                results = validate_feeds_toml(toml_path, move_invalid=True)

            # Verify results
            assert len(results) == 4

            # Check that original file now contains only valid feeds
            with open(toml_path, "r") as f:
                data = toml.load(f)
            assert len(data["feeds"]["rss"]) == 2
            assert "https://example.com/valid1.rss" in data["feeds"]["rss"]
            assert "https://example.com/valid2.rss" in data["feeds"]["rss"]
            assert "https://example.com/invalid.rss" not in data["feeds"]["rss"]
            assert "https://example.com/broken.rss" not in data["feeds"]["rss"]

            # Check that invalid-feeds.toml was created
            invalid_path = os.path.join(
                os.path.dirname(toml_path), "invalid-feeds.toml"
            )
            assert os.path.exists(invalid_path)

            # Check contents of invalid-feeds.toml
            with open(invalid_path, "r") as f:
                content = f.read()
            assert "https://example.com/invalid.rss" in content
            assert "https://example.com/broken.rss" in content
            assert "No audio content found" in content
            assert "Network error: Connection refused" in content
            assert "Blog Feed" in content

            # Cleanup
            os.unlink(invalid_path)

        finally:
            os.unlink(toml_path)

    def test_no_invalid_feeds_no_file_created(self):
        """Test that no invalid-feeds.toml is created when all feeds are valid."""
        # Create temporary TOML file with all valid feeds
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/valid1.rss",
    "https://example.com/valid2.rss"
]
""")
            toml_path = f.name

        try:
            # Mock the validator to return all valid
            with patch(
                "sync_podcasts.validate_feed.FeedValidator.validate_feed"
            ) as mock_validate:
                mock_validate.return_value = FeedValidationResult(
                    is_valid=True,
                    feed_title="Valid Podcast",
                    feed_type="RSS",
                    audio_entries=10,
                )

                # Call with move_invalid=True
                _ = validate_feeds_toml(toml_path, move_invalid=True)

            # Check that original file is unchanged
            with open(toml_path, "r") as f:
                data = toml.load(f)
            assert len(data["feeds"]["rss"]) == 2

            # Check that no invalid-feeds.toml was created
            invalid_path = os.path.join(
                os.path.dirname(toml_path), "invalid-feeds.toml"
            )
            assert not os.path.exists(invalid_path)

        finally:
            os.unlink(toml_path)


class TestTomlHelpers:
    """Test the TOML helper functions."""

    def test_update_toml_feeds(self):
        """Test updating TOML file with valid feeds."""
        # Create temporary TOML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[feeds]
rss = [
    "https://example.com/feed1.rss",
    "https://example.com/feed2.rss",
    "https://example.com/feed3.rss"
]

[other]
setting = "value"
""")
            toml_path = f.name

        try:
            # Update with only two feeds
            update_toml_feeds(
                toml_path,
                ["https://example.com/feed1.rss", "https://example.com/feed3.rss"],
            )

            # Verify
            with open(toml_path, "r") as f:
                data = toml.load(f)

            assert len(data["feeds"]["rss"]) == 2
            assert "https://example.com/feed1.rss" in data["feeds"]["rss"]
            assert "https://example.com/feed3.rss" in data["feeds"]["rss"]
            assert "https://example.com/feed2.rss" not in data["feeds"]["rss"]
            # Ensure other sections are preserved
            assert data["other"]["setting"] == "value"

        finally:
            os.unlink(toml_path)

    def test_create_invalid_feeds_toml(self):
        """Test creating invalid-feeds.toml file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "invalid-feeds.toml")

            results = {
                "https://example.com/bad1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="Network error",
                    feed_title="Bad Feed 1",
                ),
                "https://example.com/bad2.rss": FeedValidationResult(
                    is_valid=False, error_message="No audio content"
                ),
            }

            create_invalid_feeds_toml(
                Path(invalid_path),
                ["https://example.com/bad1.rss", "https://example.com/bad2.rss"],
                results,
            )

            # Verify file was created
            assert os.path.exists(invalid_path)

            # Check content
            with open(invalid_path, "r") as f:
                content = f.read()

            assert "Invalid RSS feeds moved from feeds.toml" in content
            assert "https://example.com/bad1.rss" in content
            assert "https://example.com/bad2.rss" in content
            assert "Network error" in content
            assert "No audio content" in content
            assert "Bad Feed 1" in content

    def test_append_to_existing_invalid_feeds_toml(self):
        """Test that new invalid feeds are appended to existing invalid-feeds.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid-feeds.toml"

            # First, create an initial invalid-feeds.toml
            initial_results = {
                "https://example.com/old1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="Old error 1",
                    feed_title="Old Feed 1",
                ),
            }

            create_invalid_feeds_toml(
                invalid_path,
                ["https://example.com/old1.rss"],
                initial_results,
            )

            # Verify initial file was created
            assert invalid_path.exists()
            with open(invalid_path, "r") as f:
                content = f.read()
            assert "https://example.com/old1.rss" in content
            assert "Old error 1" in content

            # Now add new invalid feeds
            new_results = {
                "https://example.com/new1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="New error 1",
                    feed_title="New Feed 1",
                ),
                "https://example.com/new2.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="New error 2",
                ),
            }

            create_invalid_feeds_toml(
                invalid_path,
                ["https://example.com/new1.rss", "https://example.com/new2.rss"],
                new_results,
            )

            # Verify all feeds are in the file
            with open(invalid_path, "r") as f:
                content = f.read()

            # Old feed should still be there
            assert "https://example.com/old1.rss" in content
            assert "Old error 1" in content
            assert "Old Feed 1" in content

            # New feeds should be added
            assert "https://example.com/new1.rss" in content
            assert "New error 1" in content
            assert "New Feed 1" in content
            assert "https://example.com/new2.rss" in content
            assert "New error 2" in content

            # Check timestamp was added
            assert "Last updated:" in content

    def test_update_existing_invalid_feed_error(self):
        """Test that updating an existing invalid feed updates its error message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid-feeds.toml"

            # Create initial invalid feed
            initial_results = {
                "https://example.com/feed1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="Network timeout",
                    feed_title="My Feed",
                ),
            }

            create_invalid_feeds_toml(
                invalid_path,
                ["https://example.com/feed1.rss"],
                initial_results,
            )

            # Now update with a different error
            updated_results = {
                "https://example.com/feed1.rss": FeedValidationResult(
                    is_valid=False,
                    error_message="No audio content found",
                    feed_title="My Feed Updated",
                ),
            }

            create_invalid_feeds_toml(
                invalid_path,
                ["https://example.com/feed1.rss"],
                updated_results,
            )

            # Verify the error was updated
            with open(invalid_path, "r") as f:
                content = f.read()

            assert "https://example.com/feed1.rss" in content
            assert "No audio content found" in content
            assert "My Feed Updated" in content
            # Old error should not be there
            assert "Network timeout" not in content
