"""
Tests for the audio transcriber library.
"""

import json
import toml
import pytest
from pathlib import Path
from beige_book import AudioTranscriber


# Test fixture path - audio file is in resources directory
# From test file: parent -> tests, parent.parent -> beige-book
# Then ../../resources/audio/harvard.wav from beige-book
HARVARD_WAV = (
    Path(__file__).parent.parent / ".." / ".." / "resources" / "audio" / "harvard.wav"
)

# Expected values for harvard.wav
EXPECTED_HASH = "971b4163670445c415c6b0fb6813c38093409ecac2f6b4d429ae3574d24ad470"
EXPECTED_SEGMENTS = 6
EXPECTED_LANGUAGE = "en"
EXPECTED_TEXT_SNIPPETS = [
    "stale smell of old beer",
    "takes heat to bring out the odor",
    "cold dip restores health",
    "salt pickle tastes fine with ham",
    "zestful food is the hot cross bun",
]


class TestAudioTranscriber:
    """Test the AudioTranscriber class"""

    @pytest.fixture
    def transcriber(self):
        """Create a transcriber instance"""
        return AudioTranscriber(model_name="tiny")

    @pytest.fixture
    def result(self, transcriber):
        """Get transcription result for harvard.wav"""
        return transcriber.transcribe_file(str(HARVARD_WAV), verbose=False)

    def test_file_hash(self, transcriber):
        """Test that file hash calculation is correct"""
        calculated_hash = transcriber.calculate_file_hash(str(HARVARD_WAV))
        assert calculated_hash == EXPECTED_HASH

    def test_transcription_metadata(self, result):
        """Test that transcription metadata is correct"""
        assert result.filename == "harvard.wav"
        assert result.file_hash == EXPECTED_HASH
        assert result.language == EXPECTED_LANGUAGE
        assert len(result.segments) == EXPECTED_SEGMENTS

    def test_transcription_content(self, result):
        """Test that transcription contains expected text"""
        full_text_lower = result.full_text.lower()
        for snippet in EXPECTED_TEXT_SNIPPETS:
            assert snippet in full_text_lower, f"Expected '{snippet}' in transcription"

    def test_segment_structure(self, result):
        """Test that segments have proper structure"""
        for i, segment in enumerate(result.segments):
            # Check segment has required attributes (duck typing)
            assert hasattr(segment, "start_ms") or hasattr(segment, "start")
            assert hasattr(segment, "end_ms") or hasattr(segment, "end")
            assert hasattr(segment, "text")

            # Get start/end in seconds for compatibility
            start = (
                segment.start_ms / 1000.0
                if hasattr(segment, "start_ms")
                else segment.start
            )
            end = segment.end_ms / 1000.0 if hasattr(segment, "end_ms") else segment.end

            assert start >= 0
            assert end > start
            assert len(segment.text.strip()) > 0

            # Check segments are in order
            if i > 0:
                prev_end = (
                    result.segments[i - 1].end_ms / 1000.0
                    if hasattr(result.segments[i - 1], "end_ms")
                    else result.segments[i - 1].end
                )
                assert start >= prev_end

    def test_time_formatting(self):
        """Test time formatting function"""
        from beige_book.transcriber_betterproto import format_time

        assert format_time(0) == "00:00:00.000"
        assert format_time(61.5) == "00:01:01.500"
        assert format_time(3661.123) == "01:01:01.123"


class TestOutputFormats:
    """Test different output format conversions"""

    @pytest.fixture
    def result(self):
        """Get transcription result for harvard.wav"""
        transcriber = AudioTranscriber(model_name="tiny")
        return transcriber.transcribe_file(str(HARVARD_WAV), verbose=False)

    def test_json_format(self, result):
        """Test JSON output format"""
        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["filename"] == "harvard.wav"
        assert data["file_hash"] == EXPECTED_HASH
        assert data["language"] == EXPECTED_LANGUAGE
        assert len(data["segments"]) == EXPECTED_SEGMENTS
        assert "full_text" in data

    def test_toml_format(self, result):
        """Test TOML output format"""
        toml_str = result.to_toml()
        data = toml.loads(toml_str)

        assert data["transcription"]["filename"] == "harvard.wav"
        assert data["transcription"]["file_hash"] == EXPECTED_HASH
        assert data["transcription"]["language"] == EXPECTED_LANGUAGE
        assert len(data["segments"]) == EXPECTED_SEGMENTS

    def test_csv_format(self, result):
        """Test CSV output format"""
        csv_str = result.to_csv()
        lines = csv_str.strip().split("\n")

        # Check metadata comments
        assert lines[0] == "# File: harvard.wav"
        assert lines[1] == f"# SHA256: {EXPECTED_HASH}"
        assert lines[2] == f"# Language: {EXPECTED_LANGUAGE}"

        # Check header
        assert lines[3] == "Start,End,Duration,Text"

        # Check data rows
        data_lines = lines[4:]
        assert len(data_lines) == EXPECTED_SEGMENTS

    def test_table_format(self, result):
        """Test table output format"""
        table_str = result.to_table()

        # Check metadata header
        assert "File: harvard.wav" in table_str
        assert f"SHA256: {EXPECTED_HASH}" in table_str
        assert f"Language: {EXPECTED_LANGUAGE}" in table_str

        # Check table structure
        assert "Start" in table_str
        assert "End" in table_str
        assert "Duration" in table_str
        assert "Text" in table_str

    def test_text_format(self, result):
        """Test text output format"""
        text = result.format("text")
        assert text == result.full_text

        # Check for expected content
        text_lower = text.lower()
        for snippet in EXPECTED_TEXT_SNIPPETS:
            assert snippet in text_lower

    def test_format_method(self, result):
        """Test the generic format method"""
        assert result.format("text") == result.full_text
        assert result.format("json") == result.to_json()
        assert result.format("csv") == result.to_csv()
        assert result.format("table") == result.to_table()
        assert result.format("toml") == result.to_toml()

        with pytest.raises(ValueError):
            result.format("invalid_format")


class TestErrorHandling:
    """Test error handling"""

    def test_file_not_found(self):
        """Test handling of non-existent files"""
        transcriber = AudioTranscriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file("non_existent_file.wav")

    def test_invalid_model_name(self):
        """Test that invalid model names are handled by whisper"""
        # Note: whisper itself will handle invalid model names
        # This test ensures our code doesn't break before reaching whisper
        transcriber = AudioTranscriber(model_name="invalid_model")
        # This should raise an error when trying to load the model
        with pytest.raises(Exception):
            transcriber.transcribe_file(str(HARVARD_WAV))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
