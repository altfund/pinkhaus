"""
Test helpers for speaker diarization tests.
"""

import os
import unittest


def has_pyannote_requirements():
    """Check if all requirements for real diarization are available."""
    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        return False, "HF_TOKEN environment variable not set"
    
    # Check for pyannote-audio
    try:
        from pyannote.audio import Pipeline
        return True, "All requirements available"
    except ImportError:
        return False, "pyannote-audio not installed"


def skipUnlessRealDiarization(test_func):
    """Decorator to skip tests unless real diarization is available."""
    available, reason = has_pyannote_requirements()
    if not available:
        return unittest.skip(f"Real diarization not available: {reason}")(test_func)
    return test_func


def get_diarization_mode():
    """Get whether to use mock or real diarization based on environment."""
    available, _ = has_pyannote_requirements()
    return not available  # use_mock = True if requirements not available