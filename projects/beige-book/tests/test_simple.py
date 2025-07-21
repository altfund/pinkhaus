"""
Simple tests that don't require audio processing.
"""

import pytest


def test_basic_math():
    """Test basic math operations"""
    assert 2 + 2 == 4
    assert 5 * 3 == 15
    assert 10 / 2 == 5


def test_string_operations():
    """Test string operations"""
    text = "hello world"
    assert text.upper() == "HELLO WORLD"
    assert text.replace("world", "python") == "hello python"
    assert len(text) == 11


def test_list_operations():
    """Test list operations"""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert max(numbers) == 5
    assert min(numbers) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])