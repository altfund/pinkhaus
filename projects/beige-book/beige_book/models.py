"""
Request and response models - now using betterproto for better performance.
This module re-exports the betterproto implementation for backward compatibility.
"""

# Re-export everything from the betterproto implementation
from .models_betterproto import *

# For complete backward compatibility, also provide string literals
from typing import Literal

# These are provided for type hints in existing code
InputTypeLiteral = Literal["file", "feed"]
ModelLiteral = Literal["tiny", "base", "small", "medium", "large"]
FormatLiteral = Literal["text", "json", "table", "csv", "toml", "sqlite"]
OrderLiteral = Literal["newest", "oldest"]