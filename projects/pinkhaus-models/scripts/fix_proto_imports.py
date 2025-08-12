#!/usr/bin/env python3
"""Fix imports in generated proto files to use relative imports."""

import re
from pathlib import Path


def fix_proto_imports(proto_dir):
    """Fix imports in generated proto files to use relative imports."""
    fixed_count = 0

    # Find all generated *_pb2_grpc.py files
    for grpc_file in Path(proto_dir).rglob("*_pb2_grpc.py"):
        print(f"Processing {grpc_file.relative_to(proto_dir)}...")

        # Read the file
        content = grpc_file.read_text()
        original_content = content

        # Find and replace absolute imports with relative imports
        # Pattern: import <name>_pb2 as <name>__pb2
        pattern = r"^import (\w+_pb2) as (\w+__pb2)$"
        replacement = r"from . import \1 as \2"

        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Write back if changed
        if content != original_content:
            grpc_file.write_text(content)
            print(f"  ✓ Fixed imports in {grpc_file.name}")
            fixed_count += 1
        else:
            print(f"  - No changes needed in {grpc_file.name}")

    print(f"\n✓ Fixed imports in {fixed_count} files")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        proto_dir = Path(sys.argv[1])
    else:
        # Default to the standard proto directory
        project_root = Path(__file__).parent.parent
        proto_dir = project_root / "pinkhaus_models" / "proto"

    if not proto_dir.exists():
        print(f"Error: Directory not found: {proto_dir}")
        sys.exit(1)

    success = fix_proto_imports(proto_dir)
    sys.exit(0 if success else 1)
