#!/usr/bin/env python3
"""Verify that generated proto files match the checked-in versions."""

import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_proto_files():
    """Verify that proto generation produces matching files."""
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "pinkhaus_models" / "proto"

    # Create a temp directory for generation
    with tempfile.TemporaryDirectory() as temp_dir:
        proto_gen_dir = Path(temp_dir) / "proto_gen"
        proto_gen_dir.mkdir(parents=True, exist_ok=True)

        # First, generate fresh proto files
        print("Generating fresh proto files...")
        generate_script = project_root / "scripts" / "generate_proto.py"
        result = subprocess.run(
            [sys.executable, str(generate_script), str(proto_gen_dir)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Failed to generate proto files:")
            print(result.stderr)
            return False

        # Now compare generated files with checked-in versions
        all_match = True
        differences = []

        # Find all generated Python files
        for gen_file in proto_gen_dir.rglob("*.py"):
            # Skip __pycache__ and other non-source files
            if "__pycache__" in str(gen_file):
                continue

            # Determine the expected location of the checked-in file
            relative_path = gen_file.relative_to(proto_gen_dir)
            checked_in_file = proto_dir / relative_path

            if not checked_in_file.exists():
                print(
                    f"✗ Generated file has no checked-in counterpart: {relative_path}"
                )
                all_match = False
                differences.append(f"Missing: {checked_in_file}")
                continue

            # Compare file contents
            if not filecmp.cmp(gen_file, checked_in_file, shallow=False):
                print(f"✗ Files differ: {relative_path}")
                all_match = False
                differences.append(f"Differs: {relative_path}")

                # Show diff for debugging
                diff_cmd = ["diff", "-u", str(checked_in_file), str(gen_file)]
                diff_result = subprocess.run(diff_cmd, capture_output=True, text=True)
                if diff_result.stdout:
                    print("Diff:")
                    print(diff_result.stdout[:1000])  # Limit output
                    if len(diff_result.stdout) > 1000:
                        print("... (truncated)")
            else:
                print(f"✓ {relative_path}")

        # Check for checked-in files that weren't generated
        for checked_file in proto_dir.rglob("*.py"):
            if "__pycache__" in str(checked_file):
                continue

            # Skip __init__.py files which we don't generate
            if checked_file.name == "__init__.py":
                continue

            relative_path = checked_file.relative_to(proto_dir)
            gen_file = proto_gen_dir / relative_path

            if not gen_file.exists():
                print(f"✗ Checked-in file was not generated: {relative_path}")
                all_match = False
                differences.append(f"Not generated: {relative_path}")

        if all_match:
            print("\n✓ All proto files match!")
            return True
        else:
            print(f"\n✗ Found {len(differences)} differences:")
            for diff in differences:
                print(f"  - {diff}")
            print("\nRun 'just proto-generate' to regenerate proto files.")
            return False


if __name__ == "__main__":
    success = verify_proto_files()
    sys.exit(0 if success else 1)
