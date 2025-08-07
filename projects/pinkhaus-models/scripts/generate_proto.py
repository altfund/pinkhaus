#!/usr/bin/env python3
"""Generate Python code from proto files using betterproto."""
import os
import subprocess
import sys
from pathlib import Path


def generate_proto_files(output_base_dir=None):
    """Generate Python files from all proto files."""
    # Get the project root
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "pinkhaus_models" / "proto"
    
    # Default output directory
    if output_base_dir is None:
        output_base_dir = project_root / "pinkhaus_models" / "proto_gen"
    
    if not proto_dir.exists():
        print(f"Proto directory not found: {proto_dir}")
        return False
    
    # Find all .proto files
    proto_files = list(proto_dir.rglob("*.proto"))
    if not proto_files:
        print(f"No proto files found in {proto_dir}")
        return True
    
    print(f"Found {len(proto_files)} proto files")
    
    # Process each subdirectory separately to maintain structure
    for subdir in proto_dir.iterdir():
        if subdir.is_dir() and list(subdir.glob("*.proto")):
            print(f"\nProcessing {subdir.name}...")
            
            # Output directory for generated files
            output_dir = Path(output_base_dir) / subdir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate using betterproto
            cmd = [
                "python", "-m", "grpc_tools.protoc",
                f"--proto_path={subdir}",
                f"--python_betterproto_out={output_dir}",
            ]
            
            # Add all proto files in this subdirectory
            proto_files_in_subdir = list(subdir.glob("*.proto"))
            cmd.extend(str(f) for f in proto_files_in_subdir)
            
            print(f"Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✓ Generated Python code for {subdir.name}")
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to generate Python code for {subdir.name}")
                print(f"Error: {e.stderr}")
                
                # Try standard protoc as fallback
                print("\nTrying standard protoc generation...")
                cmd_standard = [
                    "python", "-m", "grpc_tools.protoc",
                    f"--proto_path={subdir}",
                    f"--python_out={output_dir}",
                    f"--grpc_python_out={output_dir}",
                ]
                cmd_standard.extend(str(f) for f in proto_files_in_subdir)
                
                try:
                    result = subprocess.run(cmd_standard, check=True, capture_output=True, text=True)
                    print(f"✓ Generated Python code using standard protoc for {subdir.name}")
                except subprocess.CalledProcessError as e2:
                    print(f"✗ Standard protoc also failed: {e2.stderr}")
                    return False
    
    print("\n✓ All proto files processed successfully")
    return True


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    success = generate_proto_files(output_dir)
    sys.exit(0 if success else 1)