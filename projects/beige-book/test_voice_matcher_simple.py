#!/usr/bin/env python3
"""
Simple test of voice matching functionality.
"""

import os
import tempfile
from pathlib import Path

# Check for HF token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("Error: HF_TOKEN environment variable is required")
    exit(1)

# Create temporary database
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
    db_path = tmp.name

print(f"Database: {db_path}")

# Path to test audio
harvard_path = Path("/Users/price/development/ai-projects/pinkhaus2/resources/audio/harvard.wav")
if not harvard_path.exists():
    print(f"Test audio not found: {harvard_path}")
    exit(1)

print("\n=== Step 1: Process audio file with speaker profiling ===")
import subprocess

# Process with speaker profiling
cmd = [
    "beige-book", str(harvard_path),
    "--db-path", db_path,
    "--format", "sqlite",
    "--model", "tiny",
    "--diarize",
    "--speaker-profiles",
    "--embedding-method", "speechbrain"  # Use real embeddings
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

print("Success! Audio processed with speaker profiling.")

print("\n=== Step 2: Test voice matching ===")

# Now use voice matcher
cmd2 = [
    "beige-book-match-voice",
    str(harvard_path),
    db_path,
    "--embedding-method", "speechbrain",
    "--threshold", "0.85"
]

print(f"Running: {' '.join(cmd2)}")
result2 = subprocess.run(cmd2, capture_output=True, text=True)

if result2.returncode != 0:
    print(f"Error: {result2.stderr}")
    exit(1)

print("\nOutput:")
print(result2.stdout)

print(f"\n\nTest complete. Database at: {db_path}")
print("You can now test with your own audio:")
print(f"  beige-book-match-voice /path/to/your/audio.wav {db_path}")