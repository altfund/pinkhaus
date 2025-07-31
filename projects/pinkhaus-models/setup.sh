#!/usr/bin/env bash
# Setup script to install pinkhaus-models in development mode

set -e

echo "Setting up pinkhaus-models..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Error: No virtual environment detected"
    echo "Please activate your virtual environment first"
    exit 1
fi

# Install in development mode
echo "Installing pinkhaus-models in development mode..."
cd "$SCRIPT_DIR"
pip install -e .

echo "pinkhaus-models installed successfully!"
echo "You can now import it with: from pinkhaus_models import TranscriptionDatabase"