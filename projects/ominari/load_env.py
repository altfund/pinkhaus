#!/usr/bin/env python3
"""
Load environment variables from .env file
"""
import os
from pathlib import Path

def load_dotenv():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        
if __name__ == "__main__":
    load_dotenv()
    print("Environment variables loaded from .env")
    # Show Discord status
    webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if webhook:
        print(f"✅ Discord webhook configured: {webhook[:50]}...")
    else:
        print("❌ No Discord webhook configured")