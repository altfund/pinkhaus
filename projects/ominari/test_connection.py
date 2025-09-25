#!/usr/bin/env python3
"""Test connection to running dashboard"""

import urllib.request
import socket

def test_connection():
    try:
        print("Testing connection to localhost:8888...")
        response = urllib.request.urlopen('http://localhost:8888', timeout=10)
        content = response.read().decode()
        print(f"✅ Connection successful!")
        print(f"Status: {response.status}")
        print(f"Content length: {len(content)}")
        print(f"First 200 chars: {content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Try with 127.0.0.1
        try:
            response = urllib.request.urlopen('http://127.0.0.1:8888', timeout=10)
            content = response.read().decode()
            print(f"✅ Connection to 127.0.0.1:8888 successful!")
            print(f"Content length: {len(content)}")
            return True
        except Exception as e2:
            print(f"❌ Connection to 127.0.0.1:8888 also failed: {e2}")
            return False

if __name__ == "__main__":
    test_connection()