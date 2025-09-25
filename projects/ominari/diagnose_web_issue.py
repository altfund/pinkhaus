#!/usr/bin/env python3
"""
Diagnose web dashboard connectivity issues
"""

import socket
import subprocess
import time
import os

def check_network():
    """Check basic network connectivity"""
    print("🔍 Diagnosing network connectivity...")
    
    # Check if localhost resolves
    try:
        socket.gethostbyname('localhost')
        print("✅ localhost resolves")
    except:
        print("❌ localhost DNS issue")
    
    # Check if 127.0.0.1 is accessible
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 80))
        sock.close()
        print("✅ 127.0.0.1 network accessible")
    except:
        print("❌ 127.0.0.1 network issue")
    
    # Check port 8888 availability
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 8888))
        sock.close()
        print("✅ Port 8888 is available")
        return True
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print("⚠️ Port 8888 already in use")
            return False
        else:
            print(f"❌ Port 8888 issue: {e}")
            return False

def kill_existing_processes():
    """Kill any existing Flask/web processes"""
    print("🧹 Cleaning up existing processes...")
    subprocess.run(['pkill', '-f', 'web_monitor'], capture_output=True)
    subprocess.run(['pkill', '-f', 'flask'], capture_output=True)
    subprocess.run(['pkill', '-f', '8888'], capture_output=True)
    time.sleep(2)

def start_minimal_server():
    """Start minimal HTTP server to test connectivity"""
    print("🚀 Starting minimal HTTP server on port 8888...")
    
    try:
        # Start simple Python HTTP server
        cmd = ['python3', '-m', 'http.server', '8888', '--bind', '127.0.0.1']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test connectivity
        try:
            import urllib.request
            response = urllib.request.urlopen('http://127.0.0.1:8888', timeout=5)
            content = response.read(100).decode()
            print("✅ HTTP server accessible!")
            print(f"Response preview: {content[:50]}...")
            
            # Kill the test server
            proc.terminate()
            return True
            
        except Exception as e:
            print(f"❌ HTTP server test failed: {e}")
            proc.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Failed to start HTTP server: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("=" * 60)
    print("🔧 Ominari Dashboard Connectivity Diagnostics")
    print("=" * 60)
    
    # Clean up first
    kill_existing_processes()
    
    # Check network basics
    if not check_network():
        print("\n❌ Network connectivity issues detected!")
        return False
    
    # Test with minimal server
    if not start_minimal_server():
        print("\n❌ Basic HTTP server test failed!")
        return False
    
    print("\n✅ Network connectivity is working!")
    print("The issue is likely with the Flask application itself.")
    print("\nNext steps:")
    print("1. Check web_monitor.py for Flask configuration issues")
    print("2. Ensure PostgreSQL connection is working")
    print("3. Try starting with simpler Flask config")
    
    return True

if __name__ == "__main__":
    main()