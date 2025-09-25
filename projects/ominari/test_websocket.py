#!/usr/bin/env python3
"""Test the WebSocket connection"""

import socketio
import time

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print("✅ Connected to WebSocket")
    # Request dashboard data
    sio.emit('request_dashboard_data')

@sio.event
def dashboard_update(data):
    print("✅ Received dashboard update!")
    if 'markets' in data:
        print(f"   - Markets: {len(data['markets'])}")
    if 'session' in data:
        print(f"   - Session: {data['session'].get('session_id', 'N/A')}")
    if 'portfolio' in data:
        print(f"   - Portfolio value: ${data['portfolio'].get('portfolio_value', 0):.2f}")
    if 'stats' in data:
        print(f"   - Chunks: {data['stats'].get('total_chunks', 0)}")
    
    # Disconnect after receiving data
    sio.disconnect()

@sio.event
def activity(data):
    print(f"✅ Activity: {data.get('message', 'N/A')}")

@sio.event
def disconnect():
    print("📡 Disconnected from WebSocket")

try:
    print("Connecting to WebSocket at http://localhost:8888...")
    sio.connect('http://localhost:8888')
    # Wait a bit for data
    time.sleep(2)
except Exception as e:
    print(f"❌ WebSocket error: {e}")
finally:
    if sio.connected:
        sio.disconnect()