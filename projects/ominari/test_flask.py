#\!/usr/bin/env python3
"""Test if Flask is available"""
try:
    import flask
    print(f"Flask version: {flask.__version__}")
    print("Flask is available\!")
except ImportError as e:
    print(f"Flask not available: {e}")

try:
    import flask_socketio
    print("Flask-SocketIO is available\!")
except ImportError as e:
    print(f"Flask-SocketIO not available: {e}")
EOF < /dev/null
