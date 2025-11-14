#!/usr/bin/env python3
"""Simple dashboard starter"""
from web_dashboard_real_odds import app, socketio

if __name__ == '__main__':
    print("Starting dashboard on http://localhost:8888")
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)