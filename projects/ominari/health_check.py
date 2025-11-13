#!/usr/bin/env python3
"""
Health check endpoint for Ominari DApp
"""

from flask import Flask, jsonify
import os
import psycopg2
import requests
from datetime import datetime
import socket

app = Flask(__name__)

def check_database():
    """Check PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('PG_HOST', 'localhost'),
            port=os.environ.get('PG_PORT', '5999'),
            user=os.environ.get('PG_USER', 'ominari_user'),
            password=os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            database=os.environ.get('PG_DB', 'ominari_production')
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market WHERE sport LIKE '%Soccer%'")
        count = cursor.fetchone()[0]
        conn.close()
        return True, f"{count} markets"
    except Exception as e:
        return False, str(e)

def check_dashboard():
    """Check if dashboard is running"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8888))
        sock.close()
        return result == 0, "Port 8888"
    except:
        return False, "Error"

def check_blockchain():
    """Check if local blockchain is running"""
    try:
        response = requests.post(
            'http://localhost:8545',
            json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1},
            timeout=1
        )
        if response.status_code == 200:
            block = int(response.json()['result'], 16)
            return True, f"Block #{block}"
    except:
        pass
    return False, "Not running"

@app.route('/health')
def health():
    """Main health check endpoint"""
    
    # Check all services
    db_ok, db_msg = check_database()
    dash_ok, dash_msg = check_dashboard()
    blockchain_ok, blockchain_msg = check_blockchain()
    
    # Overall status
    all_ok = db_ok and dash_ok
    
    health_data = {
        'status': 'healthy' if all_ok else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'database': {
                'status': 'up' if db_ok else 'down',
                'message': db_msg
            },
            'dashboard': {
                'status': 'up' if dash_ok else 'down',
                'message': dash_msg
            },
            'blockchain': {
                'status': 'up' if blockchain_ok else 'down',
                'message': blockchain_msg
            }
        },
        'version': '1.0.0',
        'mode': 'hybrid' if not blockchain_ok else 'full-dapp'
    }
    
    return jsonify(health_data), 200 if all_ok else 503

@app.route('/ready')
def ready():
    """Readiness check"""
    db_ok, _ = check_database()
    return jsonify({'ready': db_ok}), 200 if db_ok else 503

@app.route('/live')
def live():
    """Liveness check"""
    return jsonify({'alive': True}), 200

if __name__ == '__main__':
    # Set up environment
    os.environ['USE_POSTGRESQL'] = '1'
    app.run(host='0.0.0.0', port=8889, debug=False)