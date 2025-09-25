#!/usr/bin/env python3
"""Check Ominari System Status"""
import requests
import psycopg2
from datetime import datetime

def check_web_server():
    try:
        response = requests.get("http://localhost:8888", timeout=2)
        return response.status_code == 200, "Web server is running"
    except:
        return False, "Web server is not responding"

def check_database():
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ominari_trading",
            user="ominari_user", password="ominari_password"
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM market WHERE is_finished = false")
        active_markets = cur.fetchone()[0]
        conn.close()
        return True, f"Database OK - {active_markets} active markets"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def check_chunks_in_log():
    try:
        with open('web_monitor_fixed.log', 'r') as f:
            lines = f.readlines()[-100:]
        chunk_lines = [l for l in lines if "Created" in l and "chunks" in l]
        if chunk_lines:
            return True, "Chunks system active"
        return False, "No chunk creation in recent logs"
    except:
        return False, "Cannot read log file"

def main():
    print("=" * 60)
    print("OMINARI SYSTEM STATUS CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    checks = [
        ("Web Server", check_web_server()),
        ("Database", check_database()),
        ("Chunks", check_chunks_in_log())
    ]
    
    for name, (status, msg) in checks:
        icon = "OK" if status else "FAIL"
        print(f"[{icon}] {name}: {msg}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()