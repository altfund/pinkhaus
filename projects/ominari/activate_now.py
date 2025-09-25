#!/usr/bin/env python3
"""
Activate all fixes immediately
"""
import os
import subprocess
import time
import signal
import sys

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success")
            if result.stdout:
                print(result.stdout[:500])  # Show first 500 chars
        else:
            print(f"❌ Failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    return True

def main():
    print("🚀 ACTIVATING ALL OMINARI FIXES")
    print("=" * 50)
    
    # 1. Kill existing web_monitor
    print("\n1️⃣ Stopping any existing web_monitor...")
    subprocess.run("pkill -f web_monitor.py", shell=True)
    time.sleep(2)
    
    # 2. Test PostgreSQL
    print("\n2️⃣ Testing PostgreSQL connection...")
    test_cmd = """python3 -c "
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})
from paper_trading_postgres_integrated import PaperTradingSessionManager
m = PaperTradingSessionManager()
print('PostgreSQL connection successful!')
" """
    
    if not run_command(test_cmd, "Testing PostgreSQL"):
        print("\n❌ Cannot connect to PostgreSQL. Please ensure it's running on port 5999")
        return
    
    # 3. Create fresh session
    print("\n3️⃣ Creating fresh paper trading session...")
    create_session_cmd = """python3 -c "
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})
from paper_trading_postgres_integrated import PaperTradingSessionManager
m = PaperTradingSessionManager()
# Close any active sessions
import psycopg2
conn = psycopg2.connect(host='localhost', port=5999, database='ominari_production', user='ominari_user', password='ominari_2025_secure')
cur = conn.cursor()
cur.execute('UPDATE paper_trading_sessions SET status = \\'closed\\' WHERE status = \\'active\\' OR status = \\'ACTIVE\\'')
conn.commit()
# Create new session
session_id = m.create_session(10000, 'Fresh Start - Fixed Odds')
print(f'Created session: {session_id}')
" """
    
    run_command(create_session_cmd, "Creating fresh session")
    
    # 4. Start web_monitor
    print("\n4️⃣ Starting web_monitor.py...")
    web_proc = subprocess.Popen(
        ["python3", "web_monitor.py"],
        stdout=open("web_monitor_activated.log", "w"),
        stderr=subprocess.STDOUT
    )
    
    print(f"✅ Started with PID: {web_proc.pid}")
    
    # 5. Wait and check
    print("\n5️⃣ Waiting for startup...")
    time.sleep(5)
    
    if web_proc.poll() is None:
        print("✅ Web monitor is running!")
    else:
        print("❌ Web monitor crashed! Check web_monitor_activated.log")
        return
    
    # 6. Show status
    print("\n" + "=" * 50)
    print("✨ ALL SYSTEMS ACTIVATED!")
    print("\n📊 Dashboard: http://localhost:8888")
    print("📝 Logs: tail -f web_monitor_activated.log")
    print("🔍 Monitor: ./simple_monitor.sh")
    print("\n💡 Commands:")
    print("   Stop: pkill -f web_monitor.py")
    print("   Logs: tail -f web_monitor_activated.log")
    
    # 7. Offer to start monitor
    print("\n" + "=" * 50)
    response = input("Start live monitor? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting monitor...")
        subprocess.run("./simple_monitor.sh", shell=True)
    else:
        print("\nYou can start the monitor later with: ./simple_monitor.sh")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Activation cancelled")
        sys.exit(1)