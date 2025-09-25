#!/usr/bin/env python3
"""Test if chunks are being created and displayed"""
import requests
import json
from datetime import datetime

def test_chunks():
    """Test chunk creation by calling the API"""
    try:
        # Make a request to trigger data fetch
        print("🔍 Testing chunk creation...")
        
        # First, let's check if server is running
        try:
            response = requests.get("http://localhost:8888", timeout=5)
            print("✅ Web server is running")
        except:
            print("❌ Web server not responding at http://localhost:8888")
            print("   Please ensure web_monitor.py is running")
            return
        
        # Check the debug endpoint
        try:
            response = requests.get("http://localhost:8888/debug", timeout=5)
            if response.status_code == 200:
                print("✅ Debug endpoint accessible")
                
                # Parse response for chunk info
                html = response.text
                if "Markets:" in html:
                    markets_line = html.split("Markets:")[1].split("</h2>")[0].strip()
                    print(f"📊 Markets found: {markets_line}")
                
                if "Signals:" in html:
                    signals_line = html.split("Signals:")[1].split("</h2>")[0].strip()
                    print(f"📡 Signals found: {signals_line}")
        except Exception as e:
            print(f"⚠️  Debug endpoint error: {str(e)}")
        
        # Check recent log for chunk creation
        print("\n📋 Checking logs for chunk activity...")
        try:
            with open('web_monitor_fixed.log', 'r') as f:
                lines = f.readlines()[-200:]  # Last 200 lines
                
            # Look for chunk creation
            chunk_lines = [l.strip() for l in lines if "Created" in l and "chunks" in l]
            if chunk_lines:
                print(f"✅ Found chunk creation:")
                for line in chunk_lines[-3:]:  # Show last 3
                    print(f"   {line}")
            else:
                print("❌ No chunk creation found in recent logs")
                print("   This might mean:")
                print("   - The app needs to be restarted to pick up changes")
                print("   - No markets with proper time data")
                
            # Look for chunk selection
            selected_lines = [l.strip() for l in lines if "Selected first chunk" in l]
            if selected_lines:
                print(f"\n✅ Found chunk selection:")
                print(f"   {selected_lines[-1]}")
            
            # Look for odds distribution
            odds_lines = [l.strip() for l in lines if "Odds distribution" in l]
            if odds_lines:
                print(f"\n📊 Odds distribution:")
                print(f"   {odds_lines[-1]}")
                
                # Check for draw/away odds issue
                if "Draw: 0, Away: 0" in odds_lines[-1]:
                    print("   ⚠️  WARNING: Only HOME odds available (Draw/Away are 0)")
                    
        except Exception as e:
            print(f"❌ Error reading log: {str(e)}")
            
        print("\n" + "="*60)
        print("📌 NEXT STEPS:")
        print("1. If chunks aren't showing, restart web_monitor.py")
        print("2. Check dashboard at http://localhost:8888")
        print("3. Look for 'Market Time Windows' section above the table")
        print("4. Run ./watch_live_colored.py for colored monitoring")
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")

if __name__ == "__main__":
    test_chunks()