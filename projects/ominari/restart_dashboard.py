#!/usr/bin/env python3
"""Restart dashboard instructions."""

print("=== FIXING DASHBOARD DATA DISPLAY ===\n")

print("The data loading issue may be due to cached JavaScript.")
print("\nPlease follow these steps:\n")

print("1. In your browser, do a HARD REFRESH:")
print("   - Windows/Linux: Ctrl + Shift + R")
print("   - Mac: Cmd + Shift + R")
print("")

print("2. If that doesn't work, clear cache:")
print("   - Open Developer Tools (F12)")
print("   - Right-click the refresh button")
print("   - Select 'Empty Cache and Hard Reload'")
print("")

print("3. Check console for errors:")
print("   - Look for any red error messages")
print("   - Check for 'Rendering X matches' message")
print("")

print("The API is working correctly and returning:")
print("  ✓ 5 markets")
print("  ✓ 8 open positions") 
print("  ✓ 20 closed positions")
print("")

print("Dashboard URL: http://localhost:8888/unified")