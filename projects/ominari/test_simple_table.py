#!/usr/bin/env python3
"""Test simple table creation."""

print("""
=== DEBUGGING BLANK TABLE ===

Please open the browser console (F12) and check for:

1. Any JavaScript errors (red text)
2. Console messages showing:
   - "updateMatchesAndPositions called with: X markets and {...}"
   - "Rendering X matches to tbody with Y existing rows"
   - Any error messages

To manually test in console, run:

fetch('/api/dashboard/unified')
  .then(r => r.json())
  .then(data => {
    console.log('Markets:', data.markets.length);
    console.log('Positions:', data.positions);
    updateMatchesAndPositions(data.markets, data.positions);
  });

Also try the debug endpoint:
http://localhost:8888/debug

This will show if the API is working correctly.
""")