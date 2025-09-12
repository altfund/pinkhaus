#!/usr/bin/env python3
"""Debug why no data is showing."""

print("""
=== DEBUGGING NO DATA ISSUE ===

Please check the browser console (F12) for errors.

Try running this in the console to see what's happening:

console.log('Functions defined?');
console.log('updateMatchesAndPositions:', typeof updateMatchesAndPositions);
console.log('loadDashboardData:', typeof loadDashboardData);

// Check if data is being fetched
fetch('/api/dashboard/unified')
  .then(r => r.json())
  .then(data => {
    console.log('API data received:', data);
    console.log('Markets:', data.markets?.length);
    console.log('Positions:', data.positions);
    
    // Try to manually update
    if (typeof updateMatchesAndPositions === 'function') {
      console.log('Calling updateMatchesAndPositions...');
      updateMatchesAndPositions(data.markets || [], data.positions || {});
    } else {
      console.error('updateMatchesAndPositions is not defined!');
    }
  })
  .catch(e => console.error('Error:', e));

Also check for any errors when the page loads.
""")