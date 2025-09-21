# Odds Data Fix Summary

## Problem Identified

The web_monitor.py was using incorrect query logic to fetch odds:
- ❌ Was joining on `position = 0, 1, 2` to identify home/draw/away
- ❌ The position field doesn't reliably map to outcomes
- ❌ This caused draw and away odds to always be NULL

## Solution Applied

Fixed the SQL query to use outcome values directly:

### Before (Incorrect):
```sql
LEFT JOIN odd o1 ON o1.source_id = m.source_id AND o1.outcome = 'home' AND o1.position = 0
LEFT JOIN odd o2 ON o2.source_id = m.source_id AND o2.outcome = 'draw' AND o2.position = 1
LEFT JOIN odd o3 ON o3.source_id = m.source_id AND o3.outcome = 'away' AND o3.position = 2
```

### After (Correct):
```sql
LEFT JOIN odd o ON o.source_id = m.source_id
...
MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) as home_odds,
MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) as draw_odds,
MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) as away_odds
```

## Files Updated

1. **web_monitor.py** - Fixed the market query (lines 95-131)

## Verification

Run these scripts to verify the fix:
```bash
# Check if draw/away odds are now available
python3 verify_odds_fix.py

# Test with safe database queries
python3 fix_odds_query.py
```

## Expected Results

After the fix:
- ✅ Draw odds should appear for Soccer matches
- ✅ Away odds should appear for all sports
- ✅ Paper trading will diversify across home/draw/away
- ✅ Portfolio exposure should stabilize below 100%

## Next Steps

1. **Restart web_monitor.py** to use the fixed query
2. **Monitor with:** `./simple_monitor.sh`
3. **Check dashboard** for draw/away bets being placed
4. **Run emergency fix** if needed: `python3 fix_overleveraging_postgres.py`

## Important Notes

- Basketball typically doesn't have draw odds (only home/away)
- Soccer should have all three: home/draw/away
- The fix addresses the query logic, not missing data
- If odds are still missing after the fix, it means they're not in the database