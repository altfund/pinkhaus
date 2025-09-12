#!/usr/bin/env python3
"""Verify that all template literals have been replaced."""

import re

# Read the web_monitor.py file
with open('web_monitor.py', 'r') as f:
    content = f.read()

# Find the SINGLE_PAGE_DASHBOARD template
start = content.find('SINGLE_PAGE_DASHBOARD = """')
end = content.find('"""', start + len('SINGLE_PAGE_DASHBOARD = """'))
dashboard_template = content[start:end]

# Search for remaining template literals
template_literals = re.findall(r'`[^`]+`', dashboard_template)

if template_literals:
    print("❌ FOUND REMAINING TEMPLATE LITERALS:")
    print(f"Total: {len(template_literals)}")
    print("\nFirst few examples:")
    for i, tl in enumerate(template_literals[:5]):
        print(f"{i+1}. {tl[:80]}{'...' if len(tl) > 80 else ''}")
else:
    print("✅ NO TEMPLATE LITERALS FOUND!")
    print("\nThe dashboard should now work properly.")
    print("\nTo test:")
    print("1. Start the web monitor: python web_monitor.py")
    print("2. Visit: http://localhost:8888/unified")
    print("3. Check browser console for any errors")

# Also check for multi-line strings that might still have issues
multiline_issues = re.findall(r'innerHTML\s*=\s*["\'][^"\']*\n', dashboard_template)
if multiline_issues:
    print(f"\n⚠️  Found {len(multiline_issues)} potential multi-line string issues")