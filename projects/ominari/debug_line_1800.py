#!/usr/bin/env python3
"""Debug line 1800 issue."""

with open('web_monitor.py', 'r') as f:
    lines = f.readlines()

# Check around line 1800 in the actual file
print("Lines around 1800 in web_monitor.py:")
for i in range(1795, 1810):
    if i < len(lines):
        line = lines[i].rstrip()
        print(f"{i+1}: {repr(line)}")

print("\n" + "="*50 + "\n")

# Find SINGLE_PAGE_DASHBOARD and count lines within it
content = ''.join(lines)
start = content.find('SINGLE_PAGE_DASHBOARD = """')
if start != -1:
    end = content.find('"""', start + 27)
    dashboard = content[start+27:end]
    
    # The error line 1800 is probably relative to the start of the HTML
    dashboard_lines = dashboard.split('\n')
    
    print(f"Lines around 1800 in the dashboard HTML (if it exists):")
    for i in range(1795, 1810):
        if i < len(dashboard_lines):
            line = dashboard_lines[i].rstrip()
            if "'" in line and not line.strip().endswith("'"):
                print(f"⚠️  {i+1}: {repr(line)}")
            else:
                print(f"{i+1}: {repr(line[:100])}")