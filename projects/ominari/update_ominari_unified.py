#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update ominari_unified.py to use database_v2 patterns.
"""

import re

# Read the file
with open('ominari_unified.py', 'r') as f:
    content = f.read()

# Pattern to find db = SessionLocal() with try/finally blocks
pattern = r'(\s*)db = SessionLocal\(\)\s*\n\s*try:(.*?)\n\s*finally:\s*\n\s*db\.close\(\)'

def replace_with_context_manager(match):
    indent = match.group(1)
    body = match.group(2)
    
    # The body needs to be dedented by one level (4 spaces)
    lines = body.split('\n')
    dedented_lines = []
    for line in lines:
        if line.startswith('    '):
            dedented_lines.append(line[4:])
        else:
            dedented_lines.append(line)
    dedented_body = '\n'.join(dedented_lines)
    
    return f"{indent}with db_manager.get_db_session() as db:{dedented_body}"

# Replace all occurrences
content = re.sub(pattern, replace_with_context_manager, content, flags=re.DOTALL)

# Write back
with open('ominari_unified.py', 'w') as f:
    f.write(content)

print("Updated ominari_unified.py to use database_v2 patterns")