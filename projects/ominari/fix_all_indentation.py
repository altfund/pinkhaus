#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix all indentation issues in ominari_unified.py
"""

# Read the file
with open('ominari_unified.py', 'r') as f:
    lines = f.readlines()

# Fix specific problematic sections
fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Fix the paper trading section
    if i >= 258 and i <= 289:
        # These lines need extra indentation
        if line.strip() and not line.startswith('                        '):
            # Add 12 more spaces (3 levels of indentation)
            line = '            ' + line.lstrip()
    
    fixed_lines.append(line)
    i += 1

# Write back
with open('ominari_unified.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed all indentation issues")