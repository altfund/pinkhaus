#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix indentation issues in ominari_unified.py
"""


# Read the file
with open('ominari_unified.py', 'r') as f:
    lines = f.readlines()

# Fix lines where indentation is wrong after "with db_manager.get_db_session() as db:"
fixed_lines = []
in_db_context = False
expected_indent = 0

for i, line in enumerate(lines):
    # Check if we're starting a db context
    if 'with db_manager.get_db_session() as db:' in line:
        in_db_context = True
        # Get the indentation level
        expected_indent = len(line) - len(line.lstrip()) + 4
        fixed_lines.append(line)
        continue
    
    # If we're in a db context and the line has less indentation than expected
    if in_db_context:
        current_indent = len(line) - len(line.lstrip())
        
        # Check if we're exiting the context (empty line or less indentation)
        if line.strip() == '' or (current_indent < expected_indent - 4 and line.strip() != ''):
            in_db_context = False
            fixed_lines.append(line)
        else:
            # Fix indentation if needed
            if current_indent < expected_indent and line.strip() != '':
                # Add proper indentation
                fixed_line = ' ' * expected_indent + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open('ominari_unified.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed indentation in ominari_unified.py")