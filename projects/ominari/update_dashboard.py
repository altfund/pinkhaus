#!/usr/bin/env python3
"""Script to update the dashboard HTML template with enhanced features"""

# Read the current file
with open('web_monitor_blockchain.py', 'r') as f:
    content = f.read()

# Find the start and end of the HTML template
template_start = content.find('BLOCKCHAIN_DASHBOARD_HTML = """') + len('BLOCKCHAIN_DASHBOARD_HTML = """')
template_end = content.rfind('"""')

# Get the current template
current_template = content[template_start:template_end]

# Add new CSS styles after the existing styles
new_styles = """
        .capital-flow-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 15px;
        }
        
        .capital-bar {
            display: flex;
            height: 40px;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.5);
        }
        
        .capital-segment {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: #000;
            transition: all 0.3s;
            position: relative;
        }
        
        .capital-available { background: #00ff00; }
        .capital-pending { background: #ffaa00; }
        .capital-in-play { background: #0088ff; }
        .capital-settlement { background: #ff88ff; }
        
        .capital-legend {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 12px;
            gap: 10px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }
        
        .chunk-timeline {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            max-height: 150px;
            overflow-y: auto;
        }
        
        .chunk-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            margin-bottom: 5px;
            background: #222;
            border-radius: 3px;
            border-left: 3px solid #00ff00;
        }
        
        .chunk-info {
            font-size: 12px;
        }
        
        .dynamic-indicator {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .dynamic-enabled {
            background: #00ff00;
            color: #000;
        }
        
        .dynamic-disabled {
            background: #666;
            color: #fff;
        }
        
        .empirical-data-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-size: 12px;
        }
        
        .empirical-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        
        .empirical-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .empirical-value {
            color: #00ff00;
            font-weight: bold;
        }"""

# Insert new styles before the closing </style> tag
style_end_pos = current_template.find('    </style>')
if style_end_pos > 0:
    updated_template = current_template[:style_end_pos] + new_styles + '\n' + current_template[style_end_pos:]
else:
    updated_template = current_template

# Save the updated file
new_content = content[:template_start] + updated_template + content[template_end:]

with open('web_monitor_blockchain_updated.py', 'w') as f:
    f.write(new_content)

print("Dashboard template updated successfully!")
print("Created: web_monitor_blockchain_updated.py")