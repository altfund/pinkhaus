#!/usr/bin/env python3
"""
Generate API documentation from OpenAPI specification.
Creates both Markdown and HTML documentation.
"""

import yaml
import json
from datetime import datetime
from typing import Dict, List, Any

def load_openapi_spec(filepath: str) -> Dict:
    """Load OpenAPI specification from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def generate_markdown_docs(spec: Dict) -> str:
    """Generate Markdown documentation from OpenAPI spec."""
    md = []
    
    # Title and description
    info = spec['info']
    md.append(f"# {info['title']}")
    md.append(f"\nVersion: {info['version']}")
    md.append(f"\n{info['description']}")
    
    # Table of contents
    md.append("\n## Table of Contents\n")
    for tag in spec.get('tags', []):
        md.append(f"- [{tag['name']}](#{tag['name'].lower().replace(' ', '-')})")
    
    # Servers
    md.append("\n## Servers\n")
    for server in spec.get('servers', []):
        md.append(f"- **{server.get('description', 'Server')}**: `{server['url']}`")
    
    # Group endpoints by tag
    endpoints_by_tag = {}
    for path, methods in spec['paths'].items():
        for method, details in methods.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                tags = details.get('tags', ['Other'])
                for tag in tags:
                    if tag not in endpoints_by_tag:
                        endpoints_by_tag[tag] = []
                    endpoints_by_tag[tag].append({
                        'path': path,
                        'method': method,
                        'details': details
                    })
    
    # Document each tag group
    for tag_info in spec.get('tags', []):
        tag = tag_info['name']
        md.append(f"\n## {tag}\n")
        md.append(f"{tag_info.get('description', '')}\n")
        
        if tag in endpoints_by_tag:
            for endpoint in endpoints_by_tag[tag]:
                md.append(f"### {endpoint['method'].upper()} {endpoint['path']}\n")
                details = endpoint['details']
                
                md.append(f"**{details.get('summary', 'No summary')}**\n")
                
                if 'description' in details:
                    md.append(f"{details['description']}\n")
                
                # Parameters
                if 'parameters' in details:
                    md.append("#### Parameters\n")
                    md.append("| Name | In | Type | Required | Description |")
                    md.append("|------|-----|------|----------|-------------|")
                    for param in details['parameters']:
                        required = param.get('required', False)
                        param_type = param.get('schema', {}).get('type', 'string')
                        md.append(f"| {param['name']} | {param['in']} | {param_type} | {'Yes' if required else 'No'} | {param.get('description', '')} |")
                    md.append("")
                
                # Request body
                if 'requestBody' in details:
                    md.append("#### Request Body\n")
                    content = details['requestBody'].get('content', {})
                    for content_type, schema_info in content.items():
                        md.append(f"Content-Type: `{content_type}`\n")
                        if 'schema' in schema_info:
                            md.append("```json")
                            md.append(json.dumps(schema_info['schema'], indent=2))
                            md.append("```\n")
                
                # Responses
                md.append("#### Responses\n")
                for status, response in details.get('responses', {}).items():
                    md.append(f"**{status}**: {response.get('description', '')}\n")
                    
                    if 'content' in response:
                        for content_type, schema_info in response['content'].items():
                            if 'schema' in schema_info and '$ref' in schema_info['schema']:
                                ref = schema_info['schema']['$ref'].split('/')[-1]
                                md.append(f"Returns: [`{ref}`](#{ref.lower()})\n")
                
                md.append("---\n")
    
    # Document schemas
    md.append("\n## Schemas\n")
    schemas = spec.get('components', {}).get('schemas', {})
    for schema_name, schema_def in schemas.items():
        md.append(f"### {schema_name}\n")
        
        if 'description' in schema_def:
            md.append(f"{schema_def['description']}\n")
        
        if 'properties' in schema_def:
            md.append("| Property | Type | Description |")
            md.append("|----------|------|-------------|")
            
            for prop_name, prop_def in schema_def['properties'].items():
                prop_type = prop_def.get('type', 'object')
                if '$ref' in prop_def:
                    prop_type = f"[{prop_def['$ref'].split('/')[-1]}](#{prop_def['$ref'].split('/')[-1].lower()})"
                elif prop_type == 'array' and 'items' in prop_def:
                    if '$ref' in prop_def['items']:
                        item_type = prop_def['items']['$ref'].split('/')[-1]
                        prop_type = f"array of [{item_type}](#{item_type.lower()})"
                    else:
                        prop_type = f"array of {prop_def['items'].get('type', 'object')}"
                
                desc = prop_def.get('description', '')
                if 'enum' in prop_def:
                    desc += f" (enum: {', '.join(prop_def['enum'])})"
                
                md.append(f"| {prop_name} | {prop_type} | {desc} |")
        
        md.append("")
    
    return '\n'.join(md)

def generate_html_docs(spec: Dict, markdown_content: str) -> str:
    """Generate HTML documentation with nice styling."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{spec['info']['title']} - API Documentation</title>
    <style>
        :root {{
            --primary-color: #00ff00;
            --secondary-color: #00ffff;
            --bg-color: #0a0a0a;
            --card-bg: #1a1a1a;
            --text-color: #e0e0e0;
            --border-color: #333;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1, h2, h3, h4 {{
            color: var(--primary-color);
            margin-bottom: 1rem;
        }}
        
        h1 {{
            font-size: 2.5em;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 10px;
        }}
        
        h2 {{
            font-size: 2em;
            margin-top: 2rem;
            color: var(--secondary-color);
        }}
        
        h3 {{
            font-size: 1.5em;
            margin-top: 1.5rem;
        }}
        
        h4 {{
            font-size: 1.2em;
            margin-top: 1rem;
        }}
        
        .endpoint {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .method {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.9em;
            margin-right: 10px;
        }}
        
        .method.get {{ background-color: #4CAF50; color: white; }}
        .method.post {{ background-color: #2196F3; color: white; }}
        .method.put {{ background-color: #FF9800; color: white; }}
        .method.delete {{ background-color: #f44336; color: white; }}
        .method.patch {{ background-color: #9C27B0; color: white; }}
        
        .path {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 1.1em;
            color: var(--secondary-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            border: 1px solid var(--border-color);
            padding: 10px;
            text-align: left;
        }}
        
        th {{
            background-color: var(--card-bg);
            color: var(--primary-color);
        }}
        
        tr:hover {{
            background-color: rgba(0, 255, 0, 0.05);
        }}
        
        code {{
            background-color: rgba(0, 255, 0, 0.1);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            color: var(--primary-color);
        }}
        
        pre {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
        }}
        
        pre code {{
            background-color: transparent;
            color: var(--text-color);
        }}
        
        .toc {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin: 10px 0;
        }}
        
        .toc a {{
            color: var(--secondary-color);
            text-decoration: none;
            transition: color 0.3s;
        }}
        
        .toc a:hover {{
            color: var(--primary-color);
        }}
        
        .server-list {{
            background-color: var(--card-bg);
            border-left: 4px solid var(--primary-color);
            padding: 15px;
            margin: 15px 0;
        }}
        
        .schema {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .back-to-top {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: var(--primary-color);
            color: var(--bg-color);
            padding: 10px 15px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            transition: background-color 0.3s;
        }}
        
        .back-to-top:hover {{
            background-color: var(--secondary-color);
        }}
        
        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2rem 0;
        }}
        
        .version {{
            color: var(--secondary-color);
            font-size: 1.2em;
        }}
        
        .description {{
            font-size: 1.1em;
            margin: 20px 0;
            line-height: 1.8;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 3rem;
            text-align: center;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        <div id="content"></div>
        <a href="#top" class="back-to-top">↑ Top</a>
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script>
        // Convert markdown to HTML
        const markdownContent = `{markdown_content.replace('`', '\\`')}`;
        document.getElementById('content').innerHTML = marked.parse(markdownContent);
        
        // Add classes to elements
        document.querySelectorAll('h3').forEach(h3 => {{
            const text = h3.textContent;
            if (text.match(/^(GET|POST|PUT|DELETE|PATCH)\\s/)) {{
                const [method, ...pathParts] = text.split(' ');
                h3.innerHTML = `<span class="method ${{method.toLowerCase()}}">${{method}}</span> <span class="path">${{pathParts.join(' ')}}</span>`;
                
                // Wrap in endpoint div
                const wrapper = document.createElement('div');
                wrapper.className = 'endpoint';
                h3.parentNode.insertBefore(wrapper, h3);
                
                let sibling = h3;
                while (sibling && sibling.tagName !== 'H3' && sibling.tagName !== 'H2') {{
                    const next = sibling.nextSibling;
                    wrapper.appendChild(sibling);
                    sibling = next;
                }}
            }}
        }});
        
        // Wrap schemas
        const schemaHeaders = Array.from(document.querySelectorAll('h2')).filter(h => h.textContent === 'Schemas');
        if (schemaHeaders.length > 0) {{
            let current = schemaHeaders[0].nextElementSibling;
            while (current && current.tagName !== 'H2') {{
                if (current.tagName === 'H3') {{
                    const wrapper = document.createElement('div');
                    wrapper.className = 'schema';
                    current.parentNode.insertBefore(wrapper, current);
                    
                    let sibling = current;
                    while (sibling && sibling.tagName !== 'H3' && sibling.tagName !== 'H2') {{
                        const next = sibling.nextSibling;
                        wrapper.appendChild(sibling);
                        sibling = next;
                    }}
                }} else {{
                    current = current.nextElementSibling;
                }}
            }}
        }}
        
        // Smooth scroll
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    return html

def main():
    """Generate API documentation."""
    print("Generating API documentation...")
    
    # Load OpenAPI spec
    spec = load_openapi_spec('openapi_spec.yaml')
    
    # Generate Markdown
    markdown_content = generate_markdown_docs(spec)
    
    # Save Markdown
    with open('API_DOCUMENTATION.md', 'w') as f:
        f.write(markdown_content)
    print("✓ Generated API_DOCUMENTATION.md")
    
    # Generate HTML
    html_content = generate_html_docs(spec, markdown_content)
    
    # Save HTML
    with open('api_documentation.html', 'w') as f:
        f.write(html_content)
    print("✓ Generated api_documentation.html")
    
    # Generate JSON version for tools
    with open('openapi_spec.json', 'w') as f:
        json.dump(spec, f, indent=2)
    print("✓ Generated openapi_spec.json")
    
    print("\nDocumentation generation complete!")
    print("\nView documentation:")
    print("  - Markdown: API_DOCUMENTATION.md")
    print("  - HTML: api_documentation.html")
    print("  - OpenAPI JSON: openapi_spec.json")

if __name__ == "__main__":
    main()