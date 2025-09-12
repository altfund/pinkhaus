#!/usr/bin/env python3
"""
Quick setup script for API authentication
"""

import os
import json
import subprocess
import sys

def setup_api_auth():
    """Set up API authentication for Ominari."""
    print("=== Ominari API Authentication Setup ===\n")
    
    # Check if api_keys.json exists
    if os.path.exists('api_keys.json'):
        print("✅ API keys file already exists")
        
        # Load and display keys
        with open('api_keys.json', 'r') as f:
            data = json.load(f)
            
        print(f"\nFound {len(data['keys'])} existing keys:")
        for key in data['keys']:
            status = "Active" if key['is_active'] else "Inactive"
            print(f"  - {key['name']}: {status} ({', '.join(key['permissions'])})")
    else:
        print("📝 No API keys found. Let's create some!\n")
        
        # Create default keys
        keys_to_create = [
            {
                'name': 'Development Key',
                'permissions': ['read', 'trade'],
                'rate_limit': 120,
                'description': 'For local development and testing'
            },
            {
                'name': 'Monitoring Key',
                'permissions': ['read'],
                'rate_limit': 60,
                'description': 'Read-only access for monitoring'
            },
            {
                'name': 'Admin Key',
                'permissions': ['admin'],
                'rate_limit': 300,
                'description': 'Full admin access'
            }
        ]
        
        created_keys = []
        
        for key_config in keys_to_create:
            print(f"\nCreating {key_config['name']}...")
            print(f"  Description: {key_config['description']}")
            print(f"  Permissions: {', '.join(key_config['permissions'])}")
            print(f"  Rate limit: {key_config['rate_limit']} req/min")
            
            # Run the command
            cmd = [
                'python', 'api_auth.py', 'create',
                '--name', key_config['name'],
                '--permissions'] + key_config['permissions'] + [
                '--rate-limit', str(key_config['rate_limit'])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extract the key from output
            for line in result.stdout.split('\n'):
                if line.startswith('Key: '):
                    api_key = line.split(': ')[1]
                    created_keys.append({
                        'name': key_config['name'],
                        'key': api_key,
                        'permissions': key_config['permissions']
                    })
                    print(f"  ✅ Created: {api_key}")
                    break
        
        if created_keys:
            print("\n=== API Keys Created Successfully ===")
            print("\n⚠️  IMPORTANT: Save these keys securely - they won't be shown again!\n")
            
            for key in created_keys:
                print(f"{key['name']}:")
                print(f"  Key: {key['key']}")
                print(f"  Permissions: {', '.join(key['permissions'])}")
                print()
    
    # Check environment variable
    print("\n=== Environment Setup ===")
    
    current_key = os.getenv('OMINARI_API_KEY')
    if current_key:
        print(f"✅ OMINARI_API_KEY is set: {current_key[:15]}...")
    else:
        print("❌ OMINARI_API_KEY not set in environment")
        print("\nTo set it, add to your ~/.bashrc or ~/.zshrc:")
        print('export OMINARI_API_KEY="omin_your-key-here"')
        print("\nOr set it temporarily:")
        print('export OMINARI_API_KEY="omin_your-key-here"')
    
    # Test the API
    print("\n=== Testing API Access ===")
    
    # First, check if web monitor is running
    import requests
    try:
        response = requests.get('http://localhost:8888/health', timeout=2)
        print("✅ Web monitor is running")
        
        # Test authenticated endpoint
        if current_key:
            headers = {'X-API-Key': current_key}
            response = requests.get('http://localhost:8888/api/status', 
                                  headers=headers, timeout=2)
            if response.status_code == 200:
                print("✅ API authentication working!")
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Active markets: {data.get('active_markets', 0)}")
            elif response.status_code == 401:
                print("❌ Authentication failed - check your API key")
            else:
                print(f"⚠️  Unexpected response: {response.status_code}")
        else:
            print("⏭️  Skipping auth test (no API key set)")
            
    except requests.exceptions.ConnectionError:
        print("❌ Web monitor not running")
        print("\nTo start it:")
        print("python web_monitor_auth.py")
        print("\nOr to use the legacy monitor:")
        print("python web_monitor.py")
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    # Show next steps
    print("\n=== Next Steps ===")
    print("\n1. Start the authenticated web monitor:")
    print("   python web_monitor_auth.py")
    print("\n2. Test with the demo script:")
    print("   python demo_authenticated_api.py")
    print("\n3. Use the API client in your code:")
    print("   from ominari_api_client import OminariAPIClient")
    print("   client = OminariAPIClient()  # Uses OMINARI_API_KEY env var")
    print("\n4. View API documentation:")
    print("   curl http://localhost:8888/api/docs")
    
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    setup_api_auth()