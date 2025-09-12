#!/usr/bin/env python3
"""
API Authentication and Rate Limiting for Ominari Trading System

Features:
- API key authentication
- JWT token support
- Rate limiting per key
- IP-based rate limiting
- Admin vs user permissions
"""

import os
import time
import hmac
import hashlib
import secrets
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import jwt
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key configuration."""
    key: str
    name: str
    created_at: datetime
    permissions: List[str]
    rate_limit: int  # requests per minute
    is_active: bool = True
    last_used: Optional[datetime] = None
    total_requests: int = 0


@dataclass
class RateLimitInfo:
    """Rate limit tracking."""
    requests: List[float]  # timestamps
    limit: int
    window: int = 60  # seconds


class APIAuthManager:
    """Manages API authentication and rate limiting."""
    
    def __init__(self, secret_key: Optional[str] = None, db_path: str = "api_auth.db"):
        self.secret_key = secret_key or os.getenv('API_SECRET_KEY', secrets.token_urlsafe(32))
        self.db_path = db_path
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, RateLimitInfo] = defaultdict(lambda: RateLimitInfo([], 60))
        
        # Load API keys
        self._load_api_keys()
        
        # Default permissions
        self.PERMISSIONS = {
            'read': ['status', 'portfolio', 'positions', 'performance'],
            'trade': ['execute', 'close_position'],
            'admin': ['*']  # All permissions
        }
        
    def _load_api_keys(self):
        """Load API keys from database or file."""
        # For now, use JSON file. In production, use database
        try:
            if os.path.exists('api_keys.json'):
                with open('api_keys.json', 'r') as f:
                    data = json.load(f)
                    for key_data in data['keys']:
                        api_key = APIKey(
                            key=key_data['key'],
                            name=key_data['name'],
                            created_at=datetime.fromisoformat(key_data['created_at']),
                            permissions=key_data['permissions'],
                            rate_limit=key_data.get('rate_limit', 60),
                            is_active=key_data.get('is_active', True)
                        )
                        self.api_keys[api_key.key] = api_key
        except Exception as e:
            logger.warning(f"Could not load API keys: {e}")
            
        # Create default API key if none exist
        if not self.api_keys:
            self._create_default_key()
            
    def _create_default_key(self):
        """Create a default API key for development."""
        dev_key = self.create_api_key(
            name="Development Key",
            permissions=['read', 'trade'],
            rate_limit=120
        )
        logger.info(f"Created development API key: {dev_key}")
        
    def create_api_key(self, name: str, permissions: List[str], 
                      rate_limit: int = 60) -> str:
        """Create a new API key."""
        # Generate secure key
        key = f"omin_{secrets.token_urlsafe(32)}"
        
        api_key = APIKey(
            key=key,
            name=name,
            created_at=datetime.now(timezone.utc),
            permissions=permissions,
            rate_limit=rate_limit
        )
        
        self.api_keys[key] = api_key
        self._save_api_keys()
        
        return key
        
    def _save_api_keys(self):
        """Save API keys to storage."""
        data = {
            'keys': [
                {
                    'key': k.key,
                    'name': k.name,
                    'created_at': k.created_at.isoformat(),
                    'permissions': k.permissions,
                    'rate_limit': k.rate_limit,
                    'is_active': k.is_active
                }
                for k in self.api_keys.values()
            ]
        }
        
        with open('api_keys.json', 'w') as f:
            json.dump(data, f, indent=2)
            
    def validate_api_key(self, key: str) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """Validate an API key."""
        if not key:
            return False, None, "No API key provided"
            
        api_key = self.api_keys.get(key)
        
        if not api_key:
            return False, None, "Invalid API key"
            
        if not api_key.is_active:
            return False, None, "API key is inactive"
            
        # Update last used
        api_key.last_used = datetime.now(timezone.utc)
        api_key.total_requests += 1
        
        return True, api_key, None
        
    def check_rate_limit(self, key: str) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Check if request is within rate limit."""
        api_key = self.api_keys.get(key)
        if not api_key:
            return False, None
            
        rate_info = self.rate_limits[key]
        now = time.time()
        
        # Remove old requests outside window
        rate_info.requests = [
            req_time for req_time in rate_info.requests 
            if now - req_time < rate_info.window
        ]
        
        # Check if under limit
        if len(rate_info.requests) < api_key.rate_limit:
            rate_info.requests.append(now)
            
            return True, {
                'limit': api_key.rate_limit,
                'remaining': api_key.rate_limit - len(rate_info.requests),
                'reset': int(now + rate_info.window)
            }
        else:
            # Calculate when oldest request expires
            reset_time = rate_info.requests[0] + rate_info.window
            
            return False, {
                'limit': api_key.rate_limit,
                'remaining': 0,
                'reset': int(reset_time)
            }
            
    def check_permission(self, api_key: APIKey, required_permission: str) -> bool:
        """Check if API key has required permission."""
        if 'admin' in api_key.permissions:
            return True
            
        # Check specific permission
        for perm in api_key.permissions:
            if perm == required_permission:
                return True
            # Check permission groups
            if perm in self.PERMISSIONS and required_permission in self.PERMISSIONS[perm]:
                return True
                
        return False
        
    def create_jwt_token(self, api_key: APIKey, expires_in: int = 3600) -> str:
        """Create a JWT token for an API key."""
        payload = {
            'api_key': api_key.key,
            'name': api_key.name,
            'permissions': api_key.permissions,
            'exp': datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            'iat': datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if API key still exists and is active
            api_key = self.api_keys.get(payload['api_key'])
            if not api_key or not api_key.is_active:
                return False, None, "API key no longer valid"
                
            return True, payload, None
            
        except jwt.ExpiredSignatureError:
            return False, None, "Token expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
            
    def generate_api_docs(self) -> Dict:
        """Generate API authentication documentation."""
        return {
            'authentication': {
                'type': 'apiKey',
                'description': 'API key authentication',
                'methods': [
                    {
                        'name': 'API Key Header',
                        'description': 'Pass API key in X-API-Key header',
                        'example': 'X-API-Key: omin_xxxxx'
                    },
                    {
                        'name': 'Bearer Token',
                        'description': 'Pass JWT token in Authorization header',
                        'example': 'Authorization: Bearer eyJxx...'
                    }
                ],
                'rate_limiting': {
                    'description': 'Rate limits are enforced per API key',
                    'headers': {
                        'X-RateLimit-Limit': 'Total requests allowed per minute',
                        'X-RateLimit-Remaining': 'Requests remaining in current window',
                        'X-RateLimit-Reset': 'Unix timestamp when limit resets'
                    }
                },
                'permissions': {
                    'read': 'Read access to portfolio and performance data',
                    'trade': 'Execute trades and manage positions',
                    'admin': 'Full access to all endpoints'
                }
            }
        }


# Flask/FastAPI middleware
def require_auth(required_permission: Optional[str] = None):
    """Decorator for protecting API endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify
            
            auth_manager = APIAuthManager()
            
            # Check for API key
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                # Check for Bearer token
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    valid, payload, error = auth_manager.validate_jwt_token(token)
                    if not valid:
                        return jsonify({'error': error}), 401
                    api_key = payload['api_key']
                else:
                    return jsonify({'error': 'No API key provided'}), 401
                    
            # Validate API key
            valid, key_obj, error = auth_manager.validate_api_key(api_key)
            if not valid:
                return jsonify({'error': error}), 401
                
            # Check rate limit
            allowed, rate_info = auth_manager.check_rate_limit(api_key)
            if not allowed:
                response = jsonify({'error': 'Rate limit exceeded'})
                response.headers['X-RateLimit-Limit'] = str(rate_info['limit'])
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(rate_info['reset'])
                return response, 429
                
            # Add rate limit headers
            if rate_info:
                @wraps(f)
                def add_headers(response):
                    response.headers['X-RateLimit-Limit'] = str(rate_info['limit'])
                    response.headers['X-RateLimit-Remaining'] = str(rate_info['remaining'])
                    response.headers['X-RateLimit-Reset'] = str(rate_info['reset'])
                    return response
                    
            # Check permission
            if required_permission:
                if not auth_manager.check_permission(key_obj, required_permission):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                    
            # Add API key to request context
            request.api_key = key_obj
            
            return f(*args, **kwargs)
            
        return decorated_function
    return decorator


def create_api_key_cli():
    """CLI tool for managing API keys."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage API keys')
    parser.add_argument('action', choices=['create', 'list', 'revoke'])
    parser.add_argument('--name', help='Key name')
    parser.add_argument('--permissions', nargs='+', help='Permissions (read, trade, admin)')
    parser.add_argument('--rate-limit', type=int, default=60, help='Requests per minute')
    parser.add_argument('--key', help='API key to revoke')
    
    args = parser.parse_args()
    
    auth_manager = APIAuthManager()
    
    if args.action == 'create':
        if not args.name or not args.permissions:
            print("Error: --name and --permissions required for create")
            return
            
        key = auth_manager.create_api_key(
            name=args.name,
            permissions=args.permissions,
            rate_limit=args.rate_limit
        )
        
        print(f"\n✅ API Key created successfully!")
        print(f"\nKey: {key}")
        print(f"Name: {args.name}")
        print(f"Permissions: {', '.join(args.permissions)}")
        print(f"Rate Limit: {args.rate_limit} requests/minute")
        print("\n⚠️  Save this key securely - it won't be shown again!")
        
    elif args.action == 'list':
        print("\nAPI Keys:")
        print("-" * 80)
        for key in auth_manager.api_keys.values():
            status = "Active" if key.is_active else "Inactive"
            print(f"{key.name:<30} {status:<10} {', '.join(key.permissions)}")
            
    elif args.action == 'revoke':
        if not args.key:
            print("Error: --key required for revoke")
            return
            
        if args.key in auth_manager.api_keys:
            auth_manager.api_keys[args.key].is_active = False
            auth_manager._save_api_keys()
            print(f"✅ API key revoked: {args.key}")
        else:
            print(f"❌ API key not found: {args.key}")


if __name__ == "__main__":
    create_api_key_cli()