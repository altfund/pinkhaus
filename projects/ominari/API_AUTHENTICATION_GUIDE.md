# Ominari Trading System - API Authentication Guide

## Overview

The Ominari Trading System API uses API key authentication with rate limiting and permission-based access control. This guide covers how to authenticate, manage API keys, and use the API client.

## Quick Start

### 1. Create an API Key

```bash
# Create a new API key with trade permissions
python api_auth.py create --name "My Trading Bot" --permissions read trade --rate-limit 120

# Output:
# ✅ API Key created successfully!
# 
# Key: omin_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Name: My Trading Bot
# Permissions: read, trade
# Rate Limit: 120 requests/minute
# 
# ⚠️  Save this key securely - it won't be shown again!
```

### 2. Set Environment Variable

```bash
export OMINARI_API_KEY="omin_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Use the API Client

```python
from ominari_api_client import OminariAPIClient

# Client automatically uses OMINARI_API_KEY from environment
client = OminariAPIClient()

# Or provide directly
client = OminariAPIClient(api_key="omin_xxxxx")

# Make API calls
status = client.get_system_status()
portfolio = client.get_portfolio("default")
```

## Authentication Methods

### 1. API Key Header

Pass your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: omin_xxxxx" http://localhost:8888/api/status
```

### 2. Bearer Token (JWT)

For longer sessions, exchange your API key for a JWT token:

```python
# Get JWT token
response = requests.post('http://localhost:8888/api/auth/token', 
                        headers={'X-API-Key': 'omin_xxxxx'})
token = response.json()['token']

# Use token
requests.get('http://localhost:8888/api/status',
            headers={'Authorization': f'Bearer {token}'})
```

## Permission Levels

### read
- View portfolio and positions
- Get performance metrics
- Check system status
- View market data

### trade
- All read permissions
- Execute trades
- Close positions
- Manage paper trading sessions

### admin
- All permissions
- View system statistics
- Manage API keys
- Access debug endpoints

## Rate Limiting

Each API key has a rate limit (default: 60 requests/minute). Rate limit info is returned in headers:

```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 119
X-RateLimit-Reset: 1694123456
```

## API Key Management

### List API Keys
```bash
python api_auth.py list

# Output:
# API Keys:
# --------------------------------------------------------------------------------
# Development Key               Active     read, trade
# Production Bot               Active     read, trade, admin
# Test Key                     Inactive   read
```

### Revoke API Key
```bash
python api_auth.py revoke --key omin_xxxxx

# Output:
# ✅ API key revoked: omin_xxxxx
```

## Using the Python Client

### Basic Usage

```python
from ominari_api_client import OminariAPIClient

# Initialize client
client = OminariAPIClient(
    base_url="http://localhost:8888",
    api_key="omin_xxxxx",
    timeout=30
)

# Check system health
if client.health_check():
    print("✅ API is healthy")

# Get trading status
status = client.get_trading_status()
print(f"Trading active: {status['is_active']}")

# Get portfolio
portfolio = client.get_portfolio("default")
print(f"Total value: ${portfolio.total_value:,.2f}")
print(f"Open positions: {portfolio.open_positions}")

# Execute trades (dry run)
result = client.execute_trades(
    session_id="default",
    dry_run=True
)
print(f"Recommendations: {len(result['recommendations'])}")

# Get recent activity
activity = client.get_recent_activity(hours=24)
print(f"Recent bets: {len(activity['recent_bets'])}")
```

### Async Usage

```python
import asyncio
from ominari_api_client import AsyncOminariAPIClient

async def main():
    async with AsyncOminariAPIClient(api_key="omin_xxxxx") as client:
        status = await client.get_system_status()
        print(f"Status: {status['status']}")

asyncio.run(main())
```

### Error Handling

```python
from ominari_api_client import OminariAPIClient, OminariAPIError

client = OminariAPIClient(api_key="omin_xxxxx")

try:
    portfolio = client.get_portfolio("invalid_session")
except OminariAPIError as e:
    print(f"API Error: {e}")
    # Handle error appropriately
```

## API Endpoints

### Public Endpoints (No Auth Required)
- `GET /health` - Health check
- `GET /api/health` - Health check
- `GET /api/docs` - API documentation

### Protected Endpoints

#### System Status (read permission)
- `GET /api/status` - System status
- `GET /api/database-stats` - Database statistics

#### Trading (read permission)
- `GET /api/trading/status` - Trading system status
- `GET /api/trading/portfolio` - Portfolio overview
- `GET /api/trading/positions` - Open positions
- `GET /api/trading/recent` - Recent activity
- `GET /api/trading/evaluation-stats` - Market evaluation stats
- `GET /api/trading/sessions` - Paper trading sessions
- `GET /api/trading/logs` - System logs

#### Trading Execution (trade permission)
- `POST /api/trading/execute` - Execute trades
- `POST /api/trading/positions/{id}/close` - Close position

#### Performance (read permission)
- `GET /api/performance` - Performance metrics

#### Dashboard (read permission)
- `GET /api/dashboard/unified` - Unified dashboard data

#### Admin (admin permission)
- `GET /api/admin/stats` - API usage statistics

## Security Best Practices

1. **Store API Keys Securely**
   - Never commit API keys to version control
   - Use environment variables or secure vaults
   - Rotate keys regularly

2. **Use Appropriate Permissions**
   - Only grant necessary permissions
   - Use read-only keys for monitoring
   - Restrict admin keys

3. **Monitor Usage**
   - Check rate limit headers
   - Monitor for unusual activity
   - Review logs regularly

4. **HTTPS in Production**
   - Always use HTTPS in production
   - Consider IP whitelisting
   - Implement request signing for extra security

## Troubleshooting

### Authentication Errors

```json
{"error": "No API key provided"}
```
Solution: Add `X-API-Key` header or set `OMINARI_API_KEY` environment variable

```json
{"error": "Invalid API key"}
```
Solution: Check your API key is correct and hasn't been revoked

```json
{"error": "API key is inactive"}
```
Solution: Your key has been revoked. Create a new one.

### Rate Limit Errors

```json
{"error": "Rate limit exceeded"}
```
Solution: Wait for rate limit reset (check `X-RateLimit-Reset` header) or upgrade your key's rate limit

### Permission Errors

```json
{"error": "Insufficient permissions"}
```
Solution: Your API key doesn't have the required permission. Create a new key with appropriate permissions.

## Example: Complete Trading Bot

```python
#!/usr/bin/env python3
"""
Example trading bot using authenticated API
"""

import os
import time
import logging
from ominari_api_client import OminariAPIClient, OminariAPIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv('OMINARI_API_KEY')
SESSION_ID = os.getenv('TRADING_SESSION', 'default')
DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'

def main():
    """Run trading bot."""
    # Initialize client
    client = OminariAPIClient(api_key=API_KEY)
    
    # Wait for API to be ready
    if not client.wait_for_ready():
        logger.error("API not available")
        return
    
    logger.info("Trading bot started")
    
    while True:
        try:
            # Check trading status
            status = client.get_trading_status()
            if not status['is_active']:
                logger.info("Trading not active, waiting...")
                time.sleep(60)
                continue
            
            # Get current portfolio
            portfolio = client.get_portfolio(SESSION_ID)
            logger.info(f"Portfolio value: ${portfolio.total_value:,.2f}")
            logger.info(f"Open positions: {portfolio.open_positions}")
            
            # Execute trading recommendations
            result = client.execute_trades(
                session_id=SESSION_ID,
                dry_run=DRY_RUN
            )
            
            if result['recommendations']:
                logger.info(f"Found {len(result['recommendations'])} opportunities")
                for rec in result['recommendations']:
                    logger.info(f"  {rec['market_name']}: {rec['outcome']} "
                              f"@ {rec['odds']} (edge: {rec['edge']:.2%})")
            
            # Wait before next iteration
            time.sleep(300)  # 5 minutes
            
        except OminariAPIError as e:
            logger.error(f"API error: {e}")
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
```

## Next Steps

1. Create your API key with appropriate permissions
2. Set up your environment variables
3. Install the client: `pip install requests`
4. Start making API calls!

For more examples and advanced usage, check out the demo scripts:
- `demo_api_client.py` - Basic usage examples
- `demo_trading_bot.py` - Complete trading bot
- `demo_monitoring.py` - System monitoring dashboard