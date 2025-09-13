# Ominari Trading System API Guide

## Overview

The Ominari Trading System provides a comprehensive REST API for interacting with the sports betting platform. The API enables:

- Real-time trading operations
- Portfolio management
- Performance analytics
- Market data access
- System monitoring

## Quick Start

### Using the Python Client

```python
from ominari_api_client import OminariAPIClient

# Initialize client
client = OminariAPIClient("http://localhost:8888")

# Check system health
if client.health_check():
    print("System is healthy")

# Get portfolio overview
portfolio = client.get_portfolio("your_session_id")
print(f"Total Value: ${portfolio.total_value}")
print(f"Daily P&L: ${portfolio.daily_pnl}")

# Execute trades
result = client.execute_trades(
    session_id="your_session_id",
    dry_run=True  # Set to False to execute
)
```

### Using cURL

```bash
# Get system status
curl http://localhost:8888/api/status

# Get portfolio
curl "http://localhost:8888/api/trading/portfolio?session_id=your_session_id"

# Execute trades (dry run)
curl -X POST http://localhost:8888/api/trading/execute \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your_session_id", "dry_run": true}'
```

### Using JavaScript/TypeScript

```javascript
// Get system status
const response = await fetch('http://localhost:8888/api/status');
const status = await response.json();

// Get portfolio
const portfolio = await fetch(
  `http://localhost:8888/api/trading/portfolio?session_id=${sessionId}`
).then(res => res.json());

// Execute trades
const trades = await fetch('http://localhost:8888/api/trading/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    dry_run: true
  })
}).then(res => res.json());
```

## Authentication

Currently, the API does not require authentication for development. In production:

1. API keys will be required for all requests
2. Rate limiting will be enforced
3. SSL/TLS will be mandatory

## Common Workflows

### 1. Starting a Trading Session

```python
# 1. Get available sessions
sessions = client.get_sessions()

# 2. Select or note session ID
session_id = sessions['sessions'][0]['id']

# 3. Check portfolio status
portfolio = client.get_portfolio(session_id)

# 4. Review strategy configuration
strategy = client.get_strategy_config()
```

### 2. Monitoring Positions

```python
# Get all open positions
positions = client.get_positions(session_id)

for position in positions['positions']:
    print(f"{position['home_team']} vs {position['away_team']}")
    print(f"Stake: ${position['amount']}, Potential: ${position['potential_payout']}")
    print(f"Status: {position['status']}")
    
# Monitor performance
perf = client.get_performance(session_id, days=7)
print(f"Week Performance: {perf['total_pnl']}")
```

### 3. Executing Trades

```python
# 1. Get market evaluation (dry run)
evaluation = client.execute_trades(session_id, dry_run=True)

print(f"Found {len(evaluation['recommendations'])} opportunities")
print(f"Total stake: ${evaluation['total_staked']}")
print(f"Expected return: ${evaluation['expected_return']}")

# 2. Review recommendations
for rec in evaluation['recommendations']:
    print(f"{rec['market_name']}: {rec['outcome']}")
    print(f"Edge: {rec['edge']:.2%}, Stake: ${rec['stake']}")

# 3. Execute if satisfied
if input("Execute trades? (y/n): ").lower() == 'y':
    result = client.execute_trades(session_id, dry_run=False)
    print(f"Executed {len(result['executed'])} trades")
```

### 4. Risk Monitoring

```python
# Get unified dashboard data
dashboard = client.get_unified_dashboard(session_id)

# Check risk metrics
portfolio = dashboard['portfolio']
print(f"Current Drawdown: {portfolio['current_drawdown']:.2%}")
print(f"Max Drawdown: {portfolio['max_drawdown']:.2%}")
print(f"Exposure: ${portfolio['total_exposure']}")

# Review recent activity
activity = dashboard['recent_activity']
print(f"Recent bets: {len(activity['recent_bets'])}")
```

## Error Handling

### Python Client

```python
from ominari_api_client import OminariAPIClient, OminariAPIError

client = OminariAPIClient()

try:
    portfolio = client.get_portfolio("invalid_session")
except OminariAPIError as e:
    print(f"API Error: {e}")
    # Handle error appropriately
```

### HTTP Status Codes

| Code | Description | Action |
|------|-------------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check parameters |
| 404 | Not Found | Verify endpoint/ID |
| 429 | Rate Limited | Retry after delay |
| 500 | Server Error | Retry or contact support |

## Best Practices

### 1. Rate Limiting

- Respect rate limits (when implemented)
- Implement exponential backoff for retries
- Cache responses when appropriate

### 2. Error Handling

```python
import time

def retry_request(func, max_retries=3, delay=1):
    """Retry failed requests with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except OminariAPIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (2 ** attempt))
```

### 3. Efficient Polling

```python
import asyncio

async def monitor_positions(client, session_id):
    """Efficiently monitor positions."""
    while True:
        try:
            positions = await client.get_positions(session_id)
            process_positions(positions)
            await asyncio.sleep(30)  # Poll every 30 seconds
        except Exception as e:
            print(f"Monitor error: {e}")
            await asyncio.sleep(60)  # Back off on error
```

### 4. Data Validation

Always validate API responses:

```python
def validate_portfolio(data):
    """Validate portfolio data structure."""
    required_fields = ['total_value', 'cash_balance', 'open_positions']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    if data['total_value'] < 0:
        raise ValueError("Invalid total value")
    
    return True
```

## WebSocket Support (Future)

WebSocket support is planned for real-time updates:

```javascript
// Future WebSocket implementation
const ws = new WebSocket('ws://localhost:8888/ws');

ws.on('connect', () => {
    // Subscribe to updates
    ws.send(JSON.stringify({
        action: 'subscribe',
        channels: ['positions', 'performance']
    }));
});

ws.on('message', (data) => {
    const update = JSON.parse(data);
    handleRealtimeUpdate(update);
});
```

## SDK Development

### TypeScript Types

```typescript
// ominari-types.ts
export interface Portfolio {
  total_value: number;
  cash_balance: number;
  positions_value: number;
  total_exposure: number;
  open_positions: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_pnl: number;
  total_pnl_pct: number;
  max_drawdown: number;
  current_drawdown: number;
}

export interface Position {
  id: string;
  session_id: string;
  market_id: string;
  sport: string;
  league: string;
  home_team: string;
  away_team: string;
  market_type: string;
  outcome: string;
  odds: number;
  amount: number;
  potential_payout: number;
  created_at: string;
  expires_at: string;
  status: 'pending' | 'open' | 'won' | 'lost' | 'void';
  pnl?: number;
}
```

### React Hooks (Example)

```typescript
// useOminariAPI.ts
import { useState, useEffect } from 'react';

export function usePortfolio(sessionId: string) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchPortfolio() {
      try {
        const response = await fetch(
          `/api/trading/portfolio?session_id=${sessionId}`
        );
        const data = await response.json();
        setPortfolio(data);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 30000); // Refresh every 30s
    
    return () => clearInterval(interval);
  }, [sessionId]);

  return { portfolio, loading, error };
}
```

## API Versioning

The API follows semantic versioning:

- Current version: 1.0.0
- Version in URL: `/api/v1/...` (future)
- Version in header: `X-API-Version: 1.0.0`

## Support

- Documentation: See `API_DOCUMENTATION.md`
- OpenAPI Spec: `openapi_spec.yaml`
- Issues: GitHub Issues
- Email: support@ominari.com

## Changelog

### v1.0.0 (Current)
- Initial API release
- Full trading operations
- Performance analytics
- Real-time monitoring
- Paper trading support