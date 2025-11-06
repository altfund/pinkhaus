# Ominari Trading - Altfund2 Django Integration

This Django app integrates the Ominari DApp trading protocol with the Altfund2 platform, creating a bridge between Web2 and Web3.

## Installation

1. **Copy the app to altfund2**:
```bash
cp -r ominari_trading /path/to/altfund2/
```

2. **Add to INSTALLED_APPS** in `altfund_website/settings/base.py`:
```python
INSTALLED_APPS = [
    # ... existing apps ...
    'ominari_trading',
]
```

3. **Add URL configuration** in `altfund_website/urls.py`:
```python
urlpatterns = [
    # ... existing patterns ...
    path('api/v1/ominari/', include('ominari_trading.api.urls')),
    path('ominari/', include('ominari_trading.urls')),
]
```

4. **Run migrations**:
```bash
python manage.py makemigrations ominari_trading
python manage.py migrate
```

5. **Configure Celery tasks** - Add to your Celery beat schedule:
```python
CELERY_BEAT_SCHEDULE.update({
    'sync-ominari-events': {
        'task': 'ominari_trading.tasks.sync_blockchain_events',
        'schedule': timedelta(minutes=1),
        'args': (137,)  # Polygon chain ID
    },
    'process-ominari-events': {
        'task': 'ominari_trading.tasks.process_blockchain_events',
        'schedule': timedelta(seconds=30),
    },
})
```

## Configuration

### Environment Variables

Add to your environment:
```bash
# Blockchain RPCs
POLYGON_RPC_URL=https://polygon-rpc.com
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc

# Contract Addresses (after deployment)
OMINARI_TRADING_ENGINE_POLYGON=0x...
OMINARI_KELLY_OPTIMIZER_POLYGON=0x...
OMINARI_CHUNK_MANAGER_POLYGON=0x...
```

### Initial Setup

1. **Create chain networks**:
```python
from django.core.management import call_command
call_command('loaddata', 'ominari_trading/fixtures/chains.json')
```

Or manually in Django admin:
- Polygon Mainnet (chain_id: 137)
- Arbitrum One (chain_id: 42161)
- Sepolia Testnet (chain_id: 11155111)

2. **Configure contract addresses** after deployment

## Features

### Models
- **ChainNetwork**: Blockchain network configurations
- **Web3Account**: User wallet connections
- **TradingSession**: On-chain trading sessions
- **Position**: Individual bets/positions
- **MarketData**: Cached market information
- **BlockchainEvent**: Raw blockchain events for processing

### API Endpoints

#### Authentication
All endpoints require authentication except chain listings and market data.

#### Chains
- `GET /api/v1/ominari/chains/` - List supported chains

#### Wallet Management
- `GET /api/v1/ominari/wallets/` - List connected wallets
- `POST /api/v1/ominari/wallets/verify_ownership/` - Verify wallet ownership

#### Trading Sessions
- `GET /api/v1/ominari/sessions/` - List user's sessions
- `POST /api/v1/ominari/sessions/` - Create new session
- `POST /api/v1/ominari/sessions/{id}/refresh/` - Refresh from blockchain
- `POST /api/v1/ominari/sessions/{id}/close/` - Close session

#### Positions
- `GET /api/v1/ominari/positions/` - List positions
- `POST /api/v1/ominari/positions/` - Place new bet

#### Markets
- `GET /api/v1/ominari/markets/` - Browse available markets
  - Query params: `sport`, `chain`, `hours`

#### Optimization
- `POST /api/v1/ominari/optimize/optimize/` - Run Kelly optimization
- `POST /api/v1/ominari/optimize/prepare_transaction/` - Prepare blockchain transaction

### Admin Interface

The app includes a comprehensive Django admin interface for:
- Monitoring trading sessions
- Viewing positions and P/L
- Managing chain configurations
- Processing blockchain events
- Analyzing market data

### Celery Tasks

- `sync_blockchain_events`: Fetches new events from blockchain
- `process_blockchain_events`: Processes fetched events
- `update_session_stats`: Updates session statistics
- `cleanup_old_events`: Removes old processed events

## Frontend Integration

### Connect Wallet
```javascript
// Example using ethers.js
const response = await fetch('/api/v1/ominari/wallets/verify_ownership/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Token ${authToken}`
  },
  body: JSON.stringify({
    wallet_address: address,
    signature: signature,
    message: message,
    chain_id: 137
  })
});
```

### Create Session
```javascript
const response = await fetch('/api/v1/ominari/sessions/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Token ${authToken}`
  },
  body: JSON.stringify({
    chain_id: 137,
    initial_bankroll: '100.0',
    transaction_hash: txHash  // After blockchain confirmation
  })
});
```

### Optimize Portfolio
```javascript
const response = await fetch('/api/v1/ominari/optimize/optimize/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Token ${authToken}`
  },
  body: JSON.stringify({
    session_id: sessionId,
    market_ids: ['0x123...', '0x456...'],
    chunk_duration_minutes: 120,
    use_half_kelly: true,
    max_stake_percentage: 5
  })
});
```

## Development

### Running Tests
```bash
python manage.py test ominari_trading
```

### Mock Mode
The app includes a mock blockchain client for development. It's automatically used for testnets or when contract addresses aren't configured.

### Adding New Chains
1. Add to ChainNetwork model
2. Configure RPC URL
3. Deploy contracts to that chain
4. Update contract addresses in admin

## Security Considerations

1. **Wallet Verification**: Always verify wallet ownership via signature
2. **Transaction Signing**: Never store private keys - all transactions signed client-side
3. **Rate Limiting**: Implement rate limiting on API endpoints
4. **CORS**: Configure CORS appropriately for your frontend domain

## Monitoring

### Sentry Integration
Errors are automatically reported to Sentry if configured in Altfund2.

### Logging
```python
import logging
logger = logging.getLogger('ominari_trading')
```

### Metrics
- Active sessions count
- Total volume traded
- Win rate by user
- Gas costs by chain

## Future Enhancements

1. **WebSocket Support**: Real-time position updates
2. **Advanced Analytics**: Historical performance charts
3. **Social Features**: Leaderboards and strategy sharing
4. **Multi-chain Aggregation**: Cross-chain position management
5. **Automated Strategies**: Set-and-forget betting strategies