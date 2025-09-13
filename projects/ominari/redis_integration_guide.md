
# Redis Cache Integration Guide

## 1. Install Redis (if not available)
```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

## 2. Add caching to existing functions

### Market queries:
```python
@cached(ttl=300, prefix="market")
def get_market_by_id(market_id: str):
    # Existing database query
    return query_result
```

### Signal calculations:
```python
@cached(ttl=600, prefix="signal")
def calculate_signal_probability(market_data):
    # Expensive signal calculation
    return probability
```

### Blockchain data:
```python
@cached(ttl=3600, prefix="blockchain")
def get_blockchain_market_data(address, network):
    # RPC call to blockchain
    return market_data
```

## 3. Use trading cache for real-time data

```python
from redis_caching_system import trading_cache

# Cache market data
trading_cache.cache_market_data(market_id, data)

# Cache odds updates
trading_cache.cache_odds_data(market_id, odds)

# Cache signal results
trading_cache.cache_signal_result(market_id, signal_name, probability)
```

## 4. Performance monitoring

```python
# Get cache performance
summary = trading_cache.get_cache_summary()
print(f"Hit rate: {summary['performance']['hit_rate']:.1%}")
```
