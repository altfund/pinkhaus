# Ominari DApp - Testing Guide

## Overview

This guide covers testing for the Ominari DApp, focusing on the new features:
- Rate limiting
- Caching layer
- Health monitoring
- WebSocket connections

## Test Files

### 1. `test_ominari_features.py`
Feature-specific tests using pytest:
- Rate limiting functionality
- Cache operations
- Health endpoints
- WebSocket behavior
- Integration scenarios

### 2. `test_comprehensive.py`
Comprehensive test suite using unittest:
- All major components
- Error handling
- Edge cases
- Performance tests

### 3. `comprehensive_test_suite.py`
Original test suite for:
- Database operations
- Paper trading
- Portfolio optimization

## Running Tests

### Quick Test
```bash
# Run feature tests only
./run_tests.sh
```

### Full Test Suite
```bash
# Run all tests
./run_tests.sh --all
```

### Specific Test Categories
```bash
# Rate limiting tests
pytest test_ominari_features.py::TestRateLimitingFeatures -v

# Caching tests
pytest test_ominari_features.py::TestCachingFeatures -v

# Health monitoring tests
pytest test_ominari_features.py::TestHealthMonitoring -v

# WebSocket tests
pytest test_ominari_features.py::TestWebSocketFeatures -v
```

### Using Docker
```bash
# Run tests in Docker
docker-compose run --rm ominari-dashboard pytest test_ominari_features.py
```

## Test Coverage

### Rate Limiting
- ✅ Basic rate limiting per IP
- ✅ Multiple IP tracking
- ✅ Time window reset
- ✅ IP blocking
- ✅ Adaptive rate adjustment

### Caching
- ✅ Basic get/set operations
- ✅ TTL expiration
- ✅ Cache statistics
- ✅ Decorator functionality
- ✅ Market data caching

### Health Monitoring
- ✅ Health endpoint (healthy/unhealthy)
- ✅ Metrics endpoint (Prometheus format)
- ✅ Cache stats endpoint
- ✅ Service status checks

### WebSocket
- ✅ Connection establishment
- ✅ Dashboard data requests
- ✅ Rate limiting on WebSocket

### Integration
- ✅ Rate limiting with caching
- ✅ Concurrent request handling
- ✅ Full system flow

## Performance Tests

### Load Testing
```bash
# Simple load test
for i in {1..100}; do
    curl -s http://localhost:8888/health &
done
wait

# Check rate limiting
curl http://localhost:8888/cache-stats | jq .
```

### Memory Testing
```python
# Monitor memory usage during tests
python -m memory_profiler test_ominari_features.py
```

## Mocking Guidelines

### Database Mocking
```python
@patch('database_v2.db_manager')
def test_something(mock_db_manager):
    mock_session = Mock()
    mock_db_manager.get_db_session.return_value.__enter__.return_value = mock_session
```

### Time Mocking
```python
@patch('time.time')
def test_expiration(mock_time):
    mock_time.return_value = 1000  # Fixed timestamp
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run tests
        run: ./run_tests.sh --all
```

## Debugging Tests

### Verbose Output
```bash
pytest test_ominari_features.py -vvs
```

### Debug Specific Test
```bash
pytest test_ominari_features.py::TestCachingFeatures::test_cache_ttl_expiration -vvs --pdb
```

### Check Test Coverage
```bash
pytest --cov=. --cov-report=html test_ominari_features.py
open htmlcov/index.html
```

## Common Issues

### Import Errors
- Ensure PYTHONPATH includes project directory
- Check all dependencies are installed

### Database Connection
- Mock database connections in tests
- Use test database for integration tests

### Async Tests
- Use `pytest-asyncio` for async test support
- Properly await async functions

### Rate Limit Tests
- Clear rate limiter state between tests
- Mock time for time-window tests

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies
3. **Assertions**: Use specific assertions
4. **Naming**: Descriptive test names
5. **Speed**: Keep tests fast (<1s each)
6. **Coverage**: Aim for >80% code coverage

## Adding New Tests

1. Create test class in appropriate file
2. Use fixtures for common setup
3. Mock external dependencies
4. Add to test runner if needed
5. Document any special requirements

## Maintenance

- Run tests before commits
- Update tests when adding features
- Review test coverage regularly
- Keep tests simple and focused