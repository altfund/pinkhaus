# Testing Framework for Ominari Trading System

## Overview

This document describes the comprehensive testing framework for the Ominari trading system, ensuring reliable deployment and operation.

## Test Structure

### 1. Quick Deployment Tests (`run_deployment_tests.py`)
**Purpose**: Fast validation of critical systems for deployment  
**Runtime**: ~10 seconds  
**Usage**: `just test-deployment`

**Tests**:
- ✅ Database connectivity (PostgreSQL)
- ✅ Dashboard accessibility (with STOP button)
- ✅ Paper trading system functionality
- ✅ Stop loss system operations

### 2. Comprehensive Test Suite (`comprehensive_test_suite.py`)
**Purpose**: Full system validation with detailed reporting  
**Runtime**: ~2-5 minutes  
**Usage**: `just test-comprehensive`

**Test Categories**:
- Environment Setup
- Database Connectivity
- Data Quality Analysis
- Paper Trading System
- Portfolio Trading Engine
- Stop Loss System
- Dashboard Functionality
- WebSocket Connectivity
- Edge Calculation
- Risk Management
- System Integration

### 3. Pytest Deployment Tests (`tests/test_deployment.py`)
**Purpose**: Structured pytest-based deployment validation  
**Usage**: `uv run pytest tests/test_deployment.py -v`

**Markers**:
- `@pytest.mark.deployment`: Core deployment tests
- `@pytest.mark.database`: Database-dependent tests
- `@pytest.mark.dashboard`: Dashboard functionality tests
- `@pytest.mark.trading`: Trading system tests

## Quick Reference Commands

```bash
# Fast deployment validation (recommended for CI/CD)
just test-deployment

# Full system validation with detailed report
just test-comprehensive

# Structured pytest validation
just validate-deployment

# Run all existing unit/integration tests
just test

# Run specific test categories
just test-unit
just test-integration

# Run with coverage
just test-coverage
```

## Test Results Interpretation

### Quick Deployment Tests
- **4/4 PASSED**: ✅ System ready for deployment
- **Any failures**: ❌ Critical issues must be resolved

**Critical Systems Checked**:
1. **Database**: Can connect and query PostgreSQL
2. **Dashboard**: Web interface responds with STOP button
3. **Paper Trading**: Session management works
4. **Stop Loss**: Safety systems operational

### Comprehensive Test Suite
Generates detailed JSON report with:
- Overall deployment readiness assessment
- Individual test results and metrics
- Specific recommendations for failures
- Performance and quality metrics

**Sample Report Structure**:
```json
{
  "test_summary": {
    "overall_status": "PASS",
    "deployment_ready": true,
    "total_tests": 11,
    "passed": 11,
    "failed": 0,
    "critical_failures": []
  },
  "detailed_results": { ... },
  "recommendations": [ ... ]
}
```

## Integration with Development Workflow

### Before Deployment
```bash
# 1. Quick validation
just test-deployment

# 2. If passed, run comprehensive suite
just test-comprehensive

# 3. Check generated report for any concerns
cat test_report_*.json | jq '.test_summary'
```

### Continuous Integration
Add to CI pipeline:
```yaml
test:
  script: just test-deployment
  allow_failure: false
```

### Development Testing
```bash
# During development
just test-unit

# Before committing major changes  
just test-integration

# Before deployment
just validate-deployment
```

## Test Configuration

### Environment Requirements
```bash
export PG_HOST=localhost
export PG_PORT=5999
export PG_USER=ominari_user
export PG_PASSWORD=ominari_2025_secure
export PG_DB=ominari_production
export USE_POSTGRESQL=1
```

### pytest Configuration (`pytest.ini`)
- Custom markers for test categorization
- Warning filters for cleaner output
- Test discovery patterns
- Coverage configuration

## Troubleshooting

### Common Test Failures

**Database Connection Failed**
```bash
# Check PostgreSQL is running
systemctl status postgresql
# Check connection manually
psql -h localhost -p 5999 -U ominari_user -d ominari_production
```

**Dashboard Not Accessible**
```bash
# Check web monitor is running
ps aux | grep web_monitor.py
# Check port availability
netstat -tlnp | grep 8888
```

**Paper Trading Issues**
```bash
# Check session table
psql -c "SELECT * FROM paper_trading_sessions WHERE status = 'active';"
```

### Test Environment Isolation

Tests use:
- Temporary databases where applicable
- Non-destructive queries on production data
- Read-only operations for safety
- Separate test configurations

## Extending Tests

### Adding New Tests

1. **Quick Tests**: Add to `run_deployment_tests.py`
   - Keep tests fast (<2 seconds each)
   - Focus on critical functionality
   - Return boolean pass/fail

2. **Comprehensive Tests**: Add to `comprehensive_test_suite.py`
   - More detailed validation
   - Rich error reporting
   - Performance metrics

3. **Pytest Tests**: Add to `tests/test_deployment.py`
   - Use appropriate markers
   - Follow pytest conventions
   - Include documentation

### Test Categories to Consider
- **Performance Tests**: Response time, throughput
- **Security Tests**: Authentication, authorization
- **Load Tests**: Concurrent user simulation
- **Data Integrity Tests**: Database consistency
- **API Tests**: REST/WebSocket endpoints
- **Browser Tests**: UI automation

## Deployment Readiness Criteria

### ✅ System Ready When:
- All quick deployment tests pass (4/4)
- Database contains recent market data (>1000 active markets)
- Dashboard loads with STOP button visible
- Paper trading session active with positive bankroll
- Stop loss monitoring operational

### ❌ System NOT Ready When:
- Any database connectivity issues
- Dashboard not accessible
- No active paper trading session
- Stop loss system errors
- Critical exceptions in logs

## Monitoring and Alerting

### Post-Deployment Monitoring
- Run `just test-deployment` every 15 minutes
- Alert on any failures
- Log all test results for trend analysis
- Monitor system performance metrics

### Health Check Endpoint
Consider implementing:
```http
GET /health
Response: {"status": "healthy", "components": {"database": "ok", "trading": "ok"}}
```

## Summary

This testing framework provides:
- 🚀 **Fast validation** for deployment decisions
- 🔍 **Comprehensive analysis** for quality assurance  
- 🛡️ **Safety checks** for critical trading operations
- 📊 **Detailed reporting** for troubleshooting
- 🔄 **CI/CD integration** for automated workflows

**Recommendation**: Always run `just test-deployment` before any deployment to production.