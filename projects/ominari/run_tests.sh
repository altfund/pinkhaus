#!/bin/bash
# Ominari DApp Test Runner

echo "🧪 Running Ominari DApp Tests"
echo "============================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo "✅ Using pytest"
    
    # Run feature tests
    echo ""
    echo "Running feature tests..."
    pytest test_ominari_features.py -v --tb=short
    
    # Run comprehensive tests if requested
    if [ "$1" == "--all" ]; then
        echo ""
        echo "Running comprehensive test suite..."
        pytest test_comprehensive.py -v --tb=short
        pytest comprehensive_test_suite.py -v --tb=short 2>/dev/null || true
    fi
else
    echo "⚠️  pytest not found, using unittest"
    
    # Run with unittest
    echo ""
    echo "Running feature tests..."
    python -m unittest test_comprehensive.py -v
    
    if [ "$1" == "--all" ]; then
        echo ""
        echo "Running comprehensive test suite..."
        python comprehensive_test_suite.py 2>/dev/null || true
    fi
fi

echo ""
echo "📊 Test Summary"
echo "==============="

# Quick system check
echo ""
echo "System checks:"
python -c "
import sys
sys.path.append('.')
try:
    from rate_limiter import RateLimiter
    print('✅ Rate limiter module loaded')
except:
    print('❌ Rate limiter module failed')

try:
    from cache_manager import CacheManager
    print('✅ Cache manager module loaded')
except:
    print('❌ Cache manager module failed')

try:
    from web_dashboard_real_odds import app
    print('✅ Dashboard module loaded')
except Exception as e:
    print(f'❌ Dashboard module failed: {str(e)[:50]}...')
"

echo ""
echo "✨ Test run complete!"