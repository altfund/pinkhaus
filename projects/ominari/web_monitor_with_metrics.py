#!/usr/bin/env python3
"""
Enhanced Web Monitor with Authentication, Rate Limiting, and Prometheus Metrics

Features:
- API key authentication
- Rate limiting
- CORS support
- Enhanced security headers
- Prometheus metrics endpoint
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import logging
import os
from datetime import datetime, timezone
from functools import wraps
import time

# Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Import our modules
from api_auth import APIAuthManager, require_auth
from database_v2 import db_manager
from models import Market, Odd
from paper_trading_db import get_db_session as get_paper_session
from paper_trading_models_v2 import PaperTradingSession, PaperTradingPosition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
api_requests = Counter(
    'ominari_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'ominari_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

api_errors = Counter(
    'ominari_api_errors_total',
    'Total API errors',
    ['method', 'endpoint', 'error_type']
)

portfolio_value = Gauge(
    'ominari_portfolio_value',
    'Current portfolio value',
    ['session_id']
)

total_pnl = Gauge(
    'ominari_total_pnl',
    'Total P&L',
    ['session_id']
)

open_positions = Gauge(
    'ominari_open_positions',
    'Number of open positions',
    ['session_id']
)

active_markets = Gauge(
    'ominari_active_markets',
    'Number of active markets'
)

# Create Flask app
app = Flask(__name__)

# Configure CORS (restrict in production)
CORS(app, resources={
    r"/api/*": {
        "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
    }
})

# Initialize auth manager
auth_manager = APIAuthManager()


def track_request(endpoint):
    """Decorator to track API requests with Prometheus."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                
                # Extract status code if it's a tuple response
                if isinstance(result, tuple) and len(result) == 2:
                    response, status_code = result
                else:
                    response = result
                
                return result
                
            except Exception as e:
                status_code = 500
                api_errors.labels(
                    method=request.method,
                    endpoint=endpoint,
                    error_type=type(e).__name__
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                api_requests.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
                api_request_duration.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


@app.after_request
def after_request(response):
    """Add headers to all responses."""
    response = add_security_headers(response)
    
    # Add rate limit headers if present
    if hasattr(request, 'rate_info') and request.rate_info:
        response.headers['X-RateLimit-Limit'] = str(request.rate_info['limit'])
        response.headers['X-RateLimit-Remaining'] = str(request.rate_info['remaining'])
        response.headers['X-RateLimit-Reset'] = str(request.rate_info['reset'])
        
    return response


def authenticate_request():
    """Authenticate the request and check rate limits."""
    # Check for API key
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        # Check for Bearer token
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]
            valid, payload, error = auth_manager.validate_jwt_token(token)
            if not valid:
                return False, error
            api_key = payload['api_key']
        else:
            # Allow unauthenticated access to health check and metrics
            if request.path in ['/health', '/api/health', '/metrics']:
                return True, None
            return False, "No API key provided"
            
    # Validate API key
    valid, key_obj, error = auth_manager.validate_api_key(api_key)
    if not valid:
        return False, error
        
    # Check rate limit
    allowed, rate_info = auth_manager.check_rate_limit(api_key)
    if not allowed:
        request.rate_info = rate_info
        return False, "Rate limit exceeded"
        
    # Store for response headers
    request.rate_info = rate_info
    request.api_key = key_obj
    
    return True, None


@app.before_request
def before_request():
    """Run before each request."""
    # Skip auth for health check, metrics, and docs
    if request.path in ['/health', '/api/health', '/metrics', '/api/docs']:
        return
        
    # Authenticate
    valid, error = authenticate_request()
    if not valid:
        if error == "Rate limit exceeded":
            return jsonify({'error': error}), 429
        return jsonify({'error': error}), 401


# Health check endpoint (no auth required)
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
@track_request('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '2.1.0'
    })


# Metrics endpoint (no auth required)
@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    # Update system metrics
    try:
        with db_manager.get_db_session() as db:
            active = db.query(Market).filter(
                Market.is_finished == False
            ).limit(1000).count()
            active_markets.set(active)
    except:
        pass
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# API documentation
@app.route('/api/docs', methods=['GET'])
@track_request('/api/docs')
def api_docs():
    """Get API documentation."""
    docs = auth_manager.generate_api_docs()
    
    return jsonify({
        'version': '2.1.0',
        'base_url': request.host_url + 'api',
        'authentication': docs['authentication'],
        'endpoints': {
            'status': {
                'GET /api/status': 'Get system status',
                'permissions': ['read']
            },
            'portfolio': {
                'GET /api/trading/portfolio': 'Get portfolio overview',
                'permissions': ['read']
            },
            'positions': {
                'GET /api/trading/positions': 'Get open positions',
                'permissions': ['read']
            },
            'trading': {
                'POST /api/trading/execute': 'Execute trades',
                'permissions': ['trade']
            },
            'admin': {
                'GET /api/admin/stats': 'Get system statistics',
                'permissions': ['admin']
            },
            'metrics': {
                'GET /metrics': 'Prometheus metrics (no auth)',
                'permissions': []
            }
        }
    })


# Protected endpoints
@app.route('/api/status', methods=['GET'])
@track_request('/api/status')
def status():
    """Get system status."""
    try:
        # Check permission
        if not auth_manager.check_permission(request.api_key, 'status'):
            return jsonify({'error': 'Insufficient permissions'}), 403
            
        with db_manager.get_db_session() as db:
            active_markets_count = db.query(Market).filter(
                Market.is_finished == False
            ).limit(100).count()
            
        return jsonify({
            'status': 'operational',
            'active_markets': active_markets_count,
            'api_version': '2.1.0',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/trading/portfolio', methods=['GET'])
@track_request('/api/trading/portfolio')
def get_portfolio():
    """Get portfolio overview."""
    try:
        # Check permission
        if not auth_manager.check_permission(request.api_key, 'portfolio'):
            return jsonify({'error': 'Insufficient permissions'}), 403
            
        session_id = request.args.get('session_id', 'default')
        
        # Get paper trading data
        with get_paper_session() as session:
            trading_session = session.query(PaperTradingSession).filter_by(
                id=session_id
            ).first()
            
            if not trading_session:
                return jsonify({'error': 'Session not found'}), 404
                
            # Get positions
            positions = session.query(PaperTradingPosition).filter_by(
                session_id=session_id
            ).all()
            
            open_positions_list = [p for p in positions if p.status == 'open']
            
            # Calculate metrics
            total_value = trading_session.current_capital
            open_value = sum(p.stake for p in open_positions_list)
            
            # Update Prometheus metrics
            portfolio_value.labels(session_id=session_id).set(float(total_value))
            total_pnl.labels(session_id=session_id).set(float(trading_session.total_pnl))
            open_positions.labels(session_id=session_id).set(len(open_positions_list))
            
            return jsonify({
                'session_id': session_id,
                'total_value': float(total_value),
                'cash_balance': float(total_value - open_value),
                'positions_value': float(open_value),
                'open_positions': len(open_positions_list),
                'total_positions': len(positions),
                'pnl': float(trading_session.total_pnl),
                'pnl_percentage': float((trading_session.current_capital - trading_session.initial_capital) / trading_session.initial_capital * 100)
            })
            
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/trading/execute', methods=['POST'])
@track_request('/api/trading/execute')
def execute_trades():
    """Execute trading recommendations."""
    try:
        # Check permission
        if not auth_manager.check_permission(request.api_key, 'execute'):
            return jsonify({'error': 'Insufficient permissions'}), 403
            
        data = request.json
        session_id = data.get('session_id', 'default')
        dry_run = data.get('dry_run', True)
        
        if dry_run:
            # Just return mock data for dry run
            return jsonify({
                'status': 'dry_run',
                'recommendations': [
                    {
                        'market_id': 'sample_market',
                        'market_name': 'Team A vs Team B',
                        'outcome': 'Team A',
                        'edge': 0.035,
                        'odds': 2.1,
                        'stake': 50.0
                    }
                ],
                'total_stake': 50.0,
                'expected_return': 55.0
            })
        else:
            # Real execution would happen here
            return jsonify({
                'status': 'executed',
                'trades': [],
                'message': 'Trading execution not implemented in demo'
            })
            
    except Exception as e:
        logger.error(f"Execute trades error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/admin/stats', methods=['GET'])
@track_request('/api/admin/stats')
def admin_stats():
    """Get admin statistics."""
    try:
        # Check admin permission
        if not auth_manager.check_permission(request.api_key, 'admin'):
            return jsonify({'error': 'Admin access required'}), 403
            
        stats = {
            'total_api_keys': len(auth_manager.api_keys),
            'active_api_keys': sum(1 for k in auth_manager.api_keys.values() if k.is_active),
            'total_requests': sum(k.total_requests for k in auth_manager.api_keys.values()),
            'api_keys': [
                {
                    'name': k.name,
                    'created': k.created_at.isoformat(),
                    'last_used': k.last_used.isoformat() if k.last_used else None,
                    'requests': k.total_requests,
                    'permissions': k.permissions
                }
                for k in auth_manager.api_keys.values()
            ]
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Run the web monitor."""
    port = int(os.getenv('API_PORT', 8888))
    host = os.getenv('API_HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting authenticated web monitor with metrics on {host}:{port}")
    logger.info("API authentication enabled")
    logger.info("Prometheus metrics available at /metrics")
    
    # Create default API key if needed
    if not auth_manager.api_keys:
        key = auth_manager.create_api_key(
            name="Default Key",
            permissions=['read', 'trade'],
            rate_limit=120
        )
        logger.info(f"Created default API key: {key}")
        logger.info("Save this key - it won't be shown again!")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()