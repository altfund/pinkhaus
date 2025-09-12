#!/usr/bin/env python3
"""
Ominari Trading System API Client Library

A comprehensive Python client for interacting with the Ominari Trading System API.
Provides easy-to-use methods for all API endpoints with proper error handling.
"""

import requests
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


class PositionStatus(Enum):
    """Position status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    WON = "won"
    LOST = "lost"
    VOID = "void"


@dataclass
class Position:
    """Trading position data class."""
    id: str
    session_id: str
    market_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    market_type: str
    outcome: str
    odds: float
    amount: float
    potential_payout: float
    created_at: datetime
    expires_at: datetime
    status: PositionStatus
    pnl: Optional[float] = None


@dataclass
class PortfolioOverview:
    """Portfolio overview data class."""
    total_value: float
    cash_balance: float
    positions_value: float
    total_exposure: float
    open_positions: int
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    current_drawdown: float


class OminariAPIError(Exception):
    """Base exception for Ominari API errors."""
    pass


class OminariAPIClient:
    """
    Client for interacting with the Ominari Trading System API.
    
    Example:
        client = OminariAPIClient("http://localhost:8888")
        status = client.get_system_status()
        portfolio = client.get_portfolio("session_id")
    """
    
    def __init__(self, base_url: str = "http://localhost:8888", 
                 api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the Ominari API
            api_key: API key for authentication (optional, can use env var)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('OMINARI_API_KEY')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Set up authentication
        if self.api_key:
            self.session.headers['X-API-Key'] = self.api_key
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Union[Dict, List]:
        """Make a request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            
            # Return JSON if available
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.Timeout:
            raise OminariAPIError(f"Request to {url} timed out")
        except requests.exceptions.ConnectionError:
            raise OminariAPIError(f"Failed to connect to {url}")
        except requests.exceptions.HTTPError as e:
            raise OminariAPIError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise OminariAPIError(f"Unexpected error: {str(e)}")
    
    # System endpoints
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return self._request('GET', '/api/status')
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self._request('GET', '/api/database-stats')
    
    # Trading endpoints
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get trading system status."""
        return self._request('GET', '/api/trading/status')
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get current strategy configuration."""
        return self._request('GET', '/api/strategy')
    
    def get_portfolio(self, session_id: str) -> PortfolioOverview:
        """
        Get portfolio overview for a session.
        
        Args:
            session_id: Paper trading session ID
            
        Returns:
            PortfolioOverview object
        """
        data = self._request('GET', '/api/trading/portfolio', params={'session_id': session_id})
        return PortfolioOverview(**data)
    
    def get_positions(self, 
                     session_id: Optional[str] = None,
                     include_closed: bool = False) -> Dict[str, Any]:
        """
        Get trading positions.
        
        Args:
            session_id: Optional session ID filter
            include_closed: Include closed positions
            
        Returns:
            Dict with positions list and summary
        """
        params = {}
        if session_id:
            params['session_id'] = session_id
        if include_closed:
            params['include_closed'] = True
            
        return self._request('GET', '/api/trading/positions', params=params)
    
    def close_position(self, position_id: str, final_odds: Optional[float] = None) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            position_id: Position ID to close
            final_odds: Optional final odds for settlement
            
        Returns:
            Position close result
        """
        data = {}
        if final_odds is not None:
            data['final_odds'] = final_odds
            
        return self._request('POST', f'/api/trading/positions/{position_id}/close', json=data)
    
    def execute_trades(self, 
                      session_id: str,
                      dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute trading recommendations.
        
        Args:
            session_id: Paper trading session ID
            dry_run: If True, only generate recommendations without executing
            
        Returns:
            Trading execution result with recommendations
        """
        return self._request('POST', '/api/trading/execute', json={
            'session_id': session_id,
            'dry_run': dry_run
        })
    
    def get_sessions(self) -> Dict[str, List[Dict]]:
        """Get all paper trading sessions."""
        return self._request('GET', '/api/trading/sessions')
    
    # Performance endpoints
    
    def get_performance(self, 
                       session_id: Optional[str] = None,
                       days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            session_id: Optional session ID filter
            days: Number of days to analyze
            
        Returns:
            Comprehensive performance metrics
        """
        params = {'days': days}
        if session_id:
            params['session_id'] = session_id
            
        return self._request('GET', '/api/performance', params=params)
    
    # Activity endpoints
    
    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get recent trading activity.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Recent bets, signals, and market activity
        """
        return self._request('GET', '/api/trading/recent', params={'hours': hours})
    
    def get_logs(self, limit: int = 100) -> Dict[str, List[Dict]]:
        """
        Get trading system logs.
        
        Args:
            limit: Maximum number of log entries
            
        Returns:
            Recent log entries
        """
        return self._request('GET', '/api/trading/logs', params={'limit': limit})
    
    # Market endpoints
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get market evaluation statistics."""
        return self._request('GET', '/api/trading/evaluation-stats')
    
    # Dashboard endpoints
    
    def get_unified_dashboard(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get unified dashboard data.
        
        Args:
            session_id: Optional session ID filter
            
        Returns:
            Complete dashboard data including portfolio, positions, and metrics
        """
        params = {}
        if session_id:
            params['session_id'] = session_id
            
        return self._request('GET', '/api/dashboard/unified', params=params)
    
    # Utility methods
    
    def health_check(self) -> bool:
        """
        Perform a health check on the API.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            status = self.get_system_status()
            return status.get('status') == 'healthy'
        except:
            return False
    
    def wait_for_ready(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """
        Wait for the API to be ready.
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Delay between attempts in seconds
            
        Returns:
            True if API becomes ready, False if timeout
        """
        import time
        
        for i in range(max_attempts):
            if self.health_check():
                return True
            if i < max_attempts - 1:
                time.sleep(delay)
        
        return False


class AsyncOminariAPIClient:
    """
    Async client for the Ominari API using aiohttp.
    
    Example:
        async with AsyncOminariAPIClient(api_key="your-key") as client:
            status = await client.get_system_status()
    """
    
    def __init__(self, base_url: str = "http://localhost:8888", 
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('OMINARI_API_KEY')
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Union[Dict, List]:
        """Make an async request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    # Implement async versions of all methods here...
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return await self._request('GET', '/api/status')


def demo_client_usage():
    """Demonstrate client usage."""
    # Get API key from environment or use development key
    api_key = os.getenv('OMINARI_API_KEY')
    if not api_key:
        print("Warning: No OMINARI_API_KEY found in environment")
        print("Using unauthenticated access (limited endpoints)\n")
    
    client = OminariAPIClient(api_key=api_key)
    
    print("=== Ominari API Client Demo ===\n")
    
    # Check authentication status
    if client.api_key:
        print(f"✅ Authenticated with API key: {client.api_key[:10]}...")
    else:
        print("⚠️  No authentication - some endpoints may be restricted")
    print()
    
    # Check system status
    print("1. System Status:")
    try:
        status = client.get_system_status()
        print(f"   Status: {status.get('status')}")
        print(f"   Database: {'Connected' if status.get('database_connected') else 'Disconnected'}")
    except OminariAPIError as e:
        print(f"   Error: {e}")
    
    # Get strategy config
    print("\n2. Strategy Configuration:")
    try:
        strategy = client.get_strategy_config()
        print(f"   Kelly Fraction: {strategy.get('kelly_fraction')}")
        print(f"   Bankroll: ${strategy.get('bankroll')}")
    except OminariAPIError as e:
        print(f"   Error: {e}")
    
    # Get performance metrics
    print("\n3. Performance Metrics:")
    try:
        perf = client.get_performance(days=7)
        print(f"   Total P&L: ${perf.get('total_pnl', 0):.2f}")
        print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    except OminariAPIError as e:
        print(f"   Error: {e}")
    
    # Get recent activity
    print("\n4. Recent Activity:")
    try:
        activity = client.get_recent_activity(hours=1)
        print(f"   Recent Bets: {len(activity.get('recent_bets', []))}")
        print(f"   Active Markets: {activity.get('active_markets', 0)}")
    except OminariAPIError as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    demo_client_usage()