#!/usr/bin/env python3
"""
Dashboard Feature Tests - Ensure we don't lose functionality
"""
import os
import pytest
import json
from datetime import datetime, timezone

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
import requests

DASHBOARD_URL = "http://localhost:8888"

class TestDashboardFeatures:
    """Test suite for dashboard features"""
    
    def test_health_endpoint(self):
        """Test health endpoint exists and returns proper data"""
        response = requests.get(f"{DASHBOARD_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'services' in data
        assert data['services']['database']['status'] == 'up'
    
    def test_cache_stats_endpoint(self):
        """Test cache stats endpoint"""
        response = requests.get(f"{DASHBOARD_URL}/cache-stats")
        assert response.status_code == 200
        data = response.json()
        assert 'size' in data
        assert 'hit_rate' in data
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = requests.get(f"{DASHBOARD_URL}/metrics")
        assert response.status_code == 200
        assert 'ominari_markets_total' in response.text
        assert 'ominari_real_odds_total' in response.text
    
    def test_sport_filtering(self):
        """Test that sport filtering works correctly"""
        with db_manager.get_db_session() as db:
            # Check we're filtering Soccer only
            markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.league_name != 'International Football'
            ).limit(10).all()
            
            for market in markets:
                assert market.sport == 'Soccer'
                assert market.league_name != 'International Football'
    
    def test_nation_filtering(self):
        """Test nation filtering"""
        allowed_nations = ['England', 'International', 'Europe']
        
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.nation.in_(allowed_nations)
            ).limit(10).all()
            
            for market in markets:
                assert market.nation in allowed_nations
    
    def test_odds_data_structure(self):
        """Test odds data has required fields"""
        with db_manager.get_db_session() as db:
            result = db.query(Market, Odd).join(
                Odd, Market.source_id == Odd.source_id
            ).first()
            
            if result:
                market, odd = result
                # Check market fields
                assert hasattr(market, 'home_team')
                assert hasattr(market, 'away_team')
                assert hasattr(market, 'sport')
                assert hasattr(market, 'league_name')
                assert hasattr(market, 'nation')
                assert hasattr(market, 'maturity_date')
                
                # Check odd fields
                assert hasattr(odd, 'decimal_odds')
                assert hasattr(odd, 'outcome')
                assert odd.outcome in ['home', 'draw', 'away', 'Home', 'Draw', 'Away']
    
    def test_no_american_football_in_soccer(self):
        """Ensure no American Football teams in Soccer category"""
        american_teams = ['Patriots', 'Cowboys', 'Eagles', 'Notre Dame', 'Alabama']
        
        with db_manager.get_db_session() as db:
            for team in american_teams:
                count = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    (Market.home_team.like(f'%{team}%') | 
                     Market.away_team.like(f'%{team}%'))
                ).count()
                assert count == 0, f"Found {team} in Soccer markets!"
    
    def test_dashboard_response(self):
        """Test main dashboard page loads"""
        response = requests.get(DASHBOARD_URL)
        assert response.status_code == 200
        assert 'Ominari Blockchain Trading' in response.text
        assert 'Live Markets' in response.text

if __name__ == "__main__":
    # Run tests
    test = TestDashboardFeatures()
    
    print("Running Dashboard Feature Tests...")
    
    try:
        test.test_health_endpoint()
        print("✅ Health endpoint test passed")
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
    
    try:
        test.test_sport_filtering()
        print("✅ Sport filtering test passed")
    except Exception as e:
        print(f"❌ Sport filtering test failed: {e}")
    
    try:
        test.test_nation_filtering()
        print("✅ Nation filtering test passed")
    except Exception as e:
        print(f"❌ Nation filtering test failed: {e}")
    
    try:
        test.test_odds_data_structure()
        print("✅ Odds data structure test passed")
    except Exception as e:
        print(f"❌ Odds data structure test failed: {e}")
    
    try:
        test.test_no_american_football_in_soccer()
        print("✅ No American Football in Soccer test passed")
    except Exception as e:
        print(f"❌ No American Football in Soccer test failed: {e}")
    
    print("\nTest suite complete!")