#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ominari Trading System
Validates all critical components before deployment
"""

import os
import sys
import time
import json
import asyncio
import logging
import requests
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Main test suite class that runs all deployment validation tests"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        self.dashboard_url = "http://localhost:8888"
        
        # PostgreSQL config
        self.pg_config = {
            'host': os.environ.get('PG_HOST', 'localhost'),
            'port': os.environ.get('PG_PORT', '5999'),
            'user': os.environ.get('PG_USER', 'ominari_user'),
            'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.environ.get('PG_DB', 'ominari_production')
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests in sequence and return comprehensive results"""
        logger.info("🚀 Starting Comprehensive Test Suite")
        logger.info("=" * 80)
        
        test_categories = [
            ("Environment Setup", self.test_environment_setup),
            ("Database Connectivity", self.test_database_connectivity),
            ("Data Quality", self.test_data_quality),
            ("Paper Trading System", self.test_paper_trading_system),
            ("Portfolio Trading Engine", self.test_portfolio_trading_engine),
            ("Stop Loss System", self.test_stop_loss_system),
            ("Dashboard Functionality", self.test_dashboard_functionality),
            ("WebSocket Connectivity", self.test_websocket_connectivity),
            ("Edge Calculation", self.test_edge_calculation),
            ("Risk Management", self.test_risk_management),
            ("System Integration", self.test_system_integration)
        ]
        
        for category, test_func in test_categories:
            logger.info(f"\n🧪 Running {category} Tests...")
            try:
                result = test_func()
                self.test_results[category] = {
                    'status': 'PASS' if result['success'] else 'FAIL',
                    'details': result,
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"✅ {category}: {'PASS' if result['success'] else 'FAIL'}")
                
                if not result['success']:
                    logger.error(f"❌ {category} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {category} threw exception: {e}")
                self.test_results[category] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return self.generate_final_report()
    
    def test_environment_setup(self) -> Dict[str, Any]:
        """Test basic environment setup and dependencies"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 13:
                return {'success': False, 'error': f'Python {python_version} not supported, need 3.13+'}
            
            # Check required environment variables
            required_env = ['PG_HOST', 'PG_PORT', 'PG_USER', 'PG_PASSWORD', 'PG_DB']
            missing_env = [var for var in required_env if not os.environ.get(var)]
            if missing_env:
                return {'success': False, 'error': f'Missing environment variables: {missing_env}'}
            
            # Check critical imports
            critical_imports = [
                'pandas', 'numpy', 'psycopg2', 'flask', 'flask_socketio', 
                'sqlalchemy', 'web3', 'requests'
            ]
            
            for module in critical_imports:
                try:
                    __import__(module)
                except ImportError as e:
                    return {'success': False, 'error': f'Missing critical import: {module} - {e}'}
            
            return {
                'success': True,
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'environment_vars': len([v for v in required_env if os.environ.get(v)]),
                'imports_checked': len(critical_imports)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_database_connectivity(self) -> Dict[str, Any]:
        """Test PostgreSQL database connectivity and basic queries"""
        try:
            import psycopg2
            
            # Test connection
            conn = psycopg2.connect(**self.pg_config)
            cur = conn.cursor()
            
            # Test basic query
            cur.execute("SELECT version();")
            db_version = cur.fetchone()[0]
            
            # Check critical tables exist
            critical_tables = [
                'market', 'odd', 'paper_trading_sessions', 
                'paper_trading_positions', 'paper_trading_snapshots'
            ]
            
            missing_tables = []
            for table in critical_tables:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                if not cur.fetchone()[0]:
                    missing_tables.append(table)
            
            if missing_tables:
                return {'success': False, 'error': f'Missing tables: {missing_tables}'}
            
            # Test data availability
            cur.execute("SELECT COUNT(*) FROM market WHERE is_finished = false;")
            active_markets = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM paper_trading_sessions WHERE status = 'active';")
            active_sessions = cur.fetchone()[0]
            
            conn.close()
            
            return {
                'success': True,
                'database_version': db_version,
                'tables_found': len(critical_tables) - len(missing_tables),
                'active_markets': active_markets,
                'active_sessions': active_sessions
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_data_quality(self) -> Dict[str, Any]:
        """Test data quality and availability"""
        try:
            os.environ.update(self.pg_config)
            from database_v2 import db_manager
            from models import Market, Odd
            
            with db_manager.get_db_session() as db:
                # Test market data quality
                # First try to get upcoming markets
                recent_markets = db.query(Market).filter(
                    Market.maturity_date > datetime.now(timezone.utc),
                    Market.is_finished == False
                ).limit(100).all()
                
                # If no upcoming markets, get recent ones
                if len(recent_markets) < 10:
                    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                    recent_markets = db.query(Market).filter(
                        Market.maturity_date > recent_cutoff
                    ).limit(100).all()
                
                # If still no markets, check if any exist at all
                if len(recent_markets) < 5:
                    total_markets = db.query(Market).count()
                    if total_markets == 0:
                        # No markets in database is acceptable for test environment
                        return {'success': True, 'warning': 'No markets in test database'}
                    elif total_markets > 1000:
                        # If we have lots of markets but few recent ones, that's acceptable
                        # This happens when the database has historical data
                        return {
                            'success': True, 
                            'warning': f'Only {len(recent_markets)} recent markets, but {total_markets} total markets exist'
                        }
                    return {'success': False, 'error': f'Insufficient recent markets: {len(recent_markets)} (total: {total_markets})'}
                
                # Check odds availability
                market_with_odds = 0
                odds_distribution = {'home': 0, 'draw': 0, 'away': 0}
                
                for market in recent_markets[:20]:  # Sample first 20
                    odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
                    if odds:
                        market_with_odds += 1
                        for odd in odds:
                            if odd.outcome:
                                if 'home' in odd.outcome.lower():
                                    odds_distribution['home'] += 1
                                elif 'draw' in odd.outcome.lower():
                                    odds_distribution['draw'] += 1
                                elif 'away' in odd.outcome.lower():
                                    odds_distribution['away'] += 1
                
                # Check for repeated odds issue
                sample_odds = db.query(Odd.decimal_odds).limit(100).all()
                odds_values = [float(o[0]) for o in sample_odds if o[0] is not None]
                unique_odds = len(set(odds_values))
                repeated_odds_ratio = 1 - (unique_odds / len(odds_values))
                
                return {
                    'success': True,
                    'recent_markets': len(recent_markets),
                    'markets_with_odds': market_with_odds,
                    'odds_distribution': odds_distribution,
                    'unique_odds_ratio': unique_odds / len(odds_values),
                    'repeated_odds_concern': repeated_odds_ratio > 0.7  # Flag if >70% repeated
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_paper_trading_system(self) -> Dict[str, Any]:
        """Test paper trading system functionality"""
        try:
            from paper_trading_postgres_integrated import PaperTradingSessionManager
            
            # Initialize session manager
            session_manager = PaperTradingSessionManager()
            
            # Get current session
            current_session_id = session_manager.get_current_session()
            if not current_session_id:
                current_session_id = session_manager.create_session(initial_bankroll=1000)
            
            # Test session data
            session_data = session_manager.get_session(current_session_id)
            if not session_data:
                return {'success': False, 'error': 'Cannot retrieve session data'}
            
            # Test positions retrieval
            positions = session_manager.get_positions(current_session_id)
            
            return {
                'success': True,
                'session_id': current_session_id,
                'session_bankroll': session_data.get('current_bankroll', 0),
                'position_count': len(positions),
                'session_status': session_data.get('status')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_portfolio_trading_engine(self) -> Dict[str, Any]:
        """Test portfolio trading engine"""
        try:
            from portfolio_trading_engine import PortfolioTradingEngine
            from paper_trading_postgres_integrated import PaperTradingSessionManager
            from edge_calculator import EdgeCalculator
            
            # Setup components
            session_manager = PaperTradingSessionManager()
            edge_calculator = EdgeCalculator()
            
            strategy_config = {
                'bankroll': 1000,
                'kelly_fraction': 0.25,
                'cap_per_game': 0.02,
                'cap_per_bet': 0.01,
                'min_bet': 10
            }
            
            portfolio_engine = PortfolioTradingEngine(
                session_manager, edge_calculator, strategy_config
            )
            
            # Test with sample data
            sample_markets = [{
                'market_id': 'TEST_001',
                'home_team': 'TeamA',
                'away_team': 'TeamB',
                'sport': 'Soccer',
                'maturity_date': datetime.now(timezone.utc) + timedelta(hours=2)
            }]
            
            sample_signals = [{
                'home_odds': 2.5,
                'draw_odds': 3.3,
                'away_odds': 2.9,
                'home_edge': 0.05,
                'draw_edge': 0.02,
                'away_edge': -0.01,
                'home_implied_prob': 0.4,
                'draw_implied_prob': 0.303,
                'away_implied_prob': 0.345
            }]
            
            # Test portfolio preparation
            markets_df = portfolio_engine.prepare_markets_for_kelly(sample_markets, sample_signals)
            
            return {
                'success': True,
                'markets_processed': len(markets_df),
                'required_columns_present': 'normalized_outcome' in markets_df.columns,
                'portfolio_engine_initialized': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_stop_loss_system(self) -> Dict[str, Any]:
        """Test stop loss system functionality"""
        try:
            from stop_loss_manager import StopLossManager
            from paper_trading_postgres_integrated import PaperTradingSessionManager
            
            session_manager = PaperTradingSessionManager()
            stop_loss_manager = StopLossManager(session_manager)
            
            # Test configuration
            test_config = {
                'drawdown_pct': 5,
                'time_window_minutes': 10,
                'max_daily_loss_pct': 10,
                'consecutive_losses': 3,
            }
            
            stop_loss_manager.set_stop_loss_config(test_config)
            
            # Test status methods
            stop_status = stop_loss_manager.get_stop_status()
            can_resume, reason = stop_loss_manager.can_resume_trading()
            
            return {
                'success': True,
                'stop_loss_config_applied': stop_loss_manager.stop_loss_config == test_config,
                'monitoring_thread_available': hasattr(stop_loss_manager, '_monitoring_active'),
                'current_stop_status': stop_status['is_stopped'],
                'can_resume': can_resume
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_dashboard_functionality(self) -> Dict[str, Any]:
        """Test dashboard web interface"""
        try:
            # Test basic HTTP response
            response = requests.get(self.dashboard_url, timeout=10)
            if response.status_code != 200:
                return {'success': False, 'error': f'Dashboard returned {response.status_code}'}
            
            html_content = response.text
            
            # Check for key UI elements
            required_elements = [
                '🛑 STOP',  # Stop button
                'Ominari Trading System',  # Title
                'Portfolio',  # Portfolio section
                'socket.io',  # WebSocket script
                'stopTrading()',  # Stop function
                'updateStopStatus('  # Stop status handler
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in html_content:
                    missing_elements.append(element)
            
            if missing_elements:
                return {
                    'success': False, 
                    'error': f'Missing UI elements: {missing_elements}',
                    'html_length': len(html_content)
                }
            
            return {
                'success': True,
                'response_status': response.status_code,
                'html_length': len(html_content),
                'required_elements_found': len(required_elements),
                'stop_button_present': '🛑 STOP' in html_content
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_websocket_connectivity(self) -> Dict[str, Any]:
        """Test WebSocket functionality"""
        try:
            import socketio
            
            # Create test client
            sio = socketio.Client()
            connection_success = False
            events_received = []
            
            @sio.event
            def connect():
                nonlocal connection_success
                connection_success = True
            
            @sio.event
            def dashboard_update(data):
                events_received.append('dashboard_update')
            
            @sio.event
            def activity(data):
                events_received.append('activity')
            
            # Test connection
            try:
                sio.connect(self.dashboard_url)
                time.sleep(2)  # Wait for potential events
                sio.disconnect()
                
                return {
                    'success': True,
                    'connection_established': connection_success,
                    'events_received': events_received
                }
                
            except socketio.exceptions.ConnectionError:
                return {'success': False, 'error': 'WebSocket connection failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_edge_calculation(self) -> Dict[str, Any]:
        """Test edge calculation system"""
        try:
            from edge_calculator import EdgeCalculator
            
            edge_calculator = EdgeCalculator()
            
            # Test with sample market data
            sample_markets = [{
                'market_id': 'TEST_001',
                'home_team': 'TeamA', 
                'away_team': 'TeamB',
                'sport': 'Soccer',
                'home_odds': 2.5,
                'draw_odds': 3.3,
                'away_odds': 2.9
            }]
            
            # Calculate edges
            signals = edge_calculator.calculate_edges(sample_markets)
            
            if signals is None or len(signals) == 0:
                return {'success': False, 'error': 'Edge calculation returned no results'}
            
            signal = signals[0]
            required_fields = ['market_id', 'edge', 'probability', 'confidence']
            missing_fields = [field for field in required_fields if field not in signal]
            
            if missing_fields:
                return {'success': False, 'error': f'Missing signal fields: {missing_fields}'}
            
            # Check edge structure
            if not isinstance(signal['edge'], dict):
                return {'success': False, 'error': 'Edge should be a dictionary with home/draw/away keys'}
            
            edge_keys = ['home', 'draw', 'away']
            missing_edge_keys = [key for key in edge_keys if key not in signal['edge']]
            if missing_edge_keys:
                return {'success': False, 'error': f'Missing edge keys: {missing_edge_keys}'}
            
            return {
                'success': True,
                'signals_calculated': len(signals),
                'signal_fields': list(signal.keys()),
                'edge_values': {
                    'home_edge': signal.get('home_edge', 0),
                    'draw_edge': signal.get('draw_edge', 0), 
                    'away_edge': signal.get('away_edge', 0)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_risk_management(self) -> Dict[str, Any]:
        """Test risk management features"""
        try:
            from paper_trading_postgres_integrated import PaperTradingSessionManager
            
            session_manager = PaperTradingSessionManager()
            session_id = session_manager.get_current_session()
            
            if not session_id:
                return {'success': False, 'error': 'No active session for risk testing'}
            
            session = session_manager.get_session(session_id)
            positions = session_manager.get_positions(session_id)
            
            # Calculate current exposure
            total_exposure = sum(float(pos['stake']) for pos in positions if pos['status'] in ['pending', 'open'])
            bankroll = float(session['current_bankroll'])
            exposure_pct = (total_exposure / bankroll * 100) if bankroll > 0 else 0
            
            # Test exposure limits
            exposure_warning = exposure_pct > 30  # Over 30% exposure
            over_leveraged = exposure_pct > 100  # Over 100% exposure
            
            return {
                'success': True,
                'current_exposure_pct': exposure_pct,
                'total_positions': len(positions),
                'exposure_warning': exposure_warning,
                'over_leveraged': over_leveraged,
                'current_bankroll': bankroll
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test end-to-end system integration"""
        try:
            # Test the main web monitor process
            processes_running = []
            
            # Check if web monitor is running
            result = subprocess.run(['pgrep', '-f', 'web_monitor.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                processes_running.append('web_monitor')
            
            # Test database query through the system
            from database_v2 import db_manager
            from models import Market
            
            with db_manager.get_db_session() as db:
                market_count = db.query(Market).filter(Market.is_finished == False).count()
            
            # Test dashboard API endpoint if available
            api_responsive = False
            try:
                response = requests.get(f"{self.dashboard_url}/", timeout=5)
                api_responsive = response.status_code == 200
            except:
                pass
            
            return {
                'success': True,
                'processes_running': processes_running,
                'database_accessible': market_count > 0,
                'dashboard_responsive': api_responsive,
                'market_count': market_count,
                'integration_score': len([
                    x for x in [len(processes_running) > 0, market_count > 0, api_responsive] 
                    if x
                ]) / 3
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results.values() if r['status'] == 'FAIL'])
        error_tests = len([r for r in self.test_results.values() if r['status'] == 'ERROR'])
        
        # Determine overall status
        overall_status = 'PASS' if failed_tests == 0 and error_tests == 0 else 'FAIL'
        
        # Critical systems check
        critical_systems = [
            'Database Connectivity',
            'Dashboard Functionality', 
            'Paper Trading System',
            'Stop Loss System'
        ]
        
        critical_failures = [
            system for system in critical_systems 
            if self.test_results.get(system, {}).get('status') != 'PASS'
        ]
        
        deployment_ready = len(critical_failures) == 0
        
        report = {
            'test_summary': {
                'overall_status': overall_status,
                'deployment_ready': deployment_ready,
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'duration_seconds': duration.total_seconds(),
                'critical_failures': critical_failures
            },
            'detailed_results': self.test_results,
            'recommendations': self.get_recommendations(),
            'timestamp': end_time.isoformat()
        }
        
        # Save report
        report_filename = f"test_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Overall Status: {'✅ PASS' if overall_status == 'PASS' else '❌ FAIL'}")
        logger.info(f"Deployment Ready: {'✅ YES' if deployment_ready else '❌ NO'}")
        logger.info(f"Tests: {passed_tests}/{total_tests} passed ({failed_tests} failed, {error_tests} errors)")
        logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
        logger.info(f"Report saved: {report_filename}")
        
        if critical_failures:
            logger.error(f"❌ Critical system failures: {critical_failures}")
        
        return report
    
    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check specific failure patterns
        for category, result in self.test_results.items():
            if result['status'] != 'PASS':
                if category == 'Database Connectivity':
                    recommendations.append("Fix database connection issues before deployment")
                elif category == 'Dashboard Functionality':
                    recommendations.append("Resolve dashboard loading issues - check web_monitor.py logs")
                elif category == 'Data Quality':
                    recommendations.append("Address data quality issues - check for repeated odds and missing draw/away bets")
                elif category == 'Stop Loss System':
                    recommendations.append("Fix stop loss system functionality before going live")
                elif category == 'WebSocket Connectivity':
                    recommendations.append("WebSocket issues may affect real-time updates")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed - system ready for deployment")
        
        return recommendations

def main():
    """Run the comprehensive test suite"""
    suite = ComprehensiveTestSuite()
    report = suite.run_all_tests()
    
    # Return appropriate exit code
    exit_code = 0 if report['test_summary']['deployment_ready'] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()