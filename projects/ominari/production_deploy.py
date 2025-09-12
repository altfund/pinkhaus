#!/usr/bin/env python3
"""
Production Deployment Script for Ominari Trading System.
Integrates all components with comprehensive risk management.
"""

import os
import sys
import logging
import json
import signal
from datetime import datetime, timezone
from typing import Optional, Dict
import argparse

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ominari_production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import all components
from production_risk_config import ProductionRiskConfig, RiskLevel, get_preset_config
from risk_manager import RiskManager
from signal_registry import SignalRegistry, DynamicWeightManager
from integrate_signal_registry import initialize_signal_registry, update_signal_weights
from update_evaluate_with_registry import patch_evaluate_open_markets
from paper_trading_engine import PaperTradingEngine
from paper_trading_sessions import SessionManager
from database_v2 import db_manager


class ProductionTradingSystem:
    """Main production trading system with all components integrated."""
    
    def __init__(self, 
                 risk_config: ProductionRiskConfig,
                 initial_bankroll: float = 1000.0,
                 session_name: Optional[str] = None):
        
        self.risk_config = risk_config
        self.initial_bankroll = initial_bankroll
        self.session_name = session_name or f"prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.session_manager = SessionManager()
        self.session = None
        self.paper_trader = None
        self.risk_manager = None
        self.signal_registry = None
        self.weight_manager = None
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing production trading system...")
        
        # Validate risk configuration
        errors = self.risk_config.validate()
        if errors:
            raise ValueError(f"Invalid risk configuration: {errors}")
        
        # Create trading session
        self.session = self.session_manager.get_or_create_session(
            session_name=self.session_name,
            initial_bankroll=self.initial_bankroll
        )
        
        logger.info(f"Created session: {self.session['session_id']}")
        
        # Initialize paper trading
        self.paper_trader = PaperTradingEngine(
            db_manager=db_manager,
            session_id=self.session['session_id']
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            config=self.risk_config,
            session_id=self.session['session_id']
        )
        
        # Initialize signal registry
        patch_evaluate_open_markets()  # Patch for registry support
        self.signal_registry, self.weight_manager = initialize_signal_registry()
        
        # Log initialization
        logger.info(f"System initialized with:")
        logger.info(f"  Risk level: {self.risk_config.risk_level.value}")
        logger.info(f"  Initial bankroll: ${self.initial_bankroll:,.2f}")
        logger.info(f"  Active signals: {self.signal_registry.list_signals()}")
        logger.info(f"  Signal weights: {self.weight_manager.get_current_weights()}")
        
        self.is_running = True
        
    def run_trading_cycle(self):
        """Run one complete trading cycle."""
        try:
            logger.info("Starting trading cycle...")
            
            # 1. Check portfolio risk status
            portfolio_status = self.risk_manager.check_portfolio_risk(
                self.session['bankroll']
            )
            
            if not portfolio_status['healthy']:
                logger.warning("Portfolio unhealthy, skipping trading cycle")
                for alert in portfolio_status['alerts']:
                    logger.error(f"ALERT: {alert}")
                return
            
            # 2. Update signal weights based on performance
            if self.risk_config.signal_limits.min_signals_required > 1:
                new_weights = update_signal_weights(
                    self.signal_registry,
                    self.weight_manager,
                    method="bayesian"
                )
                logger.info(f"Updated signal weights: {new_weights}")
            
            # 3. Generate betting recommendations
            from evaluate_open_markets import generate_betting_session_report_and_save
            
            result = generate_betting_session_report_and_save(
                kelly_bankroll=self.risk_config.kelly_limits.kelly_fraction,
                execution_bankroll=self.session['bankroll'],
                kelly_fraction=self.risk_config.kelly_limits.kelly_fraction,
                cap_per_game=self.risk_config.kelly_limits.kelly_cap_per_game,
                cap_per_bet=self.risk_config.kelly_limits.kelly_cap_per_bet,
                cap_per_game_market=self.risk_config.kelly_limits.kelly_cap_per_market,
                min_bet_abs=self.risk_config.position_limits.min_bet_abs,
                min_bet_pct=self.risk_config.position_limits.min_bet_pct,
                signal_providers=None,  # Use registry
                signal_weights=None,    # Use dynamic weights
                mode="production",
                strat=f"{self.risk_config.risk_level.value}_dynamic",
                display_md=False
            )
            
            if not result or 'trimmed' not in result:
                logger.info("No betting opportunities found")
                return
            
            # 4. Validate each bet against risk limits
            bets_to_place = result['trimmed'][result['trimmed']['stake'] > 0]
            approved_bets = []
            
            for _, bet_row in bets_to_place.iterrows():
                bet_proposal = {
                    'stake': bet_row['stake'],
                    'stake_pct': bet_row['stake_fraction'],
                    'odds': bet_row['odds'],
                    'edge': bet_row.get('edge', 0),
                    'event_time': datetime.now(timezone.utc),  # Would get from market data
                    'sport': 'Soccer'  # Would get from market data
                }
                
                is_valid, violations = self.risk_manager.check_pre_bet_limits(
                    bet_proposal,
                    self.session['bankroll']
                )
                
                if is_valid:
                    approved_bets.append(bet_row)
                else:
                    logger.warning(f"Bet rejected for {bet_row['market_name']}: {violations}")
            
            # 5. Execute approved bets
            logger.info(f"Placing {len(approved_bets)} approved bets...")
            
            for bet_row in approved_bets:
                try:
                    self.paper_trader.place_bet(
                        market_id=bet_row['source_id'],
                        outcome=bet_row['normalized_outcome'],
                        stake=bet_row['stake'],
                        odds=bet_row['odds'],
                        strategy_name=f"{self.risk_config.risk_level.value}_dynamic"
                    )
                    logger.info(f"Placed bet: {bet_row['market_name']} - ${bet_row['stake']:.2f}")
                except Exception as e:
                    logger.error(f"Failed to place bet: {e}")
            
            # 6. Generate risk report
            risk_report = self.risk_manager.get_risk_report(self.session['bankroll'])
            logger.info("\n" + risk_report)
            
            # Save report
            with open(f"risk_reports/{self.session_name}_risk.txt", 'w') as f:
                f.write(risk_report)
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down production system...")
        
        self.is_running = False
        
        # Close all positions if configured
        if hasattr(self, 'paper_trader') and self.paper_trader:
            # Could implement position closing logic here
            pass
        
        # Save final state
        if self.session:
            final_report = {
                'session': self.session,
                'final_bankroll': self.session['bankroll'],
                'risk_config': self.risk_config.to_json(),
                'shutdown_time': datetime.now(timezone.utc).isoformat()
            }
            
            with open(f"sessions/{self.session_name}_final.json", 'w') as f:
                json.dump(final_report, f, indent=2)
        
        logger.info("Shutdown complete")
    
    def health_check(self) -> Dict[str, any]:
        """Perform system health check."""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check database
        try:
            with db_manager.get_db_session() as db:
                db.execute("SELECT 1")
            health['components']['database'] = 'ok'
        except Exception as e:
            health['components']['database'] = f'error: {e}'
            health['status'] = 'unhealthy'
        
        # Check risk manager
        if self.risk_manager:
            try:
                metrics = self.risk_manager.calculate_current_metrics(
                    self.session['bankroll']
                )
                health['components']['risk_manager'] = 'ok'
                health['risk_metrics'] = metrics.to_dict()
            except Exception as e:
                health['components']['risk_manager'] = f'error: {e}'
                health['status'] = 'degraded'
        
        # Check signal registry
        if self.signal_registry:
            health['components']['signals'] = {
                'count': len(self.signal_registry.list_signals()),
                'weights': self.weight_manager.get_current_weights()
            }
        
        return health


def create_production_config(args) -> ProductionRiskConfig:
    """Create production configuration from arguments."""
    if args.risk_preset:
        config = get_preset_config(RiskLevel(args.risk_preset))
    else:
        config = ProductionRiskConfig()
    
    # Override with command line arguments
    if args.kelly_fraction:
        config.kelly_limits.kelly_fraction = args.kelly_fraction
    
    if args.max_exposure:
        config.portfolio_limits.max_total_exposure_pct = args.max_exposure
    
    if args.max_drawdown:
        config.portfolio_limits.max_drawdown_pct = args.max_drawdown
    
    return config


def main():
    """Main production entry point."""
    parser = argparse.ArgumentParser(description='Ominari Production Trading System')
    
    parser.add_argument('--risk-preset', 
                       choices=['conservative', 'moderate', 'aggressive'],
                       default='moderate',
                       help='Risk level preset')
    
    parser.add_argument('--bankroll', 
                       type=float, 
                       default=1000.0,
                       help='Initial bankroll')
    
    parser.add_argument('--kelly-fraction',
                       type=float,
                       help='Kelly fraction override')
    
    parser.add_argument('--max-exposure',
                       type=float,
                       help='Max exposure override')
    
    parser.add_argument('--max-drawdown',
                       type=float,
                       help='Max drawdown override')
    
    parser.add_argument('--session-name',
                       help='Session name')
    
    parser.add_argument('--dry-run',
                       action='store_true',
                       help='Run health check only')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('risk_reports', exist_ok=True)
    os.makedirs('sessions', exist_ok=True)
    
    # Create configuration
    config = create_production_config(args)
    
    # Save configuration
    config_path = f"sessions/config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, 'w') as f:
        f.write(config.to_json())
    logger.info(f"Saved configuration to {config_path}")
    
    # Create system
    system = ProductionTradingSystem(
        risk_config=config,
        initial_bankroll=args.bankroll,
        session_name=args.session_name
    )
    
    try:
        # Initialize
        system.initialize()
        
        # Health check
        health = system.health_check()
        logger.info(f"Health check: {json.dumps(health, indent=2)}")
        
        if args.dry_run:
            logger.info("Dry run complete")
            return
        
        # Run trading cycle
        system.run_trading_cycle()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()