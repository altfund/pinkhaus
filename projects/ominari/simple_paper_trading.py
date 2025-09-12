#!/usr/bin/env python3
"""Paper trading execution using the core economic model from evaluate_open_markets."""

from datetime import datetime, timezone, timedelta
import pandas as pd
import logging
from paper_trading_sessions import PaperTradingSessionManager
from paper_trading_quotes import QuoteRecorder
from overtime_quote_service import OvertimeQuoteService
from fee_calculator import FeeCalculator
from database_v2 import db_manager
from models import Market
from evaluate_open_markets import (
    generate_betting_session_report_and_save,
    get_upcoming_overtime_markets_with_signals,
    summarize_match_schedule_from_open_markets,
    find_upcoming_game_breaks,
    extract_active_game_periods_from_breaks
)
from signals import SIGNAL_PROVIDERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def settle_finished_positions(session_manager, session_id):
    """Check and settle any finished positions."""
    try:
        current_session = session_manager.sessions["sessions"].get(session_id)
        if not current_session:
            return
        
        # Get all open positions
        positions = current_session.get('positions', {})
        if not positions:
            return
        
        # Collect market IDs
        market_ids = set()
        for pos in positions.values():
            if pos.get('market_id'):
                market_ids.add(pos.get('market_id'))
        
        # Query database for finished markets
        market_results = {}
        with db_manager.get_db_session() as db:
            for market_id in market_ids:
                market = db.query(Market).filter(Market.source_id == market_id).first()
                if market and market.is_finished and market.resolved_outcome:
                    market_results[market_id] = {
                        'is_finished': True,
                        'resolved_outcome': market.resolved_outcome,
                        'home_score': market.home_score,
                        'away_score': market.away_score
                    }
        
        if market_results:
            settled_count = session_manager.settle_finished_markets(session_id, market_results)
            if settled_count > 0:
                logger.info(f"Settled {settled_count} finished positions")
    except Exception as e:
        logger.error(f"Error settling finished positions: {e}")

def execute_simple_paper_trades(use_sessions=True, use_execution_quotes=False):
    """Execute paper trades using the core economic model."""
    logger.info("Fetching markets using core economic model")
    
    # Initialize session manager and quote recorder if enabled
    session_manager = None
    quote_recorder = None
    quote_service = None
    current_session = None
    fee_calculator = FeeCalculator()
    
    if use_sessions:
        session_manager = PaperTradingSessionManager()
        quote_recorder = QuoteRecorder()
        current_session = session_manager.get_current_session()
        logger.info(f"Using session: {current_session['session_id']}")
        
        # Settle any finished positions first
        settle_finished_positions(session_manager, current_session['session_id'])
    
    if use_execution_quotes:
        try:
            quote_service = OvertimeQuoteService()
            logger.info("Initialized Overtime quote service")
        except ValueError as e:
            logger.warning(f"Could not initialize quote service: {e}")
            use_execution_quotes = False
    
    # Get current time and calculate trading window
    now_utc = datetime.now(timezone.utc)
    
    # Configuration matching web_monitor
    min_break_minutes = 240  # 4 hour minimum break
    avg_game_duration_minutes = 180  # 3 hour games
    in_play_buffer_minutes = 15  # Don't bet on matches starting soon
    
    # Use the core economic model to generate recommendations
    signal_providers = SIGNAL_PROVIDERS
    signal_weights = {
        'implied_probability': 1.0,
        'coin_flip': 0.0
    }
    
    # Generate betting recommendations using core model
    # Set Kelly parameters matching current paper trading setup
    result = generate_betting_session_report_and_save(
        kelly_bankroll=1.0,
        execution_bankroll=current_session['current_bankroll'] if current_session else 10000.0,
        kelly_fraction=0.25,  # Conservative 25% Kelly
        cap_per_game=0.25,
        cap_per_bet=0.25,
        cap_per_game_market=0.10,
        min_bet_abs=10.0,
        min_bet_pct=0.001,
        abs_game_limit=None,  # No game limit
        as_of=now_utc,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
        base_dir="paper_trading_reports",
        display_md=False,  # Don't display markdown
        save_as_latest=False,  # Don't save as latest
        signal_providers=signal_providers,
        signal_weights=signal_weights,
        mode="paper_trading"
    )
    
    # Extract recommendations
    trimmed = result.get('trimmed', pd.DataFrame())
    
    if trimmed.empty:
        logger.info("No tradeable markets found by economic model")
        return {"success": True, "message": "No tradeable markets found"}
    
    # Filter to non-zero stakes
    to_bet = trimmed[trimmed['stake'] > 0]
    logger.info(f"Economic model recommends {len(to_bet)} positions")
    
    # Get current positions if using sessions
    current_positions = {}
    if use_sessions and current_session:
        for pos_key, pos in current_session.get('positions', {}).items():
            # Extract market_id and outcome from position key
            parts = pos_key.split('_')
            if len(parts) >= 2:
                market_id = parts[0]
                outcome = parts[1]
                current_positions[(market_id, outcome)] = pos.get('total_stake', 0)
    
    # Calculate rebalancing trades
    logger.info("Calculating rebalancing trades...")
    trade_details = []
    
    for _, bet in to_bet.iterrows():
        # Check current position
        position_key = (bet['source_id'], bet['normalized_outcome'])
        current_stake = current_positions.get(position_key, 0)
        target_stake = bet['stake']
        
        # Calculate delta (rebalancing amount)
        delta_stake = target_stake - current_stake
        
        # Skip if delta is too small (less than $10 or 0.1% of bankroll)
        execution_bankroll = current_session['current_bankroll'] if current_session else 10000.0
        if abs(delta_stake) < max(10.0, execution_bankroll * 0.001):
            logger.info(f"Skipping {bet['market_name'][:30]} - {bet['normalized_outcome']}: delta ${delta_stake:.2f} too small")
            continue
        
        # Get execution odds (from economic model, already adjusted for fees)
        execution_odds = bet['odds']  # These are already adjusted odds
        
        logger.info(f"  {bet['market_name']} - {bet['normalized_outcome']} @ {execution_odds:.2f} - Current: ${current_stake:.2f}, Target: ${target_stake:.2f}, Delta: ${delta_stake:.2f}")
        
        # Calculate fees for this trade (only on positive delta)
        fee_info = {}
        if delta_stake > 0:  # Opening/increasing position
            # The economic model already accounted for fees in odds adjustment
            # But we still need to track actual fee amounts for accounting
            raw_odds = execution_odds * 1.03  # Reverse engineer raw odds (assumes 3% total fee)
            fee_info = fee_calculator.calculate_entry_fees(abs(delta_stake), raw_odds)
            logger.info(f"    Fees: ${fee_info['fee_amount']:.2f} ({fee_info['total_fee_pct']*100:.1f}% - SafeBox: {fee_info['safebox_fee_pct']*100:.1f}%, Skew: {fee_info['skew_fee_pct']*100:.1f}%)")
        
        # Get maturity date from database
        maturity_date = None
        try:
            with db_manager.get_db_session() as db:
                market = db.query(Market).filter(Market.source_id == bet['source_id']).first()
                if market:
                    maturity_date = market.maturity_date
        except:
            pass
        
        trade_details.append({
            'market_id': bet['source_id'],
            'market_name': bet['market_name'],
            'outcome': bet['normalized_outcome'],
            'odds': raw_odds if delta_stake > 0 else bet['odds'],  # Use raw odds for position tracking
            'execution_odds': execution_odds,
            'stake': delta_stake,
            'target_stake': target_stake,
            'current_stake': current_stake,
            'edge': bet.get('edge', 0),
            'probability': bet.get('probability', 0),
            'maturity_date': maturity_date.isoformat() if maturity_date else None,
            'fee_info': fee_info
        })
    
    # Check for positions to close (not in economic model recommendations)
    if use_sessions and current_session:
        recommended_positions = set((bet['source_id'], bet['normalized_outcome']) for _, bet in trimmed.iterrows())
        
        for pos_key, pos in current_session.get('positions', {}).items():
            parts = pos_key.split('_')
            if len(parts) >= 2:
                position_tuple = (parts[0], parts[1])
                if position_tuple not in recommended_positions and pos.get('total_stake', 0) > 0:
                    # Position should be closed
                    logger.info(f"  CLOSE: {pos.get('market_name', 'Unknown')} - {parts[1]} - Current stake: ${pos.get('total_stake', 0):.2f}")
                    trade_details.append({
                        'market_id': parts[0],
                        'market_name': pos.get('market_name', 'Unknown'),
                        'outcome': parts[1],
                        'odds': pos.get('avg_odds', 0),
                        'stake': -pos.get('total_stake', 0),  # Negative to close
                        'target_stake': 0,
                        'current_stake': pos.get('total_stake', 0),
                        'edge': 0,
                        'probability': 0
                    })
    
    # Record quotes and update session if enabled
    if use_sessions and session_manager and quote_recorder:
        # Record quotes
        session_id = current_session['session_id']
        quote_recorder.record_trade_quotes(trade_details, session_id)
        
        # Update session with trades
        session_manager.record_trades(session_id, trade_details)
        
        # Get updated performance
        performance = session_manager.get_session_performance(session_id)
        logger.info(f"Session performance - Portfolio: ${performance['current_value']:.2f}, ROI: {performance['roi']:.2f}%")
    
    # Calculate net stake change
    net_stake_change = sum(trade['stake'] for trade in trade_details)
    
    # More informative message
    if len(trade_details) == 0 and len(to_bet) > 0:
        message = f"Portfolio already optimally positioned with {len(to_bet)} holdings. No rebalancing needed."
    elif len(trade_details) > 0:
        message = f"Rebalanced portfolio: {len(trade_details)} trades, net stake change: ${net_stake_change:.2f}"
    else:
        message = "No tradeable markets found in current evaluation"
    
    return {
        "success": True,
        "message": message,
        "total_stake": net_stake_change,
        "trades": len(trade_details),
        "trade_details": trade_details,
        "positions_held": len(to_bet),
        "session_id": current_session['session_id'] if current_session else None,
        "economic_model_result": result  # Include full economic model output
    }

if __name__ == "__main__":
    result = execute_simple_paper_trades()
    print(f"Result: {result['message']}")
    print(f"Trades executed: {result['trades']}")
    print(f"Net stake change: ${result['total_stake']:.2f}")