#!/usr/bin/env python3
"""Simple paper trading execution using web monitor's data path."""

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import pandas as pd
import numpy as np
from signals import SIGNAL_PROVIDERS
from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
import logging
from paper_trading_quotes import QuoteRecorder
from paper_trading_sessions import PaperTradingSessionManager
from overtime_quote_service import OvertimeQuoteService
from fee_calculator import FeeCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_markets_for_paper_trading():
    """Get markets for the next tradeable game chunk."""
    markets = []
    
    with db_manager.get_db_session() as db:
        # Get active soccer markets to find chunks
        now_utc = datetime.now(timezone.utc)
        
        # First get all upcoming markets to calculate chunks
        all_active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > now_utc
        ).order_by(Market.maturity_date).all()
        
        if not all_active_markets:
            return pd.DataFrame()
        
        # Import chunk calculation functions
        from evaluate_open_markets import (
            summarize_match_schedule_from_open_markets,
            find_upcoming_game_breaks,
            extract_active_game_periods_from_breaks
        )
        
        # Convert to DataFrame for chunk calculation
        temp_data = []
        for market in all_active_markets:
            temp_data.append({
                'source_id': market.source_id,
                'maturity_date': market.maturity_date,
                'home_team': market.home_team,
                'away_team': market.away_team,
                'league_name': market.league_name
            })
        temp_df = pd.DataFrame(temp_data)
        
        # Calculate chunks
        match_df = summarize_match_schedule_from_open_markets(temp_df)
        if match_df.empty:
            return pd.DataFrame()
            
        # Use config values from web_monitor for consistency
        min_break_minutes = 240  # 4 hour minimum break (matching web_monitor)
        avg_game_duration_minutes = 180  # 3 hour games
        
        breaks_df = find_upcoming_game_breaks(
            match_df, 
            min_break_minutes=min_break_minutes,
            avg_game_duration_minutes=avg_game_duration_minutes,
            now=now_utc
        )
        game_chunks = extract_active_game_periods_from_breaks(
            match_df, breaks_df, 
            avg_game_duration_minutes=avg_game_duration_minutes
        )
        
        # Find the next tradeable chunk (not in-play)
        if game_chunks.empty:
            return pd.DataFrame()
        
        # Add buffer to avoid betting on matches about to start (in-play)
        in_play_buffer_minutes = 15  # Don't bet on matches starting in next 15 minutes
        now_plus_buffer = now_utc + pd.Timedelta(minutes=in_play_buffer_minutes)
        
        # Find first chunk where at least some games are tradeable
        selected_chunk = None
        for idx, chunk in game_chunks.iterrows():
            chunk_start = pd.to_datetime(chunk['chunk_start'], utc=True)
            chunk_end = pd.to_datetime(chunk['chunk_end'], utc=True)
            
            # Check if chunk has tradeable games (start time > buffer)
            if chunk_start >= now_plus_buffer:
                selected_chunk = chunk
                break
            # Or if chunk is active but has games beyond buffer
            elif chunk_start <= now_utc < chunk_end:
                # Check if there are games in this chunk beyond the buffer
                chunk_markets = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.is_finished == False,
                    Market.maturity_date >= now_plus_buffer,
                    Market.maturity_date >= chunk_start,
                    Market.maturity_date < chunk_end
                ).first()
                if chunk_markets:
                    selected_chunk = chunk
                    break
        
        if selected_chunk is None:
            logger.info(f"No tradeable chunks found. Next games start beyond current chunks.")
            return pd.DataFrame()
            
        chunk_start = pd.to_datetime(selected_chunk['chunk_start'], utc=True)
        chunk_end = pd.to_datetime(selected_chunk['chunk_end'], utc=True)
        
        # Log which window we're trading
        chunk_status = "upcoming" if chunk_start > now_utc else "active (partially tradeable)"
        logger.info(f"Trading {chunk_status} chunk: {chunk_start.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')} UTC ({selected_chunk['num_games']} games)")
        
        # Now get markets only in the first chunk
        # Add buffer to avoid betting on matches about to start (in-play)
        in_play_buffer_minutes = 15  # Don't bet on matches starting in next 15 minutes
        now_plus_buffer = now_utc + pd.Timedelta(minutes=in_play_buffer_minutes)
        
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date >= now_plus_buffer,  # Exclude matches about to start
            Market.maturity_date >= chunk_start,
            Market.maturity_date < chunk_end
        ).order_by(Market.maturity_date).all()
        
        logger.info(f"Excluding matches starting before {now_plus_buffer.strftime('%Y-%m-%d %H:%M')} UTC (in-play buffer)")
        
        for market in active_markets:
            try:
                # Get latest odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                for odd in odds:
                    if odd and odd.decimal_odds and odd.decimal_odds > 0:
                        market_data = {
                            'source_id': market.source_id,
                            'market_type': 'winner',
                            'home_team': market.home_team,
                            'away_team': market.away_team,
                            'league_name': market.league_name,
                            'maturity_date': market.maturity_date,
                            'outcome': odd.outcome,
                            'odds': float(odd.decimal_odds),
                            'bookmaker': 'overtime_markets'
                        }
                        
                        # Map outcomes
                        if odd.outcome == 'option_1':
                            market_data['normalized_outcome'] = 'Home'
                        elif odd.outcome == 'option_3':
                            market_data['normalized_outcome'] = 'Draw'
                        elif odd.outcome == 'option_2':
                            market_data['normalized_outcome'] = 'Away'
                        else:
                            continue
                            
                        markets.append(market_data)
                        
            except Exception as e:
                logger.warning(f"Error processing market {market.source_id}: {e}")
                continue
                
    return pd.DataFrame(markets)

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
    """Execute paper trades using next tradeable game chunk."""
    logger.info("Fetching markets for paper trading in next tradeable game chunk")
    
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
    
    # Get markets
    df = get_markets_for_paper_trading()
    if df.empty:
        logger.info("No markets found in active game chunk")
        return {"success": False, "message": "No markets found in active game chunk"}
    
    # Log market time distribution
    if not df.empty and 'maturity_date' in df.columns:
        earliest = df['maturity_date'].min()
        latest = df['maturity_date'].max()
        logger.info(f"Found {len(df)} market outcomes from {earliest} to {latest}")
    
    # Add required columns
    df['unified_market_type'] = 'winner'
    df['normalized_line'] = 0
    df['market_name'] = df['home_team'] + ' vs ' + df['away_team']
    
    # Calculate implied probabilities
    df['implied_raw'] = 100.0 / df['odds']
    
    # Get signal probabilities
    signal = next(s for s in SIGNAL_PROVIDERS if s.name == "implied_probability")
    df['probability'] = signal.get_probs(df)
    
    # Check for NaN
    nan_count = df['probability'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN probabilities, replacing with implied")
        df['probability'] = df['probability'].fillna(df['implied_raw'] / 100.0)
    
    # Calculate edges
    df['edge'] = (df['probability'] - df['implied_raw']/100.0) * 100
    
    logger.info(f"Edge statistics: min={df['edge'].min():.2f}%, max={df['edge'].max():.2f}%, mean={df['edge'].mean():.2f}%")
    
    # Show markets with positive edge
    positive_edge = df[df['edge'] > 0]
    if len(positive_edge) > 0:
        logger.info(f"Markets with positive edge:")
        for _, m in positive_edge.iterrows():
            logger.info(f"  {m['market_name'][:30]} - {m['normalized_outcome']} @ {m['odds']:.2f} - Edge: {m['edge']:.2f}%")
    
    # Filter to tradeable markets (including negative for hedging)
    tradeable = df[df['edge'].abs() > 0.1]  # At least 0.1% edge either way
    logger.info(f"Found {len(tradeable)} tradeable outcomes after 0.1% filter")
    
    if tradeable.empty:
        return {"success": True, "message": "No tradeable markets found"}
    
    # Run Kelly optimization
    try:
        # Import the risk management function
        from evaluate_open_markets import trim_kelly_results
        
        # First get raw Kelly recommendations
        kelly_raw = calculate_kelly_stakes_with_exclusivity(
            tradeable,
            bankroll=1.0,  # Use unit bankroll for fractions
            correlation_matrix=np.eye(len(tradeable)),  # Simple independence assumption
            risk_adjusted=True
        )
        
        # Apply risk management limits
        execution_bankroll = 10000.0
        trimmed = trim_kelly_results(
            kelly_raw,
            kelly_fraction=0.25,  # Conservative 25% Kelly
            bankroll=execution_bankroll,
            cap_per_game=0.25,    # Max 25% per game
            cap_per_bet=0.25,     # Max 25% per bet  
            cap_per_game_market=0.10,  # Max 10% per market type
            min_bet_abs=10.0,     # Min $10 bet
            min_bet_pct=0.001,    # Min 0.1% of bankroll
        )
        
        # Filter to non-zero stakes after trimming
        to_bet = trimmed[trimmed['stake'] > 0]
        
        logger.info(f"Kelly optimization results:")
        logger.info(f"  Raw Kelly recommendations: {len(kelly_raw)}")
        logger.info(f"  After risk management: {len(trimmed)}")
        logger.info(f"  Non-zero stakes: {len(to_bet)}")
        
        if len(kelly_raw) > 0 and len(to_bet) == 0:
            logger.info("  Risk limits filtered out all positions")
            # Show what was filtered
            for _, bet in kelly_raw.head(5).iterrows():
                logger.info(f"    {bet.get('market_name', 'Unknown')[:30]} - {bet.get('normalized_outcome', '')} - Kelly stake: ${bet.get('stake', 0):.2f}")
        
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
            if abs(delta_stake) < max(10.0, execution_bankroll * 0.001):
                logger.info(f"Skipping {bet['market_name'][:30]} - {bet['normalized_outcome']}: delta ${delta_stake:.2f} too small (current: ${current_stake:.2f}, target: ${target_stake:.2f})")
                continue
            
            # Get execution quote if enabled
            execution_odds = bet['odds']  # Default to market odds
            
            if use_execution_quotes and quote_service and delta_stake > 0:
                # For real trading, we would fetch the execution quote here
                # This requires additional market data (merkle proof, etc) from Overtime API
                logger.info(f"  [Note: Would fetch execution quote for ${abs(delta_stake):.2f} stake]")
                # quote = quote_service.get_single_quote(market_data, position, abs(delta_stake))
                # if quote:
                #     quote_details = quote_service.extract_execution_price(quote)
                #     execution_odds = quote_details.get('decimal_odds', bet['odds'])
            
            logger.info(f"  {bet['market_name']} - {bet['normalized_outcome']} @ {execution_odds:.2f} - Current: ${current_stake:.2f}, Target: ${target_stake:.2f}, Delta: ${delta_stake:.2f}")
            
            # Calculate edge from probability and execution odds
            implied_prob = 1.0 / execution_odds
            edge = (bet['probability'] - implied_prob) * 100
            
            # Calculate fees for this trade
            fee_info = {}
            if delta_stake > 0:  # Opening/increasing position
                fee_info = fee_calculator.calculate_entry_fees(abs(delta_stake), execution_odds)
                logger.info(f"    Fees: ${fee_info['fee_amount']:.2f} ({fee_info['total_fee_pct']*100:.1f}% - SafeBox: {fee_info['safebox_fee_pct']*100:.1f}%, Skew: {fee_info['skew_fee_pct']*100:.1f}%)")
            
            trade_details.append({
                'market_id': bet['source_id'],
                'market_name': bet['market_name'],
                'outcome': bet['normalized_outcome'],
                'odds': bet['odds'],
                'execution_odds': execution_odds,
                'stake': delta_stake,  # Use delta instead of full stake
                'target_stake': target_stake,
                'current_stake': current_stake,
                'edge': edge,
                'probability': bet['probability'],
                'maturity_date': bet.get('maturity_date'),  # Pass through maturity date
                'fee_info': fee_info  # Add fee information
            })
        
        # Check for positions to close (Kelly no longer recommends)
        if use_sessions and current_session:
            kelly_positions = set((bet['source_id'], bet['normalized_outcome']) for _, bet in trimmed.iterrows())
            
            for pos_key, pos in current_session.get('positions', {}).items():
                parts = pos_key.split('_')
                if len(parts) >= 2:
                    position_tuple = (parts[0], parts[1])
                    if position_tuple not in kelly_positions and pos.get('total_stake', 0) > 0:
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
            "positions_held": len(to_bet) if 'to_bet' in locals() else 0,
            "session_id": current_session['session_id'] if current_session else None
        }
        
    except Exception as e:
        logger.error(f"Kelly optimization error: {e}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    result = execute_simple_paper_trades()
    print(result)