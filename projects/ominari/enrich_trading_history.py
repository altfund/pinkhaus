#!/usr/bin/env python3
"""
Enrich paper trading history with detailed odds and edge calculations.

This script adds:
- Historical odds at trade time
- Edge calculations based on signal vs implied probability
- Market movement tracking
- P&L verification
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from database_v2 import db_manager
from models import Market, Odd
from paper_trading_sessions import PaperTradingSessionManager
from sqlalchemy import and_, func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingHistoryEnricher:
    """Enriches paper trading positions with historical odds and edges."""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        
    def get_odds_history_for_position(self, position: Dict) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get complete odds history for a position's market."""
        market_id = position.get('market_id')
        if not market_id:
            return {}
        
        odds_history = {'Home': [], 'Draw': [], 'Away': []}
        
        with db_manager.get_db_session() as db:
            # Get all odds updates for this market
            odds_records = db.query(Odd).filter(
                Odd.source_id == market_id
            ).order_by(Odd.updated_at).all()
            
            for odd in odds_records:
                if odd.outcome in odds_history:
                    odds_history[odd.outcome].append(
                        (odd.updated_at, odd.decimal_odds)
                    )
        
        return odds_history
    
    def calculate_edge_at_trade_time(self, trade: Dict, position: Dict) -> Optional[float]:
        """Calculate the edge at the time of trade."""
        market_id = position.get('market_id')
        outcome = position.get('outcome')
        trade_time = trade.get('timestamp')
        
        if not all([market_id, outcome, trade_time]):
            return None
        
        try:
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
            
            with db_manager.get_db_session() as db:
                # Get the odds at trade time
                odd_at_trade = db.query(Odd).filter(
                    Odd.source_id == market_id,
                    Odd.outcome == outcome,
                    Odd.updated_at <= trade_time
                ).order_by(Odd.updated_at.desc()).first()
                
                if odd_at_trade:
                    implied_prob = 1.0 / odd_at_trade.decimal_odds
                    
                    # Get signal probability from trade
                    signal_prob = trade.get('signal_probability')
                    if not signal_prob:
                        # Try to extract from edge if available
                        raw_edge = trade.get('edge', 0)
                        if raw_edge:
                            # Reverse engineer signal probability from edge
                            # edge = (signal - implied) * 100
                            signal_prob = implied_prob + (raw_edge / 100)
                    
                    if signal_prob:
                        edge = (signal_prob - implied_prob) * 100
                        return edge
                    
        except Exception as e:
            logger.error(f"Error calculating edge: {e}")
        
        return None
    
    def enrich_position_with_detailed_history(self, position: Dict) -> Dict:
        """Add detailed historical data to a position."""
        enriched = position.copy()
        
        # Get full odds history
        odds_history = self.get_odds_history_for_position(position)
        
        # Add odds movement summary
        outcome = position.get('outcome')
        if outcome and outcome in odds_history:
            history = odds_history[outcome]
            if history:
                # Opening and closing odds
                enriched['odds_movement'] = {
                    'opening': history[0][1] if history else None,
                    'closing': history[-1][1] if history else None,
                    'min': min(h[1] for h in history) if history else None,
                    'max': max(h[1] for h in history) if history else None,
                    'changes': len(history)
                }
                
                # Calculate odds movement during position lifetime
                opened_at = position.get('opened_at')
                closed_at = position.get('closed_at')
                
                if opened_at:
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                    
                    # Find odds at position open
                    odds_at_open = None
                    for timestamp, odds in history:
                        if timestamp >= opened_at:
                            odds_at_open = odds
                            break
                    
                    if not odds_at_open and history:
                        # Use last known odds before position open
                        for i in range(len(history)-1, -1, -1):
                            if history[i][0] < opened_at:
                                odds_at_open = history[i][1]
                                break
                    
                    enriched['odds_at_open'] = odds_at_open
                    
                    # Find odds at close if position is closed
                    if closed_at and isinstance(closed_at, str):
                        closed_at = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                        odds_at_close = None
                        
                        for timestamp, odds in reversed(history):
                            if timestamp <= closed_at:
                                odds_at_close = odds
                                break
                        
                        enriched['odds_at_close'] = odds_at_close
        
        # Enrich each trade with edge calculations
        enriched_trades = []
        for trade in position.get('trades', []):
            trade_copy = trade.copy()
            
            # Calculate edge at trade time
            calculated_edge = self.calculate_edge_at_trade_time(trade, position)
            if calculated_edge is not None:
                trade_copy['calculated_edge'] = calculated_edge
                
                # Compare with stored edge
                stored_edge = trade.get('edge', 0)
                if stored_edge:
                    trade_copy['edge_difference'] = calculated_edge - stored_edge
            
            enriched_trades.append(trade_copy)
        
        enriched['trades'] = enriched_trades
        
        # Add summary statistics
        if enriched_trades:
            edges = [t.get('calculated_edge', 0) for t in enriched_trades if t.get('calculated_edge')]
            if edges:
                enriched['edge_stats'] = {
                    'avg_edge': sum(edges) / len(edges),
                    'min_edge': min(edges),
                    'max_edge': max(edges),
                    'trades_with_edge': len(edges)
                }
        
        return enriched
    
    def verify_pnl_calculation(self, position: Dict) -> Dict[str, any]:
        """Verify P&L calculation for a position."""
        verification = {
            'position_id': position.get('id'),
            'reported_pnl': position.get('pnl', 0),
            'calculated_pnl': 0,
            'matches': True,
            'details': []
        }
        
        # For closed positions, verify the P&L
        if position.get('result') in ['won', 'lost']:
            stake = position.get('total_stake', 0)
            execution_stake = position.get('execution_stake', stake)
            avg_odds = position.get('avg_odds', 0)
            result = position.get('result')
            
            if result == 'won':
                # Payout minus execution stake
                payout = stake * avg_odds
                calculated_pnl = payout - execution_stake
            else:
                # Lost entire execution stake
                calculated_pnl = -execution_stake
            
            verification['calculated_pnl'] = calculated_pnl
            verification['matches'] = abs(calculated_pnl - position.get('pnl', 0)) < 0.01
            
            if not verification['matches']:
                verification['details'].append(
                    f"P&L mismatch: reported {position.get('pnl', 0):.2f}, "
                    f"calculated {calculated_pnl:.2f}"
                )
        
        return verification
    
    def enrich_all_sessions(self, verify_pnl: bool = True):
        """Enrich all paper trading sessions with historical data."""
        logger.info("Enriching all paper trading sessions...")
        
        sessions = self.session_manager.sessions.get('sessions', {})
        total_positions = 0
        pnl_mismatches = []
        
        for session_id, session in sessions.items():
            logger.info(f"Processing session {session_id}...")
            
            # Enrich open positions
            for pos_key, position in session.get('positions', {}).items():
                enriched = self.enrich_position_with_detailed_history(position)
                session['positions'][pos_key] = enriched
                total_positions += 1
            
            # Enrich closed positions
            enriched_closed = []
            for position in session.get('closed_positions', []):
                enriched = self.enrich_position_with_detailed_history(position)
                
                # Verify P&L if requested
                if verify_pnl:
                    verification = self.verify_pnl_calculation(enriched)
                    if not verification['matches']:
                        pnl_mismatches.append(verification)
                        enriched['pnl_verification'] = verification
                
                enriched_closed.append(enriched)
                total_positions += 1
            
            session['closed_positions'] = enriched_closed
        
        # Save enriched sessions
        self.session_manager._save_sessions()
        
        # Report results
        print(f"\n✅ Enriched {total_positions} positions across {len(sessions)} sessions")
        
        if pnl_mismatches:
            print(f"\n⚠️  Found {len(pnl_mismatches)} P&L mismatches:")
            for mismatch in pnl_mismatches[:5]:
                print(f"  Position {mismatch['position_id']}: "
                      f"reported ${mismatch['reported_pnl']:.2f}, "
                      f"calculated ${mismatch['calculated_pnl']:.2f}")
        else:
            print("\n✅ All P&L calculations verified correctly")
        
        logger.info("Enrichment complete")


def main():
    """Run the trading history enrichment."""
    print("="*60)
    print("PAPER TRADING HISTORY ENRICHMENT")
    print("="*60)
    
    enricher = TradingHistoryEnricher()
    
    print("\nThis will enrich your paper trading history with:")
    print("- Historical odds at trade time")
    print("- Calculated edges based on signals")
    print("- Odds movement tracking")
    print("- P&L verification")
    print("\nProcessing...")
    
    enricher.enrich_all_sessions(verify_pnl=True)
    
    print("\n✅ Enrichment complete!")
    print("✅ Your paper trading sessions now have full historical data")


if __name__ == "__main__":
    main()