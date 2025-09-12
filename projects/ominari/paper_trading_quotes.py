#!/usr/bin/env python3
"""Quote recording system for paper trading executions."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class QuoteRecorder:
    """Records quotes at the time of trade execution for paper trading."""
    
    def __init__(self, quotes_file: str = "paper_trading_quotes.json"):
        self.quotes_file = quotes_file
        self.quotes = self._load_quotes()
    
    def _load_quotes(self) -> Dict[str, Any]:
        """Load existing quotes from file."""
        if os.path.exists(self.quotes_file):
            try:
                with open(self.quotes_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading quotes: {e}")
                return {"sessions": []}
        return {"sessions": []}
    
    def _save_quotes(self):
        """Save quotes to file."""
        try:
            with open(self.quotes_file, 'w') as f:
                json.dump(self.quotes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving quotes: {e}")
    
    def record_trade_quotes(self, trade_details: List[Dict[str, Any]], session_id: str = None) -> str:
        """
        Record quotes for executed trades.
        
        Args:
            trade_details: List of trade dictionaries with market info and odds
            session_id: Optional session identifier
            
        Returns:
            session_id used for this recording
        """
        if not session_id:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        session = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trades": []
        }
        
        for trade in trade_details:
            quote_record = {
                "market_id": trade.get("market_id"),
                "market_name": trade.get("market_name"),
                "outcome": trade.get("outcome"),
                "execution_odds": trade.get("odds"),
                "stake": trade.get("stake"),
                "edge": trade.get("edge"),
                "probability": trade.get("probability"),
                "quote_timestamp": datetime.now(timezone.utc).isoformat()
            }
            session["trades"].append(quote_record)
        
        # Add session to quotes
        self.quotes["sessions"].append(session)
        self._save_quotes()
        
        logger.info(f"Recorded {len(trade_details)} quotes for session {session_id}")
        return session_id
    
    def get_session_quotes(self, session_id: str) -> Dict[str, Any]:
        """Get quotes for a specific session."""
        for session in self.quotes.get("sessions", []):
            if session.get("session_id") == session_id:
                return session
        return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent trading sessions."""
        sessions = self.quotes.get("sessions", [])
        return sessions[-limit:] if sessions else []
    
    def update_trade_result(self, session_id: str, market_id: str, outcome: str, 
                          result: str, final_odds: float = None):
        """
        Update a trade with its result after market resolution.
        
        Args:
            session_id: Trading session ID
            market_id: Market identifier
            outcome: Bet outcome (Home/Draw/Away)
            result: 'won', 'lost', or 'pending'
            final_odds: Final odds at market close (optional)
        """
        for session in self.quotes.get("sessions", []):
            if session.get("session_id") == session_id:
                for trade in session.get("trades", []):
                    if (trade.get("market_id") == market_id and 
                        trade.get("outcome") == outcome):
                        trade["result"] = result
                        if final_odds:
                            trade["final_odds"] = final_odds
                        trade["result_timestamp"] = datetime.now(timezone.utc).isoformat()
                        self._save_quotes()
                        logger.info(f"Updated trade result for {market_id} - {outcome}: {result}")
                        return True
        return False