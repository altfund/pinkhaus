#!/usr/bin/env python3
"""Paper trading session management."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PaperTradingSessionManager:
    """Manages paper trading sessions with persistent state."""
    
    def __init__(self, sessions_file: str = "paper_trading_sessions.json"):
        self.sessions_file = sessions_file
        self.sessions = self._load_sessions()
        self.current_session_id = None
    
    def _load_sessions(self) -> Dict[str, Any]:
        """Load existing sessions from file."""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
                return {"sessions": {}, "metadata": {}}
        return {"sessions": {}, "metadata": {}}
    
    def _save_sessions(self):
        """Save sessions to file."""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def create_session(self, initial_bankroll: float = 10000.0, 
                      session_name: str = None) -> str:
        """
        Create a new trading session.
        
        Args:
            initial_bankroll: Starting bankroll amount
            session_name: Optional descriptive name
            
        Returns:
            Session ID
        """
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        session = {
            "session_id": session_id,
            "session_name": session_name or f"Session {session_id}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "initial_bankroll": initial_bankroll,
            "current_bankroll": initial_bankroll,
            "portfolio_value": initial_bankroll,
            "positions": {},
            "closed_positions": [],
            "trades": [],
            "performance": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "pending_trades": 0,
                "total_pnl": 0.0,
                "total_stake": 0.0,
                "max_drawdown": 0.0,
                "peak_value": initial_bankroll
            },
            "status": "active"
        }
        
        self.sessions["sessions"][session_id] = session
        self.current_session_id = session_id
        self._save_sessions()
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get the current active session."""
        if not self.current_session_id:
            # Find the most recent active session
            active_sessions = [
                (sid, s) for sid, s in self.sessions.get("sessions", {}).items()
                if s.get("status") == "active"
            ]
            if active_sessions:
                # Sort by creation time and get the most recent
                active_sessions.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
                self.current_session_id = active_sessions[0][0]
            else:
                # Create a new session if none exist
                self.current_session_id = self.create_session()
        
        return self.sessions["sessions"].get(self.current_session_id)
    
    def record_trades(self, session_id: str, trades: List[Dict[str, Any]]) -> bool:
        """
        Record executed trades in a session.
        
        Args:
            session_id: Session identifier
            trades: List of trade details
            
        Returns:
            Success boolean
        """
        session = self.sessions["sessions"].get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        # Calculate total stake and fees
        total_stake = 0
        total_fees = 0
        for t in trades:
            stake = t.get("stake", 0)
            fee_info = t.get("fee_info", {})
            fee_amount = fee_info.get("fee_amount", 0) if stake > 0 else 0
            total_stake += stake
            total_fees += fee_amount
        
        # Update session (deduct stake plus fees)
        session["current_bankroll"] -= (total_stake + total_fees)
        session["trades"].extend(trades)
        session["performance"]["total_trades"] += len(trades)
        session["performance"]["pending_trades"] += len(trades)
        session["performance"]["total_stake"] += abs(total_stake)
        session["performance"]["total_fees"] = session["performance"].get("total_fees", 0) + total_fees
        
        # Add to trading logs
        if "trading_logs" not in session:
            session["trading_logs"] = []
        
        # Log each trade
        for trade in trades:
            if trade.get("stake", 0) != 0:  # Only log actual trades
                log_message = f"Trade executed: {trade.get('market_name', 'Unknown')} {trade.get('outcome', '')} @ {trade.get('odds', 0):.2f} - Stake: ${abs(trade.get('stake', 0)):.2f} - Edge: {trade.get('edge', 0):.1f}%"
                session["trading_logs"].append({
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "message": log_message,
                    "level": "trade"
                })
        
        # Update positions
        for trade in trades:
            position_key = f"{trade['market_id']}_{trade['outcome']}"
            
            if position_key in session["positions"]:
                # Update existing position
                pos = session["positions"][position_key]
                old_stake = pos["total_stake"]
                new_stake = old_stake + trade["stake"]
                
                # Check if position is being closed
                if abs(new_stake) < 0.01:  # Effectively zero
                    # Move to closed positions
                    pos["status"] = "closed"
                    pos["closed_at"] = datetime.now(timezone.utc).isoformat()
                    pos["close_reason"] = "rebalanced_out"
                    session["closed_positions"].append(pos)
                    del session["positions"][position_key]
                    logger.info(f"Closed position {position_key} via rebalancing")
                else:
                    # Update position
                    if trade["stake"] > 0:  # Only update avg odds on buys
                        pos["avg_odds"] = ((pos["avg_odds"] * old_stake) + (trade["odds"] * trade["stake"])) / new_stake
                    pos["total_stake"] = new_stake
                    pos["current_value"] = new_stake  # Will be updated with mark-to-market later
                    pos["trades"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "stake": trade["stake"],
                        "odds": trade["odds"],
                        "type": "rebalance"
                    })
            elif trade["stake"] > 0:  # Only create position for positive stakes
                # Create new position
                fee_info = trade.get("fee_info", {})
                session["positions"][position_key] = {
                    "market_id": trade["market_id"],
                    "market_name": trade["market_name"],
                    "outcome": trade["outcome"],
                    "total_stake": trade["stake"],
                    "avg_odds": trade["odds"],
                    "current_value": trade["stake"],
                    "pnl": 0.0,
                    "status": "open",
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "maturity_date": trade.get("maturity_date"),  # Store market close time
                    "fee_info": fee_info,  # Store fee information
                    "execution_stake": fee_info.get("execution_stake", trade["stake"]),  # Stake including fees
                    "trades": [{
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "stake": trade["stake"],
                        "odds": trade["odds"],
                        "type": "open",
                        "fee_info": fee_info
                    }]
                }
        
        # Update portfolio value
        positions_value = sum(p["current_value"] for p in session["positions"].values())
        session["portfolio_value"] = session["current_bankroll"] + positions_value
        
        # Check for drawdown
        if session["portfolio_value"] < session["performance"]["peak_value"]:
            drawdown = (session["performance"]["peak_value"] - session["portfolio_value"]) / session["performance"]["peak_value"]
            session["performance"]["max_drawdown"] = max(session["performance"]["max_drawdown"], drawdown)
        else:
            session["performance"]["peak_value"] = session["portfolio_value"]
        
        self._save_sessions()
        logger.info(f"Recorded {len(trades)} trades in session {session_id}")
        return True
    
    def settle_finished_markets(self, session_id: str, market_results: Dict[str, Dict[str, Any]]) -> int:
        """
        Settle positions for finished markets.
        
        Args:
            session_id: Session identifier
            market_results: Dict mapping market_id -> {is_finished, resolved_outcome, home_score, away_score}
            
        Returns:
            Number of positions settled
        """
        session = self.sessions["sessions"].get(session_id)
        if not session:
            return 0
        
        settled_count = 0
        for position_key, pos in list(session["positions"].items()):
            market_id = pos.get("market_id")
            if market_id in market_results:
                market_info = market_results[market_id]
                if market_info.get("is_finished") and market_info.get("resolved_outcome"):
                    # Settle this position
                    winning_outcome = market_info["resolved_outcome"]
                    if self.close_position(session_id, position_key, winning_outcome):
                        settled_count += 1
                        logger.info(f"Settled position {position_key} for market {market_id}")
        
        return settled_count
    
    def close_position(self, session_id: str, position_key: str, 
                      winning_outcome: str = None, final_odds: float = None) -> bool:
        """
        Close a position and calculate P&L.
        
        Args:
            session_id: Session identifier
            position_key: Position key (market_id_outcome)
            winning_outcome: The winning outcome for the market
            final_odds: Final odds at close
            
        Returns:
            Success boolean
        """
        session = self.sessions["sessions"].get(session_id)
        if not session or position_key not in session["positions"]:
            return False
        
        position = session["positions"][position_key]
        
        # Calculate P&L
        if winning_outcome:
            # Get execution stake (stake + fees)
            execution_stake = position.get("execution_stake", position["total_stake"])
            
            if position["outcome"] == winning_outcome:
                # Win - get stake * odds (gross payout)
                position["final_value"] = position["total_stake"] * position["avg_odds"]
                # P&L accounts for fees paid on entry
                position["pnl"] = position["final_value"] - execution_stake
                position["result"] = "won"
                session["performance"]["winning_trades"] += 1
            else:
                # Loss - lose entire execution stake (including fees)
                position["final_value"] = 0
                position["pnl"] = -execution_stake
                position["result"] = "lost"
                session["performance"]["losing_trades"] += 1
            
            session["performance"]["pending_trades"] -= 1
            session["performance"]["total_pnl"] += position["pnl"]
            
            # Calculate ROI based on execution stake
            if execution_stake > 0:
                position["roi"] = (position["pnl"] / execution_stake) * 100
            session["current_bankroll"] += position["final_value"]
        
        position["status"] = "closed"
        position["closed_at"] = datetime.now(timezone.utc).isoformat()
        if final_odds:
            position["final_odds"] = final_odds
        
        # Move to closed positions
        session["closed_positions"].append(position)
        del session["positions"][position_key]
        
        # Update portfolio value
        positions_value = sum(p["current_value"] for p in session["positions"].values())
        session["portfolio_value"] = session["current_bankroll"] + positions_value
        
        self._save_sessions()
        logger.info(f"Closed position {position_key} with P&L: {position['pnl']:.2f}")
        return True
    
    def update_position_values(self, session_id: str, market_odds: Dict[str, Dict[str, float]]) -> bool:
        """
        Update position values based on current market odds.
        
        Args:
            session_id: Session identifier
            market_odds: Dict mapping market_id -> outcome -> current_odds
            
        Returns:
            Success boolean
        """
        session = self.sessions["sessions"].get(session_id)
        if not session:
            return False
        
        for position_key, pos in session["positions"].items():
            market_id = pos.get("market_id")
            outcome = pos.get("outcome")
            stake = pos.get("total_stake", 0)
            entry_price = pos.get("avg_odds", 0)
            
            # Get current odds if available
            current_price = entry_price  # Default to entry price
            if market_id in market_odds and outcome in market_odds.get(market_id, {}):
                current_price = market_odds[market_id][outcome]
            
            # Calculate mark-to-market value
            # Formula: current_value = stake * (current_odds / entry_odds)
            if current_price > 0 and entry_price > 0:
                current_value = stake * (current_price / entry_price)
            else:
                current_value = stake
            
            pos["current_value"] = current_value
            pos["pnl"] = current_value - stake
        
        # Update portfolio value
        positions_value = sum(p["current_value"] for p in session["positions"].values())
        session["portfolio_value"] = session["current_bankroll"] + positions_value
        
        self._save_sessions()
        return True
    
    def get_session_performance(self, session_id: str) -> Dict[str, Any]:
        """Get performance metrics for a session."""
        session = self.sessions["sessions"].get(session_id)
        if not session:
            return {}
        
        perf = session["performance"].copy()
        
        # Add calculated metrics
        if perf["total_trades"] > 0:
            # Calculate win rate based only on resolved trades (exclude open/pending)
            resolved_trades = perf["winning_trades"] + perf["losing_trades"]
            if resolved_trades > 0:
                perf["win_rate"] = (perf["winning_trades"] / resolved_trades) * 100
            else:
                perf["win_rate"] = 0
            perf["avg_stake"] = perf["total_stake"] / perf["total_trades"]
        else:
            perf["win_rate"] = 0
            perf["avg_stake"] = 0
        
        # Calculate total execution stakes (including fees) for accurate ROI
        total_execution_stake = perf.get("total_fees", 0) + perf.get("total_stake", 0)
        
        # ROI based on total P&L vs total execution stake
        if total_execution_stake > 0:
            perf["roi_on_execution"] = (perf["total_pnl"] / total_execution_stake) * 100
        else:
            perf["roi_on_execution"] = 0
            
        # Overall portfolio ROI
        perf["roi"] = (session["portfolio_value"] - session["initial_bankroll"]) / session["initial_bankroll"] * 100
        perf["current_value"] = session["portfolio_value"]
        
        return perf
    
    def end_session(self, session_id: str) -> bool:
        """Mark a session as ended."""
        session = self.sessions["sessions"].get(session_id)
        if not session:
            return False
        
        session["status"] = "ended"
        session["ended_at"] = datetime.now(timezone.utc).isoformat()
        
        # Close all open positions as pending
        for position_key in list(session["positions"].keys()):
            self.close_position(session_id, position_key)
        
        self._save_sessions()
        logger.info(f"Ended session {session_id}")
        
        # Clear current session if it's the one being ended
        if self.current_session_id == session_id:
            self.current_session_id = None
        
        return True