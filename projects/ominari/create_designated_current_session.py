#!/usr/bin/env python3
"""
Create a designated current session for production consistency.
This ensures both heartbeat and dashboard use the same session after restarts.
"""

from paper_trading_sessions import PaperTradingSessionManager
from datetime import datetime, timezone

def create_current_production_session():
    """Create a fresh session designated as the current production session."""
    
    session_manager = PaperTradingSessionManager()
    
    # Create a fresh session with designated name
    current_time = datetime.now(timezone.utc)
    session_id = current_time.strftime("20251120_PRODUCTION_CURRENT")
    
    print(f"Creating designated production session: {session_id}")
    
    # Create session with explicit parameters  
    new_session_id = session_manager.create_session(
        initial_bankroll=10000.0,
        session_name=f"Production Current Session {current_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )
    
    # Mark all other sessions as ended to avoid conflicts
    sessions = session_manager.sessions.get("sessions", {})
    for sid, session in sessions.items():
        if sid != session_id and session.get("status") == "active":
            session["status"] = "ended"
            print(f"Marked session {sid} as ended")
    
    # Ensure this session is active
    current_session = sessions.get(session_id)
    if current_session:
        current_session["status"] = "active"
    
    # Force set as current session  
    session_manager.current_session_id = new_session_id
    session_manager._save_sessions()
    
    print(f"✅ Production session created: {session_id}")
    print(f"✅ Bankroll: ${current_session['initial_bankroll']:.2f}")
    print(f"✅ Status: {current_session['status']}")
    print(f"✅ This session will be used by both heartbeat and dashboard")
    
    return session_id

if __name__ == "__main__":
    create_current_production_session()