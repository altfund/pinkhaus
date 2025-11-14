"""
Bankroll Configuration for Paper Trading
Tracks real bankroll amounts and persists them across sessions
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

class BankrollConfig:
    """Manages persistent bankroll configuration"""
    
    def __init__(self, config_file: str = "config/bankroll.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load bankroll configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading bankroll config: {e}")
                
        # Default configuration
        return {
            "initial_bankroll": 10000.0,
            "current_bankroll": 10000.0,
            "last_updated": datetime.now().isoformat(),
            "trading_enabled": True,
            "risk_limits": {
                "max_position_size": 0.25,  # 25% of bankroll
                "max_daily_loss": 0.10,     # 10% daily loss limit
                "min_bet_size": 10.0,
                "max_bet_size": 1000.0
            },
            "performance": {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "highest_bankroll": 10000.0,
                "lowest_bankroll": 10000.0
            }
        }
        
    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        self.config["last_updated"] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def get_current_bankroll(self) -> float:
        """Get current bankroll amount"""
        return self.config["current_bankroll"]
        
    def update_bankroll(self, amount: float, reason: str = ""):
        """Update bankroll amount"""
        old_bankroll = self.config["current_bankroll"]
        self.config["current_bankroll"] = amount
        
        # Update performance tracking
        self.config["performance"]["total_pnl"] = amount - self.config["initial_bankroll"]
        
        if amount > self.config["performance"]["highest_bankroll"]:
            self.config["performance"]["highest_bankroll"] = amount
        if amount < self.config["performance"]["lowest_bankroll"]:
            self.config["performance"]["lowest_bankroll"] = amount
            
        # Log the change
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "old_bankroll": old_bankroll,
            "new_bankroll": amount,
            "change": amount - old_bankroll,
            "reason": reason
        }
        
        # Append to log file
        log_file = self.config_file.replace('.json', '_log.json')
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                pass
                
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
            
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        self.save_config()
        
    def record_bet_result(self, won: bool, pnl: float):
        """Record the result of a bet"""
        self.config["performance"]["total_bets"] += 1
        
        if won:
            self.config["performance"]["wins"] += 1
        else:
            self.config["performance"]["losses"] += 1
            
        # Update bankroll with PnL
        new_bankroll = self.config["current_bankroll"] + pnl
        self.update_bankroll(new_bankroll, f"Bet result: {'Win' if won else 'Loss'} PnL: ${pnl:.2f}")
        
    def get_risk_limits(self) -> Dict:
        """Get current risk limits"""
        return self.config["risk_limits"]
        
    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled"""
        return self.config["trading_enabled"]
        
    def set_trading_enabled(self, enabled: bool):
        """Enable or disable trading"""
        self.config["trading_enabled"] = enabled
        self.save_config()
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        perf = self.config["performance"]
        
        win_rate = 0
        if perf["total_bets"] > 0:
            win_rate = (perf["wins"] / perf["total_bets"]) * 100
            
        return {
            "current_bankroll": self.config["current_bankroll"],
            "initial_bankroll": self.config["initial_bankroll"],
            "total_pnl": perf["total_pnl"],
            "roi": (perf["total_pnl"] / self.config["initial_bankroll"]) * 100,
            "total_bets": perf["total_bets"],
            "wins": perf["wins"],
            "losses": perf["losses"],
            "win_rate": win_rate,
            "highest_bankroll": perf["highest_bankroll"],
            "lowest_bankroll": perf["lowest_bankroll"],
            "drawdown": ((perf["highest_bankroll"] - self.config["current_bankroll"]) / perf["highest_bankroll"]) * 100
        }
        
    def reset_to_initial(self):
        """Reset bankroll to initial amount"""
        self.config["current_bankroll"] = self.config["initial_bankroll"]
        self.config["performance"] = {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "highest_bankroll": self.config["initial_bankroll"],
            "lowest_bankroll": self.config["initial_bankroll"]
        }
        self.save_config()