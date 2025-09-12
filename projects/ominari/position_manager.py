#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Position Management System
Implements portfolio rebalancing, position netting, and risk management.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a betting market."""
    market_id: str
    bet_name: str
    size: float
    entry_price: float
    current_price: float
    timestamp: datetime
    correlation_cluster: Optional[int] = None
    
    @property
    def current_value(self) -> float:
        """Current value of the position."""
        return self.size * self.current_price
        
    @property
    def pnl(self) -> float:
        """Profit/loss of the position."""
        return self.size * (self.current_price - self.entry_price)
        
    @property
    def pnl_percent(self) -> float:
        """Percentage P&L."""
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0


@dataclass
class RiskLimits:
    """Risk management parameters."""
    max_position_size: float = 0.1  # Max 10% in any single bet
    max_cluster_exposure: float = 0.25  # Max 25% in correlated bets
    max_total_exposure: float = 0.6  # Max 60% total exposure
    max_loss_per_position: float = 0.02  # Stop loss at 2% portfolio loss
    min_rebalance_threshold: float = 0.05  # Only rebalance if >5% change
    min_trade_size: float = 0.001  # Minimum 0.1% position size


class PositionManager:
    """
    Advanced position management with rebalancing and risk controls.
    
    Features:
    - Position netting across correlated markets
    - Dynamic rebalancing with thresholds
    - Risk limit enforcement
    - Transaction cost consideration
    """
    
    def __init__(self, 
                 capital: float,
                 risk_limits: Optional[RiskLimits] = None,
                 correlation_lookback: int = 100):
        self.capital = capital
        self.risk_limits = risk_limits or RiskLimits()
        self.correlation_lookback = correlation_lookback
        self.positions: Dict[str, Position] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.clusters: Optional[Dict[str, int]] = None
        
    def update_correlations(self, returns_data: pd.DataFrame):
        """Update correlation matrix and clustering."""
        if len(returns_data.columns) < 2:
            return
            
        # Calculate correlation matrix
        self.correlation_matrix = returns_data.tail(
            self.correlation_lookback
        ).corr()
        
        # Perform hierarchical clustering
        if len(self.correlation_matrix) > 1:
            # Convert correlation to distance
            distances = 1 - self.correlation_matrix.abs()
            condensed_dist = squareform(distances)
            
            # Hierarchical clustering
            linkage_matrix = linkage(condensed_dist, method='average')
            
            # Form clusters (threshold at 0.5 correlation)
            cluster_labels = fcluster(linkage_matrix, 0.5, criterion='distance')
            
            # Map markets to clusters
            self.clusters = dict(zip(self.correlation_matrix.index, cluster_labels))
            
    def add_position(self, position: Position):
        """Add or update a position."""
        # Assign correlation cluster if available
        if self.clusters and position.market_id in self.clusters:
            position.correlation_cluster = self.clusters[position.market_id]
            
        self.positions[position.market_id] = position
        logger.info(f"Added position: {position.market_id} size={position.size}")
        
    def get_cluster_exposure(self, cluster_id: int) -> float:
        """Calculate total exposure in a correlation cluster."""
        cluster_positions = [
            p for p in self.positions.values() 
            if p.correlation_cluster == cluster_id
        ]
        
        total_value = sum(p.current_value for p in cluster_positions)
        return abs(total_value) / self.capital
        
    def get_total_exposure(self) -> float:
        """Calculate total portfolio exposure."""
        total_value = sum(abs(p.current_value) for p in self.positions.values())
        return total_value / self.capital
        
    def check_risk_limits(self, 
                         new_position: Position,
                         is_update: bool = False) -> Tuple[bool, str]:
        """Check if a new/updated position violates risk limits."""
        # Single position size check
        position_exposure = abs(new_position.current_value) / self.capital
        if position_exposure > self.risk_limits.max_position_size:
            return False, f"Position size {position_exposure:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}"
            
        # Cluster exposure check
        if new_position.correlation_cluster:
            current_cluster_exposure = self.get_cluster_exposure(
                new_position.correlation_cluster
            )
            
            # If updating, subtract current position first
            if is_update and new_position.market_id in self.positions:
                old_pos = self.positions[new_position.market_id]
                current_cluster_exposure -= abs(old_pos.current_value) / self.capital
                
            new_cluster_exposure = current_cluster_exposure + position_exposure
            
            if new_cluster_exposure > self.risk_limits.max_cluster_exposure:
                return False, f"Cluster exposure {new_cluster_exposure:.1%} exceeds limit {self.risk_limits.max_cluster_exposure:.1%}"
                
        # Total exposure check
        current_total = self.get_total_exposure()
        if is_update and new_position.market_id in self.positions:
            old_pos = self.positions[new_position.market_id]
            current_total -= abs(old_pos.current_value) / self.capital
            
        new_total = current_total + position_exposure
        
        if new_total > self.risk_limits.max_total_exposure:
            return False, f"Total exposure {new_total:.1%} exceeds limit {self.risk_limits.max_total_exposure:.1%}"
            
        return True, "OK"
        
    def should_rebalance(self,
                        current_position: Position,
                        target_size: float) -> bool:
        """Determine if position should be rebalanced."""
        if current_position.size == 0:
            # New position - check minimum size
            return abs(target_size) * current_position.current_price / self.capital >= self.risk_limits.min_trade_size
            
        # Calculate percentage change
        size_change = abs(target_size - current_position.size)
        pct_change = size_change / abs(current_position.size)
        
        # Check against threshold
        return pct_change >= self.risk_limits.min_rebalance_threshold
        
    def calculate_rebalancing_trades(self,
                                   target_positions: Dict[str, float],
                                   current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate trades needed to rebalance to target positions."""
        trades = []
        
        for market_id, target_size in target_positions.items():
            current_pos = self.positions.get(market_id)
            current_size = current_pos.size if current_pos else 0
            current_price = current_prices.get(market_id, 0)
            
            if current_price <= 0:
                continue
                
            # Create position object for risk checking
            new_position = Position(
                market_id=market_id,
                bet_name="",  # Would be filled from market data
                size=target_size,
                entry_price=current_price,
                current_price=current_price,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Check risk limits
            valid, reason = self.check_risk_limits(new_position, is_update=True)
            if not valid:
                logger.warning(f"Skipping {market_id}: {reason}")
                continue
                
            # Check rebalancing threshold
            if current_pos:
                if not self.should_rebalance(current_pos, target_size):
                    continue
                    
            # Calculate trade
            trade_size = target_size - current_size
            if abs(trade_size) > 0:
                trades.append({
                    'market_id': market_id,
                    'current_size': current_size,
                    'target_size': target_size,
                    'trade_size': trade_size,
                    'side': 'buy' if trade_size > 0 else 'sell',
                    'price': current_price,
                    'value': abs(trade_size * current_price)
                })
                
        return trades
        
    def net_cluster_positions(self) -> Dict[int, Dict[str, Any]]:
        """Calculate net exposure by correlation cluster."""
        if not self.clusters:
            return {}
            
        cluster_stats = {}
        
        for cluster_id in set(self.clusters.values()):
            cluster_positions = [
                p for p in self.positions.values()
                if p.correlation_cluster == cluster_id
            ]
            
            if not cluster_positions:
                continue
                
            total_value = sum(p.current_value for p in cluster_positions)
            total_size = sum(p.size for p in cluster_positions)
            market_count = len(cluster_positions)
            
            cluster_stats[cluster_id] = {
                'net_value': total_value,
                'net_size': total_size,
                'exposure_pct': abs(total_value) / self.capital,
                'market_count': market_count,
                'markets': [p.market_id for p in cluster_positions]
            }
            
        return cluster_stats
        
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions."""
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for market_id, pos in self.positions.items():
            data.append({
                'market_id': market_id,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'current_value': pos.current_value,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_percent,
                'exposure_pct': abs(pos.current_value) / self.capital,
                'cluster': pos.correlation_cluster,
                'timestamp': pos.timestamp.isoformat() if pos.timestamp else None
            })
            
        return pd.DataFrame(data)
        
    def check_stop_losses(self) -> List[str]:
        """Check for positions that hit stop loss."""
        stopped_positions = []
        
        for market_id, pos in self.positions.items():
            loss_pct = -pos.pnl / self.capital
            
            if loss_pct >= self.risk_limits.max_loss_per_position:
                stopped_positions.append(market_id)
                logger.warning(f"Stop loss triggered for {market_id}: {loss_pct:.1%} loss")
                
        return stopped_positions


def create_position_manager(capital: float = 100000) -> PositionManager:
    """Create a position manager with default settings."""
    risk_limits = RiskLimits(
        max_position_size=0.1,
        max_cluster_exposure=0.25,
        max_total_exposure=0.6,
        max_loss_per_position=0.02,
        min_rebalance_threshold=0.05,
        min_trade_size=0.001
    )
    
    return PositionManager(capital, risk_limits)


def main():
    """Example usage of position manager."""
    # Create manager
    pm = create_position_manager(capital=100000)
    
    # Add some positions
    positions = [
        Position("NFL_GAME_1", "Team A", 100, 2.0, 2.1, datetime.now(timezone.utc)),
        Position("NFL_GAME_2", "Team B", 150, 1.8, 1.75, datetime.now(timezone.utc)),
        Position("NBA_GAME_1", "Team C", 200, 2.5, 2.6, datetime.now(timezone.utc)),
    ]
    
    for pos in positions:
        pm.add_position(pos)
        
    # Check portfolio status
    print("\nPortfolio Summary:")
    print(pm.get_position_summary())
    
    print(f"\nTotal Exposure: {pm.get_total_exposure():.1%}")
    
    # Test rebalancing
    target_positions = {
        "NFL_GAME_1": 120,  # Increase
        "NFL_GAME_2": 100,  # Decrease
        "NBA_GAME_1": 200,  # Keep same
        "NHL_GAME_1": 50,   # New position
    }
    
    current_prices = {
        "NFL_GAME_1": 2.1,
        "NFL_GAME_2": 1.75,
        "NBA_GAME_1": 2.6,
        "NHL_GAME_1": 3.0,
    }
    
    trades = pm.calculate_rebalancing_trades(target_positions, current_prices)
    
    print("\nRebalancing Trades:")
    for trade in trades:
        print(f"  {trade['market_id']}: {trade['side']} {abs(trade['trade_size'])} units (${trade['value']:.2f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()