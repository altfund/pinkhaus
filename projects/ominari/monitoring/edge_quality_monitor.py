#!/usr/bin/env python3
"""
Edge Quality Monitor
Tracks predicted vs realized edges to validate probability models
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeRecord:
    """Record of a single bet's predicted and realized edge"""
    bet_id: str
    timestamp: str
    sport: str
    market_type: str

    # Predicted values (at time of bet)
    predicted_edge: float
    predicted_prob: float
    market_odds: float

    # Realized values (after settlement)
    won: Optional[bool] = None
    realized_pnl: Optional[float] = None
    settled_at: Optional[str] = None

    # Metadata
    strategy_version: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'bet_id': self.bet_id,
            'timestamp': self.timestamp,
            'sport': self.sport,
            'market_type': self.market_type,
            'predicted_edge': self.predicted_edge,
            'predicted_prob': self.predicted_prob,
            'market_odds': self.market_odds,
            'won': self.won,
            'realized_pnl': self.realized_pnl,
            'settled_at': self.settled_at,
            'strategy_version': self.strategy_version,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EdgeRecord':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class EdgeQualityMetrics:
    """Aggregated edge quality metrics"""
    total_bets: int = 0
    total_settled: int = 0

    # Win rate
    predicted_win_rate: float = 0.0
    realized_win_rate: float = 0.0
    win_rate_error: float = 0.0

    # Edge accuracy
    mean_predicted_edge: float = 0.0
    mean_realized_edge: float = 0.0
    edge_error: float = 0.0
    edge_rmse: float = 0.0

    # Probability calibration
    calibration_error: float = 0.0  # Mean absolute calibration error
    brier_score: float = 0.0  # Lower is better

    # By edge bucket (how well do high-edge bets perform vs low-edge)
    edge_buckets: Dict[str, Dict] = field(default_factory=dict)

    # By sport
    sport_metrics: Dict[str, Dict] = field(default_factory=dict)

    # Alerts
    alerts: List[str] = field(default_factory=list)


class EdgeQualityMonitor:
    """
    Monitors edge quality by comparing predicted edges to realized results
    Helps validate that probability models are well-calibrated
    """

    def __init__(self, data_file: str = "monitoring/edge_quality_data.json"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[EdgeRecord] = []
        self._load_data()

    def _load_data(self):
        """Load existing edge records from disk"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)

                self.records = [EdgeRecord.from_dict(record) for record in data]
                logger.info(f"Loaded {len(self.records)} edge records from {self.data_file}")
            except Exception as e:
                logger.error(f"Error loading edge data: {e}")
                self.records = []
        else:
            logger.info("No existing edge data found, starting fresh")

    def _save_data(self):
        """Save edge records to disk"""
        try:
            data = [record.to_dict() for record in self.records]

            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.records)} edge records to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving edge data: {e}")

    def record_bet(self,
                  bet_id: str,
                  predicted_edge: float,
                  predicted_prob: float,
                  market_odds: float,
                  sport: str = "Unknown",
                  market_type: str = "h2h",
                  strategy_version: Optional[str] = None,
                  metadata: Optional[Dict] = None):
        """Record a new bet's predicted edge"""

        record = EdgeRecord(
            bet_id=bet_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sport=sport,
            market_type=market_type,
            predicted_edge=predicted_edge,
            predicted_prob=predicted_prob,
            market_odds=market_odds,
            strategy_version=strategy_version,
            metadata=metadata or {}
        )

        self.records.append(record)
        self._save_data()

        logger.debug(f"Recorded bet {bet_id}: edge={predicted_edge:.2f}%, prob={predicted_prob:.2%}")

    def settle_bet(self, bet_id: str, won: bool, realized_pnl: float):
        """Update bet record with settlement result"""

        for record in self.records:
            if record.bet_id == bet_id:
                record.won = won
                record.realized_pnl = realized_pnl
                record.settled_at = datetime.now(timezone.utc).isoformat()

                logger.debug(f"Settled bet {bet_id}: won={won}, pnl=${realized_pnl:.2f}")
                self._save_data()
                return

        logger.warning(f"Bet {bet_id} not found in edge records")

    def calculate_metrics(self, lookback_days: Optional[int] = None) -> EdgeQualityMetrics:
        """Calculate edge quality metrics"""

        # Filter records
        records = self.records
        if lookback_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            records = [
                r for r in records
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]

        # Filter to settled bets only
        settled_records = [r for r in records if r.won is not None]

        if not settled_records:
            logger.warning("No settled bets to analyze")
            return EdgeQualityMetrics()

        metrics = EdgeQualityMetrics(
            total_bets=len(records),
            total_settled=len(settled_records)
        )

        # Calculate win rates
        predicted_probs = [r.predicted_prob for r in settled_records]
        wins = [1 if r.won else 0 for r in settled_records]

        metrics.predicted_win_rate = np.mean(predicted_probs)
        metrics.realized_win_rate = np.mean(wins)
        metrics.win_rate_error = abs(metrics.realized_win_rate - metrics.predicted_win_rate)

        # Calculate edges
        predicted_edges = [r.predicted_edge for r in settled_records]

        # Realized edge = (actual_return / theoretical_return - 1) * 100
        # Theoretical return = prob * (odds - 1) - (1 - prob)
        realized_edges = []
        for r in settled_records:
            theoretical_return = r.predicted_prob * (r.market_odds - 1) - (1 - r.predicted_prob)
            actual_return = (r.market_odds - 1) if r.won else -1
            realized_edge = ((actual_return / theoretical_return) - 1) * 100 if theoretical_return != 0 else 0
            realized_edges.append(realized_edge)

        metrics.mean_predicted_edge = np.mean(predicted_edges)
        metrics.mean_realized_edge = np.mean(realized_edges)
        metrics.edge_error = metrics.mean_realized_edge - metrics.mean_predicted_edge

        # RMSE of edge predictions
        edge_errors = np.array(realized_edges) - np.array(predicted_edges)
        metrics.edge_rmse = np.sqrt(np.mean(edge_errors**2))

        # Probability calibration - Brier score
        # Brier = mean((prob - outcome)^2)
        brier_scores = [(r.predicted_prob - (1 if r.won else 0))**2 for r in settled_records]
        metrics.brier_score = np.mean(brier_scores)

        # Mean Absolute Calibration Error
        # Group into probability bins and check calibration
        calibration_errors = []
        prob_bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]

        for low, high in prob_bins:
            bin_records = [r for r in settled_records if low <= r.predicted_prob < high]
            if bin_records:
                bin_pred_prob = np.mean([r.predicted_prob for r in bin_records])
                bin_realized_prob = np.mean([1 if r.won else 0 for r in bin_records])
                calibration_errors.append(abs(bin_pred_prob - bin_realized_prob))

        if calibration_errors:
            metrics.calibration_error = np.mean(calibration_errors)

        # Edge bucket analysis
        edge_buckets = {
            'low_edge': (0, 2),
            'medium_edge': (2, 5),
            'high_edge': (5, 100)
        }

        for bucket_name, (low, high) in edge_buckets.items():
            bucket_records = [r for r in settled_records if low <= r.predicted_edge < high]
            if bucket_records:
                bucket_win_rate = np.mean([1 if r.won else 0 for r in bucket_records])
                bucket_pred_prob = np.mean([r.predicted_prob for r in bucket_records])
                bucket_roi = sum(r.realized_pnl for r in bucket_records) / len(bucket_records)

                metrics.edge_buckets[bucket_name] = {
                    'count': len(bucket_records),
                    'predicted_prob': bucket_pred_prob,
                    'realized_win_rate': bucket_win_rate,
                    'avg_roi_per_bet': bucket_roi,
                    'calibration_error': abs(bucket_win_rate - bucket_pred_prob)
                }

        # Sport-specific analysis
        sports = set(r.sport for r in settled_records)
        for sport in sports:
            sport_records = [r for r in settled_records if r.sport == sport]
            if sport_records:
                sport_win_rate = np.mean([1 if r.won else 0 for r in sport_records])
                sport_pred_prob = np.mean([r.predicted_prob for r in sport_records])

                metrics.sport_metrics[sport] = {
                    'count': len(sport_records),
                    'predicted_prob': sport_pred_prob,
                    'realized_win_rate': sport_win_rate,
                    'calibration_error': abs(sport_win_rate - sport_pred_prob)
                }

        # Generate alerts
        metrics.alerts = self._generate_alerts(metrics)

        return metrics

    def _generate_alerts(self, metrics: EdgeQualityMetrics) -> List[str]:
        """Generate alerts based on metrics"""
        alerts = []

        # Alert if win rate is significantly off
        if metrics.win_rate_error > 0.05:  # 5% error
            alerts.append(
                f"⚠️ Win rate error: {metrics.win_rate_error:.1%} " +
                f"(predicted: {metrics.predicted_win_rate:.1%}, " +
                f"realized: {metrics.realized_win_rate:.1%})"
            )

        # Alert if edge predictions are way off
        if abs(metrics.edge_error) > 5.0:  # 5% edge error
            alerts.append(
                f"⚠️ Large edge error: {metrics.edge_error:.2f}% " +
                f"(predicted: {metrics.mean_predicted_edge:.2f}%, " +
                f"realized: {metrics.mean_realized_edge:.2f}%)"
            )

        # Alert if high-edge bets aren't performing
        if 'high_edge' in metrics.edge_buckets:
            high_edge = metrics.edge_buckets['high_edge']
            if high_edge['count'] >= 20:  # Enough sample size
                if high_edge['avg_roi_per_bet'] < 0:
                    alerts.append(
                        f"🚨 High-edge bets losing money! " +
                        f"ROI: ${high_edge['avg_roi_per_bet']:.2f} per bet " +
                        f"({high_edge['count']} bets)"
                    )

        # Alert if calibration is poor
        if metrics.calibration_error > 0.10:  # 10% calibration error
            alerts.append(
                f"⚠️ Poor probability calibration: {metrics.calibration_error:.1%} error"
            )

        # Alert if specific sport is miscalibrated
        for sport, sport_data in metrics.sport_metrics.items():
            if sport_data['count'] >= 20 and sport_data['calibration_error'] > 0.15:
                alerts.append(
                    f"⚠️ {sport} probabilities miscalibrated: " +
                    f"{sport_data['calibration_error']:.1%} error " +
                    f"({sport_data['count']} bets)"
                )

        return alerts

    def generate_report(self, lookback_days: Optional[int] = None) -> str:
        """Generate human-readable edge quality report"""

        metrics = self.calculate_metrics(lookback_days)

        report = []
        report.append("\n" + "="*60)
        report.append("EDGE QUALITY REPORT")
        if lookback_days:
            report.append(f"Period: Last {lookback_days} days")
        report.append("="*60)

        report.append(f"\n📊 Overall Statistics")
        report.append(f"  Total Bets: {metrics.total_bets}")
        report.append(f"  Settled Bets: {metrics.total_settled}")

        report.append(f"\n🎯 Win Rate")
        report.append(f"  Predicted: {metrics.predicted_win_rate:.1%}")
        report.append(f"  Realized: {metrics.realized_win_rate:.1%}")
        report.append(f"  Error: {metrics.win_rate_error:.1%}")

        report.append(f"\n📈 Edge")
        report.append(f"  Predicted: {metrics.mean_predicted_edge:.2f}%")
        report.append(f"  Realized: {metrics.mean_realized_edge:.2f}%")
        report.append(f"  Error: {metrics.edge_error:.2f}%")
        report.append(f"  RMSE: {metrics.edge_rmse:.2f}%")

        report.append(f"\n🎲 Probability Calibration")
        report.append(f"  Brier Score: {metrics.brier_score:.4f} (lower is better)")
        report.append(f"  Calibration Error: {metrics.calibration_error:.1%}")

        if metrics.edge_buckets:
            report.append(f"\n📊 Performance by Edge Bucket")
            for bucket_name, data in metrics.edge_buckets.items():
                report.append(f"\n  {bucket_name.upper()} ({data['count']} bets):")
                report.append(f"    Predicted Prob: {data['predicted_prob']:.1%}")
                report.append(f"    Realized Win Rate: {data['realized_win_rate']:.1%}")
                report.append(f"    Avg ROI/Bet: ${data['avg_roi_per_bet']:.2f}")
                report.append(f"    Calibration Error: {data['calibration_error']:.1%}")

        if metrics.sport_metrics:
            report.append(f"\n⚽ Performance by Sport")
            for sport, data in metrics.sport_metrics.items():
                report.append(f"\n  {sport} ({data['count']} bets):")
                report.append(f"    Predicted Prob: {data['predicted_prob']:.1%}")
                report.append(f"    Realized Win Rate: {data['realized_win_rate']:.1%}")
                report.append(f"    Calibration Error: {data['calibration_error']:.1%}")

        if metrics.alerts:
            report.append(f"\n🚨 ALERTS")
            for alert in metrics.alerts:
                report.append(f"  {alert}")
        else:
            report.append(f"\n✅ No alerts - edge quality looks good!")

        report.append("\n" + "="*60)

        return "\n".join(report)

    def get_calibration_curve_data(self, n_bins: int = 10) -> Tuple[List[float], List[float]]:
        """
        Get data for calibration curve plot
        Returns (predicted_probs, realized_probs) for each bin
        """
        settled_records = [r for r in self.records if r.won is not None]

        if not settled_records:
            return [], []

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        predicted_probs = []
        realized_probs = []

        for i in range(n_bins):
            low, high = bins[i], bins[i+1]
            bin_records = [r for r in settled_records if low <= r.predicted_prob < high]

            if bin_records:
                predicted_probs.append(np.mean([r.predicted_prob for r in bin_records]))
                realized_probs.append(np.mean([1 if r.won else 0 for r in bin_records]))

        return predicted_probs, realized_probs


if __name__ == "__main__":
    # Example usage
    monitor = EdgeQualityMonitor()

    # Simulate some bets
    print("Simulating edge quality monitoring...\n")

    for i in range(100):
        bet_id = f"bet_{i:04d}"

        # Simulate predicted edge and probability
        true_prob = np.random.beta(2, 2)  # Random true probability
        predicted_prob = true_prob + np.random.normal(0, 0.05)  # Add some calibration error
        predicted_prob = np.clip(predicted_prob, 0.01, 0.99)

        market_odds = 1 / true_prob * 1.05  # Bookmaker takes 5% margin
        predicted_edge = ((market_odds / (1/predicted_prob)) - 1) * 100

        # Record bet
        monitor.record_bet(
            bet_id=bet_id,
            predicted_edge=predicted_edge,
            predicted_prob=predicted_prob,
            market_odds=market_odds,
            sport="Soccer" if i % 3 == 0 else "Tennis",
            strategy_version="1.0.0"
        )

        # Simulate settlement
        won = np.random.random() < true_prob
        stake = 100
        realized_pnl = stake * (market_odds - 1) if won else -stake

        monitor.settle_bet(bet_id, won, realized_pnl)

    # Generate report
    print(monitor.generate_report())

    # Get calibration curve
    pred_probs, real_probs = monitor.get_calibration_curve_data()
    print(f"\nCalibration curve data points: {len(pred_probs)}")
