#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Research Pipeline
Implements segmented research stages from R&D to production.
Based on systematic trading principles from Robert Carver.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
import sqlite3
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ResearchStage:
    """Represents a stage in the alpha research pipeline."""
    name: str
    description: str
    min_samples: int
    min_duration_days: int
    success_criteria: Dict[str, float]
    next_stage: Optional[str] = None
    

@dataclass
class AlphaSignal:
    """An alpha signal under research."""
    signal_id: str
    name: str
    description: str
    formula: str
    parameters: Dict[str, Any]
    created_at: datetime
    current_stage: str
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    
@dataclass
class BacktestResult:
    """Results from a backtest run."""
    signal_id: str
    stage: str
    start_date: datetime
    end_date: datetime
    n_bets: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_edge: float
    information_ratio: float
    t_statistic: float
    p_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlphaResearchPipeline:
    """
    Manages the complete alpha research lifecycle.
    
    Stages:
    1. Raw R&D - Initial signal exploration
    2. In-Sample Testing - Parameter optimization
    3. Out-of-Sample Testing - Validation on unseen data
    4. Walk-Forward Analysis - Rolling window testing
    5. Paper Trading - Real-time simulation
    6. Live Trading - Production deployment
    """
    
    RESEARCH_STAGES = {
        'raw_rd': ResearchStage(
            name='raw_rd',
            description='Initial signal exploration and hypothesis testing',
            min_samples=1000,
            min_duration_days=30,
            success_criteria={
                'sharpe_ratio': 0.5,
                'win_rate': 0.52,
                'p_value': 0.05
            },
            next_stage='in_sample'
        ),
        'in_sample': ResearchStage(
            name='in_sample',
            description='Parameter optimization on training data',
            min_samples=5000,
            min_duration_days=90,
            success_criteria={
                'sharpe_ratio': 1.0,
                'win_rate': 0.53,
                'information_ratio': 0.5,
                'p_value': 0.01
            },
            next_stage='out_of_sample'
        ),
        'out_of_sample': ResearchStage(
            name='out_of_sample',
            description='Validation on unseen test data',
            min_samples=2000,
            min_duration_days=60,
            success_criteria={
                'sharpe_ratio': 0.8,
                'win_rate': 0.52,
                'degradation': 0.2  # Max performance drop vs in-sample
            },
            next_stage='walk_forward'
        ),
        'walk_forward': ResearchStage(
            name='walk_forward',
            description='Rolling window analysis for stability',
            min_samples=10000,
            min_duration_days=180,
            success_criteria={
                'sharpe_ratio': 0.7,
                'consistency': 0.7,  # Fraction of windows meeting criteria
                'stability': 0.3     # Max variance in performance
            },
            next_stage='paper_trading'
        ),
        'paper_trading': ResearchStage(
            name='paper_trading',
            description='Live simulation with real-time data',
            min_samples=500,
            min_duration_days=30,
            success_criteria={
                'sharpe_ratio': 0.5,
                'tracking_error': 0.1,  # vs backtest
                'execution_quality': 0.95
            },
            next_stage='live_trading'
        ),
        'live_trading': ResearchStage(
            name='live_trading',
            description='Production deployment with real capital',
            min_samples=100,
            min_duration_days=14,
            success_criteria={
                'sharpe_ratio': 0.5,
                'tracking_error': 0.15,
                'risk_limits': 1.0
            },
            next_stage=None
        )
    }
    
    def __init__(self, db_path: str = "alpha_research.db"):
        self.db_path = db_path
        self.signals: Dict[str, AlphaSignal] = {}
        self._init_database()
        self._load_signals()
        
    def _init_database(self):
        """Initialize research database."""
        conn = sqlite3.connect(self.db_path)
        
        # Signals table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alpha_signals (
                signal_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                formula TEXT,
                parameters TEXT,
                created_at DATETIME,
                current_stage TEXT,
                stage_history TEXT,
                performance_metrics TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Backtest results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                start_date DATETIME,
                end_date DATETIME,
                n_bets INTEGER,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_edge REAL,
                information_ratio REAL,
                t_statistic REAL,
                p_value REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES alpha_signals(signal_id)
            )
        """)
        
        # Stage transitions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                from_stage TEXT,
                to_stage TEXT,
                transition_date DATETIME,
                reason TEXT,
                metrics TEXT,
                approved_by TEXT,
                FOREIGN KEY (signal_id) REFERENCES alpha_signals(signal_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def register_signal(self, signal: AlphaSignal):
        """Register a new alpha signal for research."""
        logger.info(f"Registering new signal: {signal.name}")
        
        self.signals[signal.signal_id] = signal
        self._persist_signal(signal)
        
        # Initialize in raw R&D stage
        self._transition_stage(signal.signal_id, None, 'raw_rd', 
                            "Initial registration")
        
    def _persist_signal(self, signal: AlphaSignal):
        """Save signal to database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO alpha_signals (
                signal_id, name, description, formula, parameters,
                created_at, current_stage, stage_history, performance_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.signal_id,
            signal.name,
            signal.description,
            signal.formula,
            json.dumps(signal.parameters),
            signal.created_at,
            signal.current_stage,
            json.dumps(signal.stage_history),
            json.dumps(signal.performance_metrics)
        ))
        
        conn.commit()
        conn.close()
        
    def run_backtest(self, signal_id: str, stage: str,
                    data: pd.DataFrame, 
                    start_date: datetime,
                    end_date: datetime) -> BacktestResult:
        """Run backtest for a signal at a specific stage."""
        signal = self.signals.get(signal_id)
        if not signal:
            raise ValueError(f"Unknown signal: {signal_id}")
            
        logger.info(f"Running {stage} backtest for {signal.name}")
        
        # Filter data to date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        test_data = data[mask].copy()
        
        # Generate predictions
        predictions = self._generate_predictions(signal, test_data)
        
        # Calculate betting decisions
        bets = self._calculate_bets(predictions, test_data)
        
        # Evaluate performance
        metrics = self._evaluate_performance(bets, test_data)
        
        # Statistical tests
        stats_results = self._run_statistical_tests(bets, test_data)
        
        result = BacktestResult(
            signal_id=signal_id,
            stage=stage,
            start_date=start_date,
            end_date=end_date,
            n_bets=len(bets),
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            avg_edge=metrics['avg_edge'],
            information_ratio=metrics.get('information_ratio', 0),
            t_statistic=stats_results['t_statistic'],
            p_value=stats_results['p_value'],
            metadata={
                'prediction_stats': self._get_prediction_stats(predictions),
                'bet_stats': self._get_bet_stats(bets)
            }
        )
        
        # Store result
        self._store_backtest_result(result)
        
        # Update signal metrics
        if stage not in signal.performance_metrics:
            signal.performance_metrics[stage] = {}
        signal.performance_metrics[stage].update(metrics)
        self._persist_signal(signal)
        
        return result
        
    def _generate_predictions(self, signal: AlphaSignal, 
                            data: pd.DataFrame) -> pd.Series:
        """Generate predictions using signal formula."""
        # This would be replaced with actual signal implementation
        # For now, simulate with random predictions
        np.random.seed(hash(signal.signal_id) % 2**32)
        
        # Simple momentum signal simulation
        returns = data['close'].pct_change()
        momentum = returns.rolling(20).mean()
        
        # Convert to probabilities
        z_scores = (momentum - momentum.mean()) / momentum.std()
        probabilities = stats.norm.cdf(z_scores * 0.5 + 0.5)
        
        return probabilities
        
    def _calculate_bets(self, predictions: pd.Series, 
                       data: pd.DataFrame) -> pd.DataFrame:
        """Calculate betting decisions from predictions."""
        bets = pd.DataFrame(index=predictions.index)
        
        # Kelly criterion for sizing
        bets['probability'] = predictions
        bets['odds'] = data['odds']
        bets['edge'] = predictions - (1 / data['odds'])
        
        # Only bet on positive edge
        bets['bet_size'] = 0.0
        positive_edge = bets['edge'] > 0
        
        if positive_edge.any():
            # Simple Kelly
            bets.loc[positive_edge, 'bet_size'] = (
                bets.loc[positive_edge, 'edge'] / 
                (bets.loc[positive_edge, 'odds'] - 1)
            ).clip(0, 0.25)  # Cap at 25% of capital
            
        return bets[bets['bet_size'] > 0]
        
    def _evaluate_performance(self, bets: pd.DataFrame, 
                            data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate backtest performance."""
        if len(bets) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_edge': 0
            }
            
        # Simulate outcomes (would use real data in practice)
        np.random.seed(42)
        outcomes = np.random.random(len(bets)) < bets['probability']
        
        # Calculate returns
        returns = []
        for idx, bet in bets.iterrows():
            if outcomes[len(returns)]:
                ret = bet['bet_size'] * (bet['odds'] - 1)
            else:
                ret = -bet['bet_size']
            returns.append(ret)
            
        returns = pd.Series(returns, index=bets.index)
        
        # Calculate metrics
        total_return = returns.sum()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
            
        cumulative = (1 + returns).cumprod()
        drawdowns = (cumulative / cumulative.cummax() - 1)
        max_drawdown = drawdowns.min()
        
        win_rate = (returns > 0).mean()
        avg_edge = bets['edge'].mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_edge': avg_edge,
            'n_bets': len(bets),
            'avg_bet_size': bets['bet_size'].mean()
        }
        
    def _run_statistical_tests(self, bets: pd.DataFrame, 
                             data: pd.DataFrame) -> Dict[str, float]:
        """Run statistical significance tests."""
        if len(bets) < 30:
            return {'t_statistic': 0, 'p_value': 1}
            
        # Test if edge is significantly positive
        edges = bets['edge'].values
        t_stat, p_value = stats.ttest_1samp(edges, 0, alternative='greater')
        
        return {
            't_statistic': t_stat,
            'p_value': p_value
        }
        
    def _get_prediction_stats(self, predictions: pd.Series) -> Dict[str, float]:
        """Get statistics about predictions."""
        return {
            'mean': predictions.mean(),
            'std': predictions.std(),
            'skew': predictions.skew(),
            'kurtosis': predictions.kurtosis()
        }
        
    def _get_bet_stats(self, bets: pd.DataFrame) -> Dict[str, float]:
        """Get statistics about betting decisions."""
        if len(bets) == 0:
            return {}
            
        return {
            'n_bets': len(bets),
            'avg_size': bets['bet_size'].mean(),
            'max_size': bets['bet_size'].max(),
            'avg_edge': bets['edge'].mean(),
            'edge_std': bets['edge'].std()
        }
        
    def _store_backtest_result(self, result: BacktestResult):
        """Store backtest result in database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO backtest_results (
                signal_id, stage, start_date, end_date, n_bets,
                total_return, sharpe_ratio, max_drawdown, win_rate,
                avg_edge, information_ratio, t_statistic, p_value, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.signal_id,
            result.stage,
            result.start_date,
            result.end_date,
            result.n_bets,
            result.total_return,
            result.sharpe_ratio,
            result.max_drawdown,
            result.win_rate,
            result.avg_edge,
            result.information_ratio,
            result.t_statistic,
            result.p_value,
            json.dumps(result.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def evaluate_stage_progression(self, signal_id: str) -> Tuple[bool, str]:
        """Evaluate if signal should progress to next stage."""
        signal = self.signals.get(signal_id)
        if not signal:
            return False, "Unknown signal"
            
        current_stage = self.RESEARCH_STAGES[signal.current_stage]
        
        # Get recent results for current stage
        results = self._get_stage_results(signal_id, signal.current_stage)
        
        if len(results) < 3:
            return False, f"Need at least 3 backtests (have {len(results)})"
            
        # Check minimum requirements
        total_samples = sum(r['n_bets'] for r in results)
        if total_samples < current_stage.min_samples:
            return False, f"Need {current_stage.min_samples} samples (have {total_samples})"
            
        # Check duration
        first_date = min(r['start_date'] for r in results)
        last_date = max(r['end_date'] for r in results)
        duration_days = (last_date - first_date).days
        
        if duration_days < current_stage.min_duration_days:
            return False, f"Need {current_stage.min_duration_days} days (have {duration_days})"
            
        # Check success criteria
        avg_metrics = self._calculate_average_metrics(results)
        
        for metric, threshold in current_stage.success_criteria.items():
            if metric == 'degradation':
                # Special handling for out-of-sample degradation
                if 'in_sample' in signal.performance_metrics:
                    in_sample_sharpe = signal.performance_metrics['in_sample']['sharpe_ratio']
                    degradation = 1 - (avg_metrics['sharpe_ratio'] / in_sample_sharpe)
                    if degradation > threshold:
                        return False, f"Degradation too high: {degradation:.2f} > {threshold}"
            elif metric == 'consistency':
                # Fraction of windows meeting criteria
                good_windows = sum(1 for r in results if r['sharpe_ratio'] > 0.5)
                consistency = good_windows / len(results)
                if consistency < threshold:
                    return False, f"Consistency too low: {consistency:.2f} < {threshold}"
            elif metric == 'stability':
                # Variance in performance
                sharpes = [r['sharpe_ratio'] for r in results]
                stability = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) > 0 else 1
                if stability > threshold:
                    return False, f"Stability too low: {stability:.2f} > {threshold}"
            else:
                # Standard threshold check
                if avg_metrics.get(metric, 0) < threshold:
                    return False, f"{metric} too low: {avg_metrics.get(metric, 0):.3f} < {threshold}"
                    
        return True, "All criteria met"
        
    def _get_stage_results(self, signal_id: str, stage: str) -> List[Dict]:
        """Get backtest results for a stage."""
        conn = sqlite3.connect(self.db_path)
        
        results = conn.execute("""
            SELECT * FROM backtest_results
            WHERE signal_id = ? AND stage = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (signal_id, stage)).fetchall()
        
        conn.close()
        
        # Convert to dicts
        columns = ['signal_id', 'stage', 'start_date', 'end_date', 'n_bets',
                   'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
                   'avg_edge', 'information_ratio', 't_statistic', 'p_value']
        
        return [dict(zip(columns, r[1:14])) for r in results]
        
    def _calculate_average_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics across results."""
        metrics = {}
        
        for key in ['sharpe_ratio', 'win_rate', 'avg_edge', 'p_value']:
            values = [r[key] for r in results if r[key] is not None]
            metrics[key] = np.mean(values) if values else 0
            
        # Information ratio (if available)
        ir_values = [r.get('information_ratio', 0) for r in results]
        metrics['information_ratio'] = np.mean(ir_values)
        
        return metrics
        
    def promote_signal(self, signal_id: str, reason: str = ""):
        """Promote signal to next stage."""
        signal = self.signals.get(signal_id)
        if not signal:
            raise ValueError(f"Unknown signal: {signal_id}")
            
        current_stage = self.RESEARCH_STAGES[signal.current_stage]
        if not current_stage.next_stage:
            logger.warning(f"Signal {signal_id} already at final stage")
            return
            
        # Check if ready
        can_promote, check_reason = self.evaluate_stage_progression(signal_id)
        if not can_promote:
            raise ValueError(f"Cannot promote: {check_reason}")
            
        # Perform transition
        self._transition_stage(signal_id, signal.current_stage, 
                             current_stage.next_stage, reason)
        
    def _transition_stage(self, signal_id: str, from_stage: Optional[str],
                        to_stage: str, reason: str):
        """Transition signal between stages."""
        signal = self.signals.get(signal_id)
        if signal:
            signal.current_stage = to_stage
            signal.stage_history.append({
                'from': from_stage,
                'to': to_stage,
                'date': datetime.now(timezone.utc).isoformat(),
                'reason': reason
            })
            self._persist_signal(signal)
            
        # Record transition
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO stage_transitions (
                signal_id, from_stage, to_stage, transition_date, reason
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            signal_id,
            from_stage,
            to_stage,
            datetime.now(timezone.utc),
            reason
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Transitioned {signal_id} from {from_stage} to {to_stage}: {reason}")
        
    def run_walk_forward_analysis(self, signal_id: str, 
                                data: pd.DataFrame,
                                window_days: int = 90,
                                step_days: int = 30) -> List[BacktestResult]:
        """Run walk-forward analysis with rolling windows."""
        logger.info(f"Running walk-forward analysis for {signal_id}")
        
        results = []
        
        # Set up rolling windows
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        while current_start + timedelta(days=window_days) <= end_date:
            window_end = current_start + timedelta(days=window_days)
            
            # Split into train/test
            split_point = current_start + timedelta(days=int(window_days * 0.7))
            
            # Train on first 70%
            train_result = self.run_backtest(
                signal_id, 'walk_forward_train',
                data, current_start, split_point
            )
            
            # Test on last 30%
            test_result = self.run_backtest(
                signal_id, 'walk_forward_test',
                data, split_point, window_end
            )
            
            results.append(test_result)
            
            # Move window
            current_start += timedelta(days=step_days)
            
        return results
        
    def generate_research_report(self, signal_id: str) -> Dict[str, Any]:
        """Generate comprehensive research report for a signal."""
        signal = self.signals.get(signal_id)
        if not signal:
            return {}
            
        report = {
            'signal': {
                'id': signal.signal_id,
                'name': signal.name,
                'description': signal.description,
                'formula': signal.formula,
                'parameters': signal.parameters,
                'current_stage': signal.current_stage,
                'created_at': signal.created_at.isoformat()
            },
            'performance_by_stage': signal.performance_metrics,
            'stage_history': signal.stage_history,
            'backtest_summary': self._summarize_backtests(signal_id),
            'statistical_analysis': self._statistical_analysis(signal_id),
            'risk_analysis': self._risk_analysis(signal_id)
        }
        
        return report
        
    def _summarize_backtests(self, signal_id: str) -> Dict[str, Any]:
        """Summarize all backtests for a signal."""
        conn = sqlite3.connect(self.db_path)
        
        # Get summary by stage
        summary = conn.execute("""
            SELECT 
                stage,
                COUNT(*) as n_tests,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(win_rate) as avg_win_rate,
                SUM(n_bets) as total_bets,
                AVG(p_value) as avg_p_value
            FROM backtest_results
            WHERE signal_id = ?
            GROUP BY stage
        """, (signal_id,)).fetchall()
        
        conn.close()
        
        return {
            row[0]: {
                'n_tests': row[1],
                'avg_sharpe': row[2],
                'avg_win_rate': row[3],
                'total_bets': row[4],
                'avg_p_value': row[5]
            }
            for row in summary
        }
        
    def _statistical_analysis(self, signal_id: str) -> Dict[str, Any]:
        """Perform statistical analysis of signal performance."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all results
        results = pd.read_sql_query("""
            SELECT * FROM backtest_results
            WHERE signal_id = ?
        """, conn, params=(signal_id,))
        
        conn.close()
        
        if results.empty:
            return {}
            
        return {
            'sharpe_distribution': {
                'mean': results['sharpe_ratio'].mean(),
                'std': results['sharpe_ratio'].std(),
                'percentiles': results['sharpe_ratio'].quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'edge_persistence': self._calculate_edge_persistence(results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(results)
        }
        
    def _calculate_edge_persistence(self, results: pd.DataFrame) -> float:
        """Calculate how persistent the edge is over time."""
        if len(results) < 2:
            return 0
            
        # Sort by date and calculate autocorrelation
        results = results.sort_values('start_date')
        return results['sharpe_ratio'].autocorr()
        
    def _analyze_parameter_sensitivity(self, results: pd.DataFrame) -> Dict:
        """Analyze sensitivity to parameter changes."""
        # This would analyze how performance varies with parameters
        # For now, return placeholder
        return {
            'stable_parameters': True,
            'optimal_ranges': {}
        }
        
    def _risk_analysis(self, signal_id: str) -> Dict[str, Any]:
        """Perform risk analysis for the signal."""
        conn = sqlite3.connect(self.db_path)
        
        results = pd.read_sql_query("""
            SELECT * FROM backtest_results
            WHERE signal_id = ?
        """, conn, params=(signal_id,))
        
        conn.close()
        
        if results.empty:
            return {}
            
        return {
            'max_drawdown': {
                'worst': results['max_drawdown'].min(),
                'average': results['max_drawdown'].mean(),
                'percentile_95': results['max_drawdown'].quantile(0.05)
            },
            'var_95': self._calculate_var(results, 0.95),
            'expected_shortfall': self._calculate_expected_shortfall(results, 0.95),
            'stress_scenarios': self._analyze_stress_scenarios(results)
        }
        
    def _calculate_var(self, results: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk."""
        returns = results['total_return'].values
        return np.percentile(returns, (1 - confidence) * 100)
        
    def _calculate_expected_shortfall(self, results: pd.DataFrame, 
                                    confidence: float) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        returns = results['total_return'].values
        var = self._calculate_var(results, confidence)
        return returns[returns <= var].mean()
        
    def _analyze_stress_scenarios(self, results: pd.DataFrame) -> Dict:
        """Analyze performance in stress scenarios."""
        # Identify worst performing periods
        worst_periods = results.nsmallest(3, 'sharpe_ratio')
        
        return {
            'worst_periods': worst_periods[['start_date', 'sharpe_ratio', 'max_drawdown']].to_dict('records'),
            'recovery_time': 'Not implemented'  # Would calculate drawdown recovery
        }
        
    def update_signal_performance(self, signal_name: str, metrics: Dict):
        """Update performance metrics for a signal (simplified version)."""
        # For now, just log the update
        logger.info(f"Performance update for {signal_name}: {metrics}")
        
        # If we have a matching signal by name, update its metrics
        for signal_id, signal in self.signals.items():
            if signal.name == signal_name:
                if 'paper_trading' not in signal.performance_metrics:
                    signal.performance_metrics['paper_trading'] = {}
                signal.performance_metrics['paper_trading'].update(metrics)
                self._persist_signal(signal)
                break
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate simplified report for unified system."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'signals': {
                signal_id: {
                    'name': signal.name,
                    'stage': signal.current_stage,
                    'performance': signal.performance_metrics
                }
                for signal_id, signal in self.signals.items()
            },
            'summary': 'Alpha research pipeline active'
        }
        
    def _load_signals(self):
        """Load signals from database."""
        conn = sqlite3.connect(self.db_path)
        
        signals = conn.execute("""
            SELECT * FROM alpha_signals WHERE is_active = 1
        """).fetchall()
        
        conn.close()
        
        for row in signals:
            signal = AlphaSignal(
                signal_id=row[0],
                name=row[1],
                description=row[2],
                formula=row[3],
                parameters=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                current_stage=row[6],
                stage_history=json.loads(row[7]),
                performance_metrics=json.loads(row[8])
            )
            self.signals[signal.signal_id] = signal


def main():
    """Example usage of alpha research pipeline."""
    # Initialize pipeline
    pipeline = AlphaResearchPipeline()
    
    # Create a new signal
    signal = AlphaSignal(
        signal_id="momentum_v1",
        name="Simple Momentum",
        description="Basic momentum signal using price returns",
        formula="rolling_mean(returns, 20)",
        parameters={'window': 20, 'threshold': 0.01},
        created_at=datetime.now(timezone.utc),
        current_stage='raw_rd'
    )
    
    # Register signal
    pipeline.register_signal(signal)
    
    # Generate some dummy data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='H')
    data = pd.DataFrame({
        'close': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
        'odds': 2 + np.random.random(len(dates))
    }, index=dates)
    
    # Run initial backtest
    result = pipeline.run_backtest(
        signal.signal_id, 
        'raw_rd',
        data,
        datetime(2024, 1, 1),
        datetime(2024, 3, 31)
    )
    
    print(f"Backtest complete: Sharpe={result.sharpe_ratio:.2f}, "
          f"Win Rate={result.win_rate:.2%}")
    
    # Check if ready to promote
    can_promote, reason = pipeline.evaluate_stage_progression(signal.signal_id)
    print(f"Can promote: {can_promote} ({reason})")
    
    # Generate report
    report = pipeline.generate_research_report(signal.signal_id)
    print(f"\nResearch Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()