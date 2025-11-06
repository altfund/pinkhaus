#!/usr/bin/env python3
"""
Settlement Analyzer - Empirical Settlement Pattern Analysis

Analyzes actual settlement patterns to provide empirical data for:
- Match duration tracking (actual vs expected)
- Settlement delay patterns (blockchain confirmation times)
- Capital recycling optimization
- Risk model calibration

Feeds data back to dynamic_chunk_manager.py for continuous improvement.
"""

import os
import json
import logging
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MatchSettlementData:
    """Individual match settlement analysis"""
    match_id: str
    sport: str
    league: str
    
    # Timing data
    scheduled_start: datetime
    actual_start: Optional[datetime] = None
    expected_end: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    settlement_time: Optional[datetime] = None
    
    # Duration analysis
    scheduled_duration_minutes: float = 105  # Default soccer
    actual_duration_minutes: Optional[float] = None
    settlement_delay_minutes: Optional[float] = None
    
    # Performance metrics
    total_positions: int = 0
    total_stakes: float = 0.0
    total_pnl: float = 0.0
    
    # Settlement status
    is_settled: bool = False
    settlement_method: str = 'unknown'  # blockchain, manual, api
    
    def calculate_timings(self):
        """Calculate actual durations and delays"""
        if self.actual_start and self.actual_end:
            self.actual_duration_minutes = (self.actual_end - self.actual_start).total_seconds() / 60
        
        if self.actual_end and self.settlement_time:
            self.settlement_delay_minutes = (self.settlement_time - self.actual_end).total_seconds() / 60

@dataclass
class LeagueSettlementStats:
    """Settlement statistics for a sport/league combination"""
    sport: str
    league: str
    
    # Duration statistics
    avg_duration_minutes: float = 0.0
    median_duration_minutes: float = 0.0
    duration_std: float = 0.0
    duration_samples: int = 0
    
    # Settlement delay statistics
    avg_settlement_delay_minutes: float = 0.0
    median_settlement_delay_minutes: float = 0.0
    settlement_delay_std: float = 0.0
    settlement_samples: int = 0
    
    # Reliability metrics
    on_time_settlement_rate: float = 0.0  # % settled within expected time
    delayed_settlement_rate: float = 0.0  # % with significant delays
    
    # Performance correlation
    avg_positions_per_match: float = 0.0
    avg_pnl_per_match: float = 0.0
    
    # Timing recommendations
    recommended_duration_minutes: float = 0.0
    recommended_settlement_minutes: float = 0.0
    recommended_gap_minutes: float = 0.0
    
    def calculate_recommendations(self):
        """Calculate recommended timing parameters"""
        # Use 75th percentile for duration (accounts for delays)
        self.recommended_duration_minutes = self.avg_duration_minutes + (0.5 * self.duration_std)
        
        # Use 90th percentile for settlement (conservative estimate)
        self.recommended_settlement_minutes = self.avg_settlement_delay_minutes + (1.0 * self.settlement_delay_std)
        
        # Recommended gap = settlement buffer + risk management buffer
        self.recommended_gap_minutes = max(15, self.recommended_settlement_minutes + 10)


class SettlementAnalyzer:
    """Analyzes settlement patterns and provides empirical timing data"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {
            'host': os.environ.get('PG_HOST', 'localhost'),
            'port': os.environ.get('PG_PORT', '5999'),
            'user': os.environ.get('PG_USER', 'ominari_user'),
            'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.environ.get('PG_DB', 'ominari_production')
        }
        
        # Analysis configuration
        self.analysis_config = {
            'min_samples_for_stats': 5,      # Minimum matches needed for statistics
            'outlier_threshold': 3.0,        # Standard deviations for outlier detection
            'lookback_days': 90,             # Days of history to analyze
            'update_interval_hours': 6,      # How often to refresh statistics
        }
        
        # Cached statistics
        self.league_stats: Dict[str, LeagueSettlementStats] = {}
        self.last_update: Optional[datetime] = None
        
        # Settlement pattern tracking
        self.settlement_events: List[Dict] = []
        
    def analyze_historical_patterns(self, lookback_days: int = None) -> Dict[str, LeagueSettlementStats]:
        """Analyze historical settlement patterns"""
        lookback_days = lookback_days or self.analysis_config['lookback_days']
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        logger.info(f"Analyzing settlement patterns for last {lookback_days} days")
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Get settled matches with position data
                    cur.execute("""
                        SELECT 
                            p.match_id,
                            p.sport,
                            p.league,
                            p.start_time,
                            p.maturity_date,
                            MIN(pos.placed_at) as first_position,
                            MAX(pos.settled_at) as last_settlement,
                            COUNT(pos.bet_id) as position_count,
                            SUM(pos.stake) as total_stakes,
                            SUM(pos.pnl) as total_pnl,
                            -- Try to infer actual match end from earliest settlement
                            MIN(pos.settled_at) as inferred_match_end
                        FROM paper_trading_positions pos
                        JOIN (
                            SELECT DISTINCT 
                                match_id,
                                COALESCE(
                                    strategy_config->>'sport', 
                                    CASE 
                                        WHEN match_id LIKE '%soccer%' THEN 'soccer'
                                        WHEN match_id LIKE '%football%' THEN 'soccer'
                                        ELSE 'unknown'
                                    END
                                ) as sport,
                                COALESCE(
                                    strategy_config->>'league',
                                    'Unknown League'
                                ) as league,
                                MIN(placed_at) as start_time,
                                MIN(placed_at) + INTERVAL '105 minutes' as maturity_date
                            FROM paper_trading_positions
                            WHERE settled_at IS NOT NULL
                            AND placed_at >= %s
                            GROUP BY match_id, sport, league
                        ) p ON p.match_id = pos.match_id
                        WHERE pos.settled_at IS NOT NULL
                        AND pos.placed_at >= %s
                        GROUP BY p.match_id, p.sport, p.league, p.start_time, p.maturity_date
                        HAVING COUNT(pos.bet_id) > 0
                        ORDER BY p.sport, p.league, p.start_time
                    """, (cutoff_date, cutoff_date))
                    
                    matches_data = cur.fetchall()
                    
                    if not matches_data:
                        logger.warning("No settled matches found for analysis")
                        return {}
                    
                    logger.info(f"Found {len(matches_data)} settled matches for analysis")
                    
                    # Process matches into settlement data
                    settlement_data = []
                    for row in matches_data:
                        (match_id, sport, league, start_time, maturity_date, 
                         first_position, last_settlement, position_count, 
                         total_stakes, total_pnl, inferred_match_end) = row
                        
                        match_data = MatchSettlementData(
                            match_id=match_id,
                            sport=sport or 'unknown',
                            league=league or 'Unknown',
                            scheduled_start=start_time,
                            actual_start=start_time,  # Assume start time is accurate
                            expected_end=maturity_date,
                            actual_end=inferred_match_end,
                            settlement_time=last_settlement,
                            total_positions=position_count,
                            total_stakes=float(total_stakes or 0),
                            total_pnl=float(total_pnl or 0),
                            is_settled=True,
                            settlement_method='blockchain'
                        )
                        
                        match_data.calculate_timings()
                        settlement_data.append(match_data)
                    
                    # Calculate league statistics
                    league_stats = self._calculate_league_statistics(settlement_data)
                    
                    # Cache results
                    self.league_stats = league_stats
                    self.last_update = datetime.now(timezone.utc)
                    
                    # Save to file for dynamic chunk manager
                    self._save_empirical_data(league_stats)
                    
                    logger.info(f"Analyzed {len(settlement_data)} matches across {len(league_stats)} sport/league combinations")
                    
                    return league_stats
                    
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _calculate_league_statistics(self, settlement_data: List[MatchSettlementData]) -> Dict[str, LeagueSettlementStats]:
        """Calculate statistics by sport/league combination"""
        
        # Group by sport/league
        league_groups = defaultdict(list)
        for match in settlement_data:
            key = f"{match.sport}_{match.league}"
            league_groups[key].append(match)
        
        league_stats = {}
        
        for league_key, matches in league_groups.items():
            if len(matches) < self.analysis_config['min_samples_for_stats']:
                logger.warning(f"Insufficient data for {league_key}: {len(matches)} matches")
                continue
            
            sport, league = league_key.split('_', 1)
            
            # Extract timing data (filter out None values)
            durations = [m.actual_duration_minutes for m in matches if m.actual_duration_minutes is not None]
            settlement_delays = [m.settlement_delay_minutes for m in matches if m.settlement_delay_minutes is not None]
            
            if not durations or not settlement_delays:
                logger.warning(f"No valid timing data for {league_key}")
                continue
            
            # Remove outliers
            durations = self._remove_outliers(durations)
            settlement_delays = self._remove_outliers(settlement_delays)
            
            if not durations or not settlement_delays:
                logger.warning(f"No data after outlier removal for {league_key}")
                continue
            
            # Calculate statistics
            stats = LeagueSettlementStats(
                sport=sport,
                league=league,
                
                # Duration stats
                avg_duration_minutes=statistics.mean(durations),
                median_duration_minutes=statistics.median(durations),
                duration_std=statistics.stdev(durations) if len(durations) > 1 else 0,
                duration_samples=len(durations),
                
                # Settlement delay stats
                avg_settlement_delay_minutes=statistics.mean(settlement_delays),
                median_settlement_delay_minutes=statistics.median(settlement_delays),
                settlement_delay_std=statistics.stdev(settlement_delays) if len(settlement_delays) > 1 else 0,
                settlement_samples=len(settlement_delays),
                
                # Performance metrics
                avg_positions_per_match=statistics.mean([m.total_positions for m in matches]),
                avg_pnl_per_match=statistics.mean([m.total_pnl for m in matches]),
            )
            
            # Calculate on-time settlement rate
            expected_settlement_time = stats.avg_duration_minutes + 15  # 15 min expected delay
            on_time_count = sum(1 for d in settlement_delays if d <= expected_settlement_time)
            stats.on_time_settlement_rate = on_time_count / len(settlement_delays) if settlement_delays else 0
            
            # Calculate delayed settlement rate (>30 min delay)
            delayed_count = sum(1 for d in settlement_delays if d > 30)
            stats.delayed_settlement_rate = delayed_count / len(settlement_delays) if settlement_delays else 0
            
            # Generate recommendations
            stats.calculate_recommendations()
            
            league_stats[league_key] = stats
            
            logger.info(f"{league_key}: {len(matches)} matches, "
                       f"avg duration {stats.avg_duration_minutes:.1f}min, "
                       f"avg settlement {stats.avg_settlement_delay_minutes:.1f}min")
        
        return league_stats
    
    def _remove_outliers(self, values: List[float]) -> List[float]:
        """Remove statistical outliers using z-score method"""
        if len(values) < 3:
            return values
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return values
        
        # Remove values more than N standard deviations from mean
        threshold = self.analysis_config['outlier_threshold']
        filtered_values = [
            v for v in values 
            if abs(v - mean_val) <= threshold * std_val
        ]
        
        outliers_removed = len(values) - len(filtered_values)
        if outliers_removed > 0:
            logger.debug(f"Removed {outliers_removed} outliers from {len(values)} values")
        
        return filtered_values
    
    def _save_empirical_data(self, league_stats: Dict[str, LeagueSettlementStats]):
        """Save empirical data for dynamic chunk manager"""
        empirical_data = {}
        
        for league_key, stats in league_stats.items():
            empirical_data[league_key] = {
                'duration_minutes': stats.recommended_duration_minutes,
                'settlement_minutes': stats.recommended_settlement_minutes,
                'min_gap_minutes': stats.recommended_gap_minutes,
                'on_time_rate': stats.on_time_settlement_rate,
                'sample_count': stats.duration_samples,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        
        try:
            with open('settlement_patterns.json', 'w') as f:
                json.dump(empirical_data, f, indent=2)
            logger.info(f"Saved empirical data for {len(empirical_data)} league combinations")
        except Exception as e:
            logger.error(f"Failed to save empirical data: {e}")
    
    def get_timing_recommendations(self, sport: str, league: str) -> Dict[str, float]:
        """Get timing recommendations for a specific sport/league"""
        league_key = f"{sport}_{league}"
        
        # Check if we have recent data
        if (not self.last_update or 
            (datetime.now(timezone.utc) - self.last_update).total_seconds() > 
            self.analysis_config['update_interval_hours'] * 3600):
            logger.info("Refreshing settlement analysis...")
            self.analyze_historical_patterns()
        
        # Return specific league stats if available
        if league_key in self.league_stats:
            stats = self.league_stats[league_key]
            return {
                'duration_minutes': stats.recommended_duration_minutes,
                'settlement_minutes': stats.recommended_settlement_minutes,
                'min_gap_minutes': stats.recommended_gap_minutes,
                'confidence': min(1.0, stats.duration_samples / 20),  # Confidence based on sample size
                'on_time_rate': stats.on_time_settlement_rate
            }
        
        # Fallback to sport-level aggregation
        sport_matches = [stats for key, stats in self.league_stats.items() if key.startswith(f"{sport}_")]
        if sport_matches:
            avg_duration = statistics.mean([s.recommended_duration_minutes for s in sport_matches])
            avg_settlement = statistics.mean([s.recommended_settlement_minutes for s in sport_matches])
            avg_gap = statistics.mean([s.recommended_gap_minutes for s in sport_matches])
            avg_on_time = statistics.mean([s.on_time_settlement_rate for s in sport_matches])
            
            return {
                'duration_minutes': avg_duration,
                'settlement_minutes': avg_settlement,
                'min_gap_minutes': avg_gap,
                'confidence': 0.5,  # Medium confidence for aggregated data
                'on_time_rate': avg_on_time
            }
        
        # Default fallback
        defaults = {
            'soccer': {'duration': 105, 'settlement': 15, 'gap': 20},
            'basketball': {'duration': 150, 'settlement': 10, 'gap': 25},
            'tennis': {'duration': 180, 'settlement': 10, 'gap': 30}
        }
        
        default = defaults.get(sport.lower(), defaults['soccer'])
        return {
            'duration_minutes': default['duration'],
            'settlement_minutes': default['settlement'],
            'min_gap_minutes': default['gap'],
            'confidence': 0.1,  # Low confidence for defaults
            'on_time_rate': 0.8   # Assumed
        }
    
    def track_live_settlement(self, match_id: str, event_type: str, event_data: Dict):
        """Track real-time settlement events for live learning"""
        settlement_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'match_id': match_id,
            'event_type': event_type,  # match_start, match_end, first_settlement, full_settlement
            'event_data': event_data
        }
        
        self.settlement_events.append(settlement_event)
        
        # Log to file for historical analysis
        try:
            log_file = f"settlement_events_{datetime.now().strftime('%Y%m%d')}.json"
            with open(log_file, 'a') as f:
                f.write(json.dumps(settlement_event) + '\n')
        except Exception as e:
            logger.error(f"Failed to log settlement event: {e}")
        
        logger.info(f"Tracked settlement event: {event_type} for match {match_id}")
    
    def get_settlement_report(self) -> Dict[str, Any]:
        """Generate comprehensive settlement analysis report"""
        if not self.league_stats:
            self.analyze_historical_patterns()
        
        # Overall statistics
        all_stats = list(self.league_stats.values())
        if not all_stats:
            return {'error': 'No settlement data available'}
        
        overall_stats = {
            'total_leagues_analyzed': len(all_stats),
            'total_matches_analyzed': sum(s.duration_samples for s in all_stats),
            'avg_duration_across_sports': statistics.mean([s.avg_duration_minutes for s in all_stats]),
            'avg_settlement_delay': statistics.mean([s.avg_settlement_delay_minutes for s in all_stats]),
            'overall_on_time_rate': statistics.mean([s.on_time_settlement_rate for s in all_stats])
        }
        
        # Sport breakdown
        sport_breakdown = defaultdict(list)
        for key, stats in self.league_stats.items():
            sport = key.split('_')[0]
            sport_breakdown[sport].append(stats)
        
        sport_summary = {}
        for sport, stats_list in sport_breakdown.items():
            sport_summary[sport] = {
                'league_count': len(stats_list),
                'total_matches': sum(s.duration_samples for s in stats_list),
                'avg_duration': statistics.mean([s.avg_duration_minutes for s in stats_list]),
                'avg_settlement': statistics.mean([s.avg_settlement_delay_minutes for s in stats_list]),
                'on_time_rate': statistics.mean([s.on_time_settlement_rate for s in stats_list]),
                'recommended_gap': statistics.mean([s.recommended_gap_minutes for s in stats_list])
            }
        
        # Reliability analysis
        reliability_analysis = {
            'high_reliability_leagues': [
                key for key, stats in self.league_stats.items() 
                if stats.on_time_settlement_rate > 0.9 and stats.duration_samples > 10
            ],
            'problematic_leagues': [
                key for key, stats in self.league_stats.items()
                if stats.on_time_settlement_rate < 0.7 or stats.delayed_settlement_rate > 0.3
            ],
            'insufficient_data_leagues': [
                key for key, stats in self.league_stats.items()
                if stats.duration_samples < self.analysis_config['min_samples_for_stats']
            ]
        }
        
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'analysis_period_days': self.analysis_config['lookback_days'],
            'overall_statistics': overall_stats,
            'sport_breakdown': sport_summary,
            'reliability_analysis': reliability_analysis,
            'league_details': {key: {
                'avg_duration': stats.avg_duration_minutes,
                'avg_settlement': stats.avg_settlement_delay_minutes,
                'recommended_gap': stats.recommended_gap_minutes,
                'on_time_rate': stats.on_time_settlement_rate,
                'sample_size': stats.duration_samples
            } for key, stats in self.league_stats.items()}
        }


# Integration with dynamic chunk manager
def update_chunk_manager_empirical_data():
    """Update empirical data used by dynamic chunk manager"""
    analyzer = SettlementAnalyzer()
    league_stats = analyzer.analyze_historical_patterns()
    
    if league_stats:
        logger.info(f"Updated empirical data with {len(league_stats)} league patterns")
        return True
    else:
        logger.warning("Failed to update empirical data")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test the settlement analyzer
    analyzer = SettlementAnalyzer()
    
    print("⏱️  Settlement Pattern Analyzer")
    print("=" * 50)
    
    try:
        # Analyze historical patterns
        print("Analyzing historical settlement patterns...")
        league_stats = analyzer.analyze_historical_patterns(lookback_days=30)
        
        if league_stats:
            print(f"\nFound patterns for {len(league_stats)} sport/league combinations:")
            
            for league_key, stats in league_stats.items():
                print(f"\n{league_key}:")
                print(f"  Duration: {stats.avg_duration_minutes:.1f}±{stats.duration_std:.1f} min ({stats.duration_samples} samples)")
                print(f"  Settlement: {stats.avg_settlement_delay_minutes:.1f}±{stats.settlement_delay_std:.1f} min")
                print(f"  On-time rate: {stats.on_time_settlement_rate:.1%}")
                print(f"  Recommended gap: {stats.recommended_gap_minutes:.0f} min")
            
            # Test timing recommendations
            print(f"\nTiming Recommendations:")
            soccer_rec = analyzer.get_timing_recommendations('soccer', 'Premier League')
            print(f"Soccer/Premier League: {soccer_rec}")
            
            # Generate report
            print(f"\nGenerating settlement report...")
            report = analyzer.get_settlement_report()
            print(f"Overall on-time rate: {report['overall_statistics']['overall_on_time_rate']:.1%}")
            print(f"Sports analyzed: {list(report['sport_breakdown'].keys())}")
            
        else:
            print("No settlement patterns found (may need real trading data)")
            
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        print(traceback.format_exc())
        print("Note: This requires a database with settled positions to test properly")