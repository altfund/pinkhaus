#!/usr/bin/env python3
"""
Dynamic Chunk Manager - Intelligent Time-Based Market Grouping

Groups matches based on empirical settlement data and capital recycling patterns
instead of arbitrary fixed time windows.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MatchTiming:
    """Empirical timing data for a match"""
    match_id: str
    sport: str
    league: str
    start_time: datetime
    expected_duration_minutes: float = 105  # 90 + injury time + buffer
    settlement_delay_minutes: float = 15    # Time from end to settlement
    
    @property
    def expected_end_time(self) -> datetime:
        """When the match is expected to end"""
        return self.start_time + timedelta(minutes=self.expected_duration_minutes)
    
    @property
    def expected_settlement_time(self) -> datetime:
        """When funds are expected to be available"""
        return self.expected_end_time + timedelta(minutes=self.settlement_delay_minutes)


@dataclass
class DynamicChunk:
    """A dynamically sized time chunk of matches"""
    chunk_id: str
    start_time: datetime
    end_time: datetime
    matches: List[Dict] = field(default_factory=list)
    
    # Capital tracking
    expected_stake: float = 0.0
    expected_return: float = 0.0
    expected_settlement_time: datetime = None
    
    # Risk metrics
    concentration_score: float = 0.0  # How concentrated risk is
    
    def add_match(self, match: Dict, timing: MatchTiming):
        """Add a match to this chunk"""
        self.matches.append(match)
        
        # Update timing based on latest settlement
        if timing.expected_settlement_time > self.end_time:
            self.end_time = timing.expected_settlement_time
            self.expected_settlement_time = timing.expected_settlement_time
    
    @property
    def duration_minutes(self) -> float:
        """Duration of this chunk in minutes"""
        return (self.end_time - self.start_time).total_seconds() / 60
    
    @property
    def match_count(self) -> int:
        """Number of matches in this chunk"""
        return len(self.matches)
    
    @property
    def label(self) -> str:
        """Human-readable label"""
        now = datetime.now(timezone.utc)
        hours_away = (self.start_time - now).total_seconds() / 3600
        
        if hours_away < 0:
            return f"Live ({self.match_count} matches)"
        elif hours_away < 0.5:
            return f"Next 30min ({self.match_count} matches)" 
        elif hours_away < 1:
            return f"Next Hour ({self.match_count} matches)"
        elif hours_away < 3:
            return f"In {int(hours_away)}h ({self.match_count} matches)"
        else:
            return f"{self.start_time.strftime('%H:%M')} ({self.match_count} matches)"


class DynamicChunkManager:
    """Manages dynamic time-based chunking of markets"""
    
    def __init__(self):
        # Default timing parameters (will be updated from empirical data)
        self.default_timings = {
            'soccer': {
                'duration_minutes': 105,      # 90 + injury time
                'settlement_minutes': 15,     # Blockchain settlement
                'min_gap_minutes': 15         # Minimum gap between chunks
            },
            'basketball': {
                'duration_minutes': 150,      # 48 mins + breaks + OT potential
                'settlement_minutes': 10,
                'min_gap_minutes': 20
            },
            'tennis': {
                'duration_minutes': 180,      # Highly variable
                'settlement_minutes': 10,
                'min_gap_minutes': 30
            }
        }
        
        # Load empirical data if available
        self.empirical_data = self.load_empirical_data()
        
        # Risk parameters
        self.max_concentration_per_chunk = 0.3  # Max 30% of bankroll in one chunk
        self.max_concurrent_chunks = 5          # Max chunks with active exposure
        
    def load_empirical_data(self) -> Dict:
        """Load historical settlement patterns"""
        try:
            with open('settlement_patterns.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def get_match_timing(self, market: Dict) -> MatchTiming:
        """Get timing estimates for a market"""
        sport = market.get('sport', 'soccer').lower()
        league = market.get('league', '')
        
        # Check empirical data first
        empirical_key = f"{sport}_{league}"
        if empirical_key in self.empirical_data:
            timing_data = self.empirical_data[empirical_key]
        else:
            timing_data = self.default_timings.get(sport, self.default_timings['soccer'])
        
        return MatchTiming(
            match_id=market.get('match_id', market.get('market_id', '')),
            sport=sport,
            league=league,
            start_time=market['maturity_date'] if isinstance(market['maturity_date'], datetime) 
                      else datetime.fromisoformat(market['maturity_date'].replace('Z', '+00:00')),
            expected_duration_minutes=timing_data['duration_minutes'],
            settlement_delay_minutes=timing_data['settlement_minutes']
        )
    
    def create_dynamic_chunks(self, 
                            markets: List[Dict], 
                            current_capital_state: Dict) -> List[DynamicChunk]:
        """Create dynamic chunks based on match timing and capital availability"""
        
        if not markets:
            return []
        
        # Sort markets by start time
        sorted_markets = sorted(markets, key=lambda m: m['maturity_date'])
        
        chunks = []
        current_chunk = None
        
        for market in sorted_markets:
            timing = self.get_match_timing(market)
            
            # Decide if we need a new chunk
            if self._should_start_new_chunk(current_chunk, timing, current_capital_state):
                if current_chunk:
                    chunks.append(current_chunk)
                
                current_chunk = DynamicChunk(
                    chunk_id=f"chunk_{len(chunks)}_{timing.start_time.strftime('%H%M')}",
                    start_time=timing.start_time,
                    end_time=timing.expected_settlement_time
                )
            
            # Add market to current chunk
            if current_chunk:
                current_chunk.add_match(market, timing)
        
        # Don't forget the last chunk
        if current_chunk and current_chunk.matches:
            chunks.append(current_chunk)
        
        # Post-process chunks for risk management
        chunks = self._optimize_chunks_for_risk(chunks, current_capital_state)
        
        logger.info(f"Created {len(chunks)} dynamic chunks from {len(markets)} markets")
        self._log_chunk_summary(chunks)
        
        return chunks
    
    def _should_start_new_chunk(self, 
                               current_chunk: Optional[DynamicChunk],
                               timing: MatchTiming,
                               capital_state: Dict) -> bool:
        """Determine if we should start a new chunk"""
        
        if not current_chunk:
            return True
        
        # Check time gap
        time_gap = (timing.start_time - current_chunk.start_time).total_seconds() / 60
        sport_config = self.default_timings.get(timing.sport, self.default_timings['soccer'])
        min_gap = sport_config['min_gap_minutes']
        
        # Dynamic gap based on capital availability
        available_ratio = capital_state.get('available_cash', 0) / capital_state.get('total_bankroll', 1)
        if available_ratio < 0.3:  # Less than 30% available
            min_gap *= 2  # Double the gap when capital is tight
        
        # Start new chunk if:
        # 1. Time gap is significant
        if time_gap > min_gap:
            return True
        
        # 2. Current chunk is getting too large (risk concentration)
        if current_chunk.match_count >= 10:  # Max 10 matches per chunk
            return True
        
        # 3. Capital recovery expected before this match
        if current_chunk.expected_settlement_time and \
           timing.start_time > current_chunk.expected_settlement_time:
            return True
        
        return False
    
    def _optimize_chunks_for_risk(self, 
                                 chunks: List[DynamicChunk], 
                                 capital_state: Dict) -> List[DynamicChunk]:
        """Optimize chunks to manage risk concentration"""
        
        # Calculate expected exposure per chunk
        total_bankroll = capital_state.get('total_bankroll', 10000)
        max_per_chunk = total_bankroll * self.max_concentration_per_chunk
        
        optimized_chunks = []
        for chunk in chunks:
            # If chunk is too large, split it
            if len(chunk.matches) > 15:  # Hard limit
                # Split into smaller chunks
                mid = len(chunk.matches) // 2
                
                chunk1 = DynamicChunk(
                    chunk_id=f"{chunk.chunk_id}_1",
                    start_time=chunk.start_time,
                    end_time=chunk.matches[mid-1].get('maturity_date', chunk.end_time),
                    matches=chunk.matches[:mid]
                )
                
                chunk2 = DynamicChunk(
                    chunk_id=f"{chunk.chunk_id}_2", 
                    start_time=chunk.matches[mid].get('maturity_date', chunk.start_time),
                    end_time=chunk.end_time,
                    matches=chunk.matches[mid:]
                )
                
                optimized_chunks.extend([chunk1, chunk2])
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _log_chunk_summary(self, chunks: List[DynamicChunk]):
        """Log a summary of created chunks"""
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i+1}: {chunk.label} - "
                       f"{chunk.start_time.strftime('%H:%M')} to {chunk.end_time.strftime('%H:%M')} "
                       f"({chunk.duration_minutes:.0f} mins)")
    
    def get_settlement_schedule(self, chunks: List[DynamicChunk]) -> List[Dict]:
        """Get expected settlement schedule for capital planning"""
        schedule = []
        
        for chunk in chunks:
            if chunk.expected_settlement_time:
                schedule.append({
                    'chunk_id': chunk.chunk_id,
                    'settlement_time': chunk.expected_settlement_time,
                    'expected_return': chunk.expected_return,
                    'match_count': chunk.match_count,
                    'label': chunk.label
                })
        
        # Sort by settlement time
        schedule.sort(key=lambda x: x['settlement_time'])
        
        return schedule
    
    def update_empirical_data(self, 
                            match_id: str,
                            sport: str,
                            league: str, 
                            actual_duration: float,
                            actual_settlement_delay: float):
        """Update empirical data with actual observed timings"""
        
        key = f"{sport}_{league}"
        
        if key not in self.empirical_data:
            self.empirical_data[key] = {
                'duration_minutes': actual_duration,
                'settlement_minutes': actual_settlement_delay,
                'sample_count': 1
            }
        else:
            # Rolling average
            data = self.empirical_data[key]
            n = data['sample_count']
            
            data['duration_minutes'] = (data['duration_minutes'] * n + actual_duration) / (n + 1)
            data['settlement_minutes'] = (data['settlement_minutes'] * n + actual_settlement_delay) / (n + 1)
            data['sample_count'] = n + 1
        
        # Persist updates
        try:
            with open('settlement_patterns.json', 'w') as f:
                json.dump(self.empirical_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save empirical data: {e}")


# Example usage
if __name__ == "__main__":
    # Test the dynamic chunk manager
    manager = DynamicChunkManager()
    
    # Create some test markets
    now = datetime.now(timezone.utc)
    test_markets = [
        {
            'market_id': 'M1',
            'sport': 'soccer', 
            'league': 'Premier League',
            'home_team': 'Chelsea',
            'away_team': 'Arsenal',
            'maturity_date': now + timedelta(minutes=30)
        },
        {
            'market_id': 'M2',
            'sport': 'soccer',
            'league': 'Premier League', 
            'home_team': 'Liverpool',
            'away_team': 'Man City',
            'maturity_date': now + timedelta(minutes=35)
        },
        {
            'market_id': 'M3',
            'sport': 'soccer',
            'league': 'La Liga',
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona', 
            'maturity_date': now + timedelta(hours=3)
        }
    ]
    
    # Test capital state
    capital_state = {
        'total_bankroll': 10000,
        'available_cash': 7000,
        'pending_stakes': 2000,
        'in_play_exposure': 1000
    }
    
    # Create chunks
    chunks = manager.create_dynamic_chunks(test_markets, capital_state)
    
    # Show settlement schedule
    schedule = manager.get_settlement_schedule(chunks)
    print("\nExpected Settlement Schedule:")
    for item in schedule:
        print(f"  {item['settlement_time'].strftime('%H:%M')} - {item['label']} "
              f"(Expected return: ${item['expected_return']:.2f})")