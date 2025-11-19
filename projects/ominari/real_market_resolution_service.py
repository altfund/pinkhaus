#!/usr/bin/env python3
"""
Real Market Resolution Service
Resolves markets using actual sports results from real data sources.
"""

import os
import sys
import logging
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Load environment variables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from paper_trading_sessions import PaperTradingSessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Real game result from sports data source."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: str
    source: str
    match_confidence: float = 0.0


class RealSportsDataProvider:
    """Provider for real sports results from multiple sources."""
    
    def __init__(self):
        self.sources = [
            self._get_espn_results,
            self._get_thesportsdb_results,
            self._get_openligadb_results
        ]
    
    def _normalize_team_name(self, name: str) -> str:
        """Normalize team names for matching."""
        if not name:
            return ""
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'\b(FC|CF|SC|AC|BC|United|City|Town|Athletic|Sporting|Club)\b', '', name, flags=re.IGNORECASE)
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        
        return normalized
    
    def _calculate_match_confidence(self, market_home: str, market_away: str, 
                                  result_home: str, result_away: str) -> float:
        """Calculate confidence that this result matches the market."""
        market_home_norm = self._normalize_team_name(market_home)
        market_away_norm = self._normalize_team_name(market_away)
        result_home_norm = self._normalize_team_name(result_home)
        result_away_norm = self._normalize_team_name(result_away)
        
        # Check for exact matches
        if market_home_norm == result_home_norm and market_away_norm == result_away_norm:
            return 1.0
        
        # Check for partial matches (at least one word)
        home_words_market = set(market_home_norm.split())
        away_words_market = set(market_away_norm.split())
        home_words_result = set(result_home_norm.split())
        away_words_result = set(result_away_norm.split())
        
        home_intersection = len(home_words_market & home_words_result)
        away_intersection = len(away_words_market & away_words_result)
        
        if home_intersection > 0 and away_intersection > 0:
            home_score = home_intersection / max(len(home_words_market), len(home_words_result))
            away_score = away_intersection / max(len(away_words_market), len(away_words_result))
            return (home_score + away_score) / 2
        
        return 0.0
    
    def _get_espn_results(self) -> List[GameResult]:
        """Get results from ESPN API."""
        results = []
        try:
            # Try different ESPN soccer leagues
            leagues = ['usa.1', 'eng.1', 'ger.1', 'esp.1']
            
            for league in leagues:
                url = f'https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard'
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get('events', [])
                    
                    for event in events:
                        status = event.get('status', {}).get('type', {}).get('description', '')
                        
                        # Only get completed games
                        if status.lower() in ['final', 'completed']:
                            competitors = event.get('competitions', [{}])[0].get('competitors', [])
                            
                            if len(competitors) == 2:
                                home = competitors[0] if competitors[0].get('homeAway') == 'home' else competitors[1]
                                away = competitors[1] if competitors[1].get('homeAway') == 'away' else competitors[0]
                                
                                home_score = int(home.get('score', 0))
                                away_score = int(away.get('score', 0))
                                
                                results.append(GameResult(
                                    home_team=home.get('team', {}).get('displayName', ''),
                                    away_team=away.get('team', {}).get('displayName', ''),
                                    home_score=home_score,
                                    away_score=away_score,
                                    status='final',
                                    source='ESPN'
                                ))
                                
        except Exception as e:
            logger.debug(f"ESPN API error: {e}")
            
        return results
    
    def _get_thesportsdb_results(self) -> List[GameResult]:
        """Get results from TheSportsDB."""
        results = []
        try:
            url = 'https://www.thesportsdb.com/api/v1/json/3/latestsoccer.php'
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    # Only completed games with scores
                    if event.get('intHomeScore') is not None and event.get('intAwayScore') is not None:
                        results.append(GameResult(
                            home_team=event.get('strHomeTeam', ''),
                            away_team=event.get('strAwayTeam', ''),
                            home_score=int(event.get('intHomeScore', 0)),
                            away_score=int(event.get('intAwayScore', 0)),
                            status='final',
                            source='TheSportsDB'
                        ))
                        
        except Exception as e:
            logger.debug(f"TheSportsDB API error: {e}")
            
        return results
    
    def _get_openligadb_results(self) -> List[GameResult]:
        """Get results from OpenLigaDB (German leagues)."""
        results = []
        try:
            # Get recent completed matches from current season
            url = 'https://api.openligadb.de/getmatchdata/bl1/2024'
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for match in data:
                    # Only completed matches
                    if match.get('matchIsFinished') and match.get('matchResults'):
                        final_result = match['matchResults'][-1]  # Latest result
                        
                        results.append(GameResult(
                            home_team=match.get('team1', {}).get('teamName', ''),
                            away_team=match.get('team2', {}).get('teamName', ''),
                            home_score=final_result.get('pointsTeam1', 0),
                            away_score=final_result.get('pointsTeam2', 0),
                            status='final',
                            source='OpenLigaDB'
                        ))
                        
        except Exception as e:
            logger.debug(f"OpenLigaDB API error: {e}")
            
        return results
    
    def get_all_results(self) -> List[GameResult]:
        """Get results from all available sources."""
        all_results = []
        
        for source_func in self.sources:
            try:
                results = source_func()
                all_results.extend(results)
                logger.debug(f"Got {len(results)} results from {source_func.__name__}")
            except Exception as e:
                logger.debug(f"Error from {source_func.__name__}: {e}")
        
        logger.info(f"Total real results collected: {len(all_results)}")
        return all_results


class RealMarketResolutionService:
    """Service to resolve markets using real sports results."""
    
    def __init__(self):
        self.data_provider = RealSportsDataProvider()
        self.paper_trading = PaperTradingSessionManager()
        self.min_confidence_threshold = 0.7  # 70% confidence required for match
    
    def resolve_markets_with_real_data(self) -> Dict:
        """Resolve markets using real sports results."""
        results = {
            'resolved_count': 0,
            'no_match_count': 0,
            'low_confidence_count': 0,
            'resolved_markets': [],
            'unresolved_markets': []
        }
        
        try:
            # Get real sports results
            real_results = self.data_provider.get_all_results()
            logger.info(f"Found {len(real_results)} real sports results")
            
            if not real_results:
                logger.warning("No real sports results available")
                return results
            
            with db_manager.get_db_session() as db:
                now = datetime.now(timezone.utc)
                
                # Find unfinished past markets
                past_markets = db.query(Market).filter(
                    Market.is_finished == False,
                    Market.maturity_date < now - timedelta(hours=2)  # Allow 2h buffer
                ).all()
                
                logger.info(f"Found {len(past_markets)} past markets to resolve")
                
                market_resolutions = {}
                
                for market in past_markets:
                    best_match = None
                    best_confidence = 0
                    
                    # Try to match with real results
                    for result in real_results:
                        confidence = self.data_provider._calculate_match_confidence(
                            market.home_team, market.away_team,
                            result.home_team, result.away_team
                        )
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = result
                    
                    if best_match and best_confidence >= self.min_confidence_threshold:
                        # Resolve with real result
                        winning_outcome = self._determine_winning_outcome(
                            best_match.home_score, best_match.away_score
                        )
                        
                        market.is_finished = True
                        
                        market_resolutions[market.source_id] = {
                            'is_finished': True,
                            'resolved_outcome': winning_outcome,
                            'resolution_time': now.isoformat(),
                            'real_result': True,
                            'confidence': best_confidence,
                            'source': best_match.source,
                            'final_score': f"{best_match.home_score}-{best_match.away_score}"
                        }
                        
                        results['resolved_markets'].append({
                            'market_id': market.source_id,
                            'home_team': market.home_team,
                            'away_team': market.away_team,
                            'winning_outcome': winning_outcome,
                            'real_score': f"{best_match.home_score}-{best_match.away_score}",
                            'confidence': best_confidence,
                            'source': best_match.source
                        })
                        
                        results['resolved_count'] += 1
                        
                        logger.info(f"✅ REAL RESULT: {market.home_team} vs {market.away_team} -> {winning_outcome} ({best_match.home_score}-{best_match.away_score}) [Confidence: {best_confidence:.2f}]")
                    
                    elif best_match and best_confidence > 0.3:
                        # Low confidence match
                        results['low_confidence_count'] += 1
                        results['unresolved_markets'].append({
                            'market': f"{market.home_team} vs {market.away_team}",
                            'best_match': f"{best_match.home_team} vs {best_match.away_team}",
                            'confidence': best_confidence,
                            'reason': 'Low confidence match'
                        })
                        
                        logger.warning(f"❌ Low confidence: {market.home_team} vs {market.away_team} -> {best_match.home_team} vs {best_match.away_team} (confidence: {best_confidence:.2f})")
                    
                    else:
                        # No match found
                        results['no_match_count'] += 1
                        results['unresolved_markets'].append({
                            'market': f"{market.home_team} vs {market.away_team}",
                            'reason': 'No matching real result found'
                        })
                
                db.commit()
                
                # Settle paper trading positions with real results
                if market_resolutions:
                    try:
                        # Get active sessions - simplified approach
                        session_files = []  # Would need to implement get_active_session_ids
                        for session_id in ['default']:  # Use default session for now
                            try:
                                settled = self.paper_trading.settle_finished_markets(session_id, market_resolutions)
                                if settled > 0:
                                    logger.info(f"Settled {settled} positions with REAL results in session {session_id}")
                            except:
                                pass  # Session might not exist
                    except Exception as e:
                        logger.debug(f"Paper trading settlement error: {e}")
                
                logger.info(f"✅ REAL RESOLUTION COMPLETE: {results['resolved_count']} with real data, {results['no_match_count']} no match, {results['low_confidence_count']} low confidence")
                
        except Exception as e:
            logger.error(f"Error resolving markets with real data: {e}")
            results['error'] = str(e)
        
        return results
    
    def _determine_winning_outcome(self, home_score: int, away_score: int) -> str:
        """Determine winning outcome from real scores."""
        if home_score > away_score:
            return 'home'
        elif away_score > home_score:
            return 'away'
        else:
            return 'draw'


def main():
    """Run real market resolution service."""
    logger.info("🏁 Starting REAL Market Resolution Service")
    
    service = RealMarketResolutionService()
    results = service.resolve_markets_with_real_data()
    
    logger.info(f"Final Results: {results}")
    
    if results['resolved_count'] > 0:
        logger.info(f"🎯 SUCCESS: Resolved {results['resolved_count']} markets with REAL sports results!")
    else:
        logger.warning("⚠️ No markets resolved - check if real sports results match our market names")


if __name__ == "__main__":
    main()