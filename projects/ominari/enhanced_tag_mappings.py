#!/usr/bin/env python3
"""
Enhanced Tag Mappings for Blockchain Sports Data

Complete mappings for sport and league IDs used in Overtime Markets contracts.
Based on Chainlink oracle standards and Overtime's implementation.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class SportID(IntEnum):
    """Sport ID enumeration matching blockchain tags."""
    FOOTBALL = 1      # American Football
    BASKETBALL = 2
    BASEBALL = 3
    HOCKEY = 4
    SOCCER = 5
    MMA = 6
    BOXING = 7
    TENNIS = 8
    GOLF = 9
    CRICKET = 10
    RUGBY = 11
    MOTORSPORT = 12
    ESPORTS = 13
    VOLLEYBALL = 14
    HANDBALL = 15


class LeagueID(IntEnum):
    """League ID enumeration (second tag in array)."""
    # American Football
    NFL = 101
    NCAA_FOOTBALL = 102
    CFL = 103
    
    # Basketball
    NBA = 201
    WNBA = 202
    NCAA_BASKETBALL = 203
    EUROLEAGUE = 204
    NBL = 205  # Australia
    
    # Baseball
    MLB = 301
    NPB = 302  # Japan
    KBO = 303  # Korea
    
    # Hockey
    NHL = 401
    KHL = 402
    SHL = 403  # Sweden
    
    # Soccer - Major Leagues
    EPL = 501  # English Premier League
    LA_LIGA = 502
    SERIE_A = 503
    BUNDESLIGA = 504
    LIGUE_1 = 505
    EREDIVISIE = 506
    PRIMEIRA_LIGA = 507
    
    # Soccer - International
    CHAMPIONS_LEAGUE = 550
    EUROPA_LEAGUE = 551
    WORLD_CUP = 552
    EUROS = 553
    COPA_AMERICA = 554
    
    # Soccer - Other
    MLS = 560
    LIGA_MX = 561
    BRASILEIRAO = 562
    ARGENTINE_PRIMERA = 563
    A_LEAGUE = 564  # Australia
    J_LEAGUE = 565  # Japan
    K_LEAGUE = 566  # Korea
    CHINESE_SUPER_LEAGUE = 567
    INDIAN_SUPER_LEAGUE = 568
    
    # MMA
    UFC = 601
    BELLATOR = 602
    ONE_FC = 603
    PFL = 604
    
    # Boxing
    BOXING_CHAMPIONSHIP = 701
    
    # Tennis
    ATP = 801
    WTA = 802
    GRAND_SLAM = 803
    
    # Golf
    PGA_TOUR = 901
    EUROPEAN_TOUR = 902
    LPGA = 903
    
    # Cricket
    IPL = 1001
    BIG_BASH = 1002
    INTERNATIONAL_TEST = 1003
    INTERNATIONAL_ODI = 1004
    INTERNATIONAL_T20 = 1005
    
    # Esports
    LOL_WORLDS = 1301
    CS_MAJOR = 1302
    DOTA_TI = 1303


@dataclass
class SportMetadata:
    """Additional metadata for sports."""
    sport_id: int
    name: str
    short_name: str
    positions: List[str]  # e.g., ["home", "away"] or ["home", "away", "draw"]
    has_draw: bool
    typical_duration_minutes: int
    scoring_type: str  # "goals", "points", "runs", etc.


@dataclass
class LeagueMetadata:
    """Additional metadata for leagues."""
    league_id: int
    sport_id: int
    name: str
    short_name: str
    country: Optional[str]
    level: int  # 1 = top tier, 2 = second tier, etc.
    season_type: str  # "regular", "playoffs", "tournament"


class EnhancedTagMapper:
    """Enhanced mapping service for blockchain sports data."""
    
    def __init__(self):
        self._init_sport_metadata()
        self._init_league_metadata()
        
    def _init_sport_metadata(self):
        """Initialize sport metadata."""
        self.sport_metadata = {
            SportID.FOOTBALL: SportMetadata(
                sport_id=SportID.FOOTBALL,
                name="American Football",
                short_name="Football",
                positions=["home", "away"],
                has_draw=False,
                typical_duration_minutes=180,
                scoring_type="points"
            ),
            SportID.BASKETBALL: SportMetadata(
                sport_id=SportID.BASKETBALL,
                name="Basketball",
                short_name="Basketball",
                positions=["home", "away"],
                has_draw=False,
                typical_duration_minutes=150,
                scoring_type="points"
            ),
            SportID.BASEBALL: SportMetadata(
                sport_id=SportID.BASEBALL,
                name="Baseball",
                short_name="Baseball",
                positions=["home", "away"],
                has_draw=False,
                typical_duration_minutes=180,
                scoring_type="runs"
            ),
            SportID.HOCKEY: SportMetadata(
                sport_id=SportID.HOCKEY,
                name="Ice Hockey",
                short_name="Hockey",
                positions=["home", "away"],
                has_draw=False,
                typical_duration_minutes=180,
                scoring_type="goals"
            ),
            SportID.SOCCER: SportMetadata(
                sport_id=SportID.SOCCER,
                name="Soccer",
                short_name="Soccer",
                positions=["home", "away", "draw"],
                has_draw=True,
                typical_duration_minutes=105,
                scoring_type="goals"
            ),
            SportID.MMA: SportMetadata(
                sport_id=SportID.MMA,
                name="Mixed Martial Arts",
                short_name="MMA",
                positions=["fighter1", "fighter2"],
                has_draw=True,
                typical_duration_minutes=25,
                scoring_type="fight"
            ),
            SportID.TENNIS: SportMetadata(
                sport_id=SportID.TENNIS,
                name="Tennis",
                short_name="Tennis",
                positions=["player1", "player2"],
                has_draw=False,
                typical_duration_minutes=120,
                scoring_type="sets"
            ),
            SportID.GOLF: SportMetadata(
                sport_id=SportID.GOLF,
                name="Golf",
                short_name="Golf",
                positions=["winner"],  # Multiple players
                has_draw=False,
                typical_duration_minutes=1440,  # 4 days
                scoring_type="strokes"
            )
        }
    
    def _init_league_metadata(self):
        """Initialize league metadata."""
        self.league_metadata = {
            # NFL
            LeagueID.NFL: LeagueMetadata(
                league_id=LeagueID.NFL,
                sport_id=SportID.FOOTBALL,
                name="National Football League",
                short_name="NFL",
                country="USA",
                level=1,
                season_type="regular"
            ),
            
            # NBA
            LeagueID.NBA: LeagueMetadata(
                league_id=LeagueID.NBA,
                sport_id=SportID.BASKETBALL,
                name="National Basketball Association",
                short_name="NBA",
                country="USA",
                level=1,
                season_type="regular"
            ),
            
            # MLB
            LeagueID.MLB: LeagueMetadata(
                league_id=LeagueID.MLB,
                sport_id=SportID.BASEBALL,
                name="Major League Baseball",
                short_name="MLB",
                country="USA",
                level=1,
                season_type="regular"
            ),
            
            # NHL
            LeagueID.NHL: LeagueMetadata(
                league_id=LeagueID.NHL,
                sport_id=SportID.HOCKEY,
                name="National Hockey League",
                short_name="NHL",
                country="USA/Canada",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - England
            LeagueID.EPL: LeagueMetadata(
                league_id=LeagueID.EPL,
                sport_id=SportID.SOCCER,
                name="English Premier League",
                short_name="EPL",
                country="England",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - Spain
            LeagueID.LA_LIGA: LeagueMetadata(
                league_id=LeagueID.LA_LIGA,
                sport_id=SportID.SOCCER,
                name="La Liga",
                short_name="La Liga",
                country="Spain",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - Italy
            LeagueID.SERIE_A: LeagueMetadata(
                league_id=LeagueID.SERIE_A,
                sport_id=SportID.SOCCER,
                name="Serie A",
                short_name="Serie A",
                country="Italy",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - Germany
            LeagueID.BUNDESLIGA: LeagueMetadata(
                league_id=LeagueID.BUNDESLIGA,
                sport_id=SportID.SOCCER,
                name="Bundesliga",
                short_name="Bundesliga",
                country="Germany",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - France
            LeagueID.LIGUE_1: LeagueMetadata(
                league_id=LeagueID.LIGUE_1,
                sport_id=SportID.SOCCER,
                name="Ligue 1",
                short_name="Ligue 1",
                country="France",
                level=1,
                season_type="regular"
            ),
            
            # Soccer - International
            LeagueID.CHAMPIONS_LEAGUE: LeagueMetadata(
                league_id=LeagueID.CHAMPIONS_LEAGUE,
                sport_id=SportID.SOCCER,
                name="UEFA Champions League",
                short_name="UCL",
                country="Europe",
                level=1,
                season_type="tournament"
            ),
            
            # MMA
            LeagueID.UFC: LeagueMetadata(
                league_id=LeagueID.UFC,
                sport_id=SportID.MMA,
                name="Ultimate Fighting Championship",
                short_name="UFC",
                country="International",
                level=1,
                season_type="events"
            ),
            
            # Add more leagues as needed...
        }
    
    def decode_tags(self, tags: List[int]) -> Tuple[str, str]:
        """
        Decode tags array to sport and league names.
        
        Args:
            tags: List of tag integers [sport_id, league_id, ...]
            
        Returns:
            Tuple of (sport_name, league_name)
        """
        if not tags or len(tags) < 2:
            return ("Unknown", "Unknown")
        
        sport_id = tags[0]
        league_id = tags[1]
        
        sport = self.get_sport_name(sport_id)
        league = self.get_league_name(league_id)
        
        return (sport, league)
    
    def get_sport_name(self, sport_id: int) -> str:
        """Get sport name from ID."""
        try:
            sport_enum = SportID(sport_id)
            metadata = self.sport_metadata.get(sport_enum)
            return metadata.name if metadata else f"Sport_{sport_id}"
        except ValueError:
            return f"Sport_{sport_id}"
    
    def get_league_name(self, league_id: int) -> str:
        """Get league name from ID."""
        try:
            league_enum = LeagueID(league_id)
            metadata = self.league_metadata.get(league_enum)
            return metadata.name if metadata else f"League_{league_id}"
        except ValueError:
            return f"League_{league_id}"
    
    def get_sport_metadata(self, sport_id: int) -> Optional[SportMetadata]:
        """Get complete metadata for a sport."""
        try:
            sport_enum = SportID(sport_id)
            return self.sport_metadata.get(sport_enum)
        except ValueError:
            return None
    
    def get_league_metadata(self, league_id: int) -> Optional[LeagueMetadata]:
        """Get complete metadata for a league."""
        try:
            league_enum = LeagueID(league_id)
            return self.league_metadata.get(league_enum)
        except ValueError:
            return None
    
    def get_positions_for_tags(self, tags: List[int]) -> List[str]:
        """Get position names for a sport based on tags."""
        if not tags:
            return ["home", "away"]
        
        sport_meta = self.get_sport_metadata(tags[0])
        if sport_meta:
            return sport_meta.positions
        return ["home", "away"]
    
    def parse_game_label(self, game_label: str, sport_id: int) -> Dict[str, str]:
        """
        Parse game label to extract team/player names.
        
        Args:
            game_label: String like "Team A vs Team B" or "Player A vs Player B"
            sport_id: Sport ID to determine parsing logic
            
        Returns:
            Dict with parsed components
        """
        result = {
            "home": "",
            "away": "",
            "full_label": game_label
        }
        
        # Common separators
        separators = [" vs ", " v ", " @ ", " - "]
        
        for sep in separators:
            if sep in game_label:
                parts = game_label.split(sep, 1)
                if len(parts) == 2:
                    result["home"] = parts[0].strip()
                    result["away"] = parts[1].strip()
                    break
        
        # Handle special cases for individual sports
        if sport_id in [SportID.MMA, SportID.BOXING, SportID.TENNIS]:
            result["fighter1"] = result.get("home", "")
            result["fighter2"] = result.get("away", "")
            result["player1"] = result.get("home", "")
            result["player2"] = result.get("away", "")
        
        return result


# Singleton instance
tag_mapper = EnhancedTagMapper()


def test_mappings():
    """Test the tag mapping functionality."""
    mapper = EnhancedTagMapper()
    
    # Test cases
    test_cases = [
        ([5, 501], "Soccer - EPL"),
        ([1, 101], "Football - NFL"),
        ([2, 201], "Basketball - NBA"),
        ([6, 601], "MMA - UFC"),
        ([5, 550], "Soccer - Champions League"),
    ]
    
    print("Testing Tag Mappings:")
    print("-" * 50)
    
    for tags, expected in test_cases:
        sport, league = mapper.decode_tags(tags)
        print(f"Tags {tags}: {sport} - {league}")
        
        # Get metadata
        sport_meta = mapper.get_sport_metadata(tags[0])
        if sport_meta:
            print(f"  Positions: {sport_meta.positions}")
            print(f"  Has Draw: {sport_meta.has_draw}")
    
    print("\nTesting Game Label Parsing:")
    print("-" * 50)
    
    test_labels = [
        ("Liverpool vs Manchester City", SportID.SOCCER),
        ("Lakers @ Warriors", SportID.BASKETBALL),
        ("McGregor vs Poirier", SportID.MMA),
    ]
    
    for label, sport_id in test_labels:
        parsed = mapper.parse_game_label(label, sport_id)
        print(f"{label}: {parsed}")


if __name__ == "__main__":
    test_mappings()