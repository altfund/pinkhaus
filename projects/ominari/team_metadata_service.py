#!/usr/bin/env python3
"""
Team Metadata Service

Enriches blockchain sports data with team metadata including:
- Canonical team names
- Abbreviations
- Conferences/Divisions
- Team colors
- Stadium/Arena information
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass
class TeamMetadata:
    """Complete metadata for a sports team."""
    team_id: str
    sport: str
    league: str
    full_name: str
    short_name: str
    abbreviation: str
    aliases: List[str]  # Alternative names
    conference: Optional[str] = None
    division: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    founded_year: Optional[int] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    venue: Optional[str] = None
    venue_capacity: Optional[int] = None


class TeamMetadataService:
    """Service for managing and enriching team metadata."""
    
    def __init__(self, db_path: str = "team_metadata.db"):
        self.db_path = db_path
        self._init_database()
        self._init_default_teams()
        
    def _init_database(self):
        """Initialize the metadata database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_metadata (
                team_id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                league TEXT NOT NULL,
                full_name TEXT NOT NULL,
                short_name TEXT NOT NULL,
                abbreviation TEXT NOT NULL,
                aliases TEXT,  -- JSON array
                conference TEXT,
                division TEXT,
                city TEXT,
                country TEXT,
                founded_year INTEGER,
                primary_color TEXT,
                secondary_color TEXT,
                venue TEXT,
                venue_capacity INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_team_names 
            ON team_metadata(full_name, short_name, abbreviation)
        """)
        
        conn.commit()
        conn.close()
    
    def _init_default_teams(self):
        """Initialize with default team metadata."""
        # Check if already initialized
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM team_metadata").fetchone()[0]
        conn.close()
        
        if count > 0:
            return
        
        # NFL Teams
        nfl_teams = [
            TeamMetadata(
                team_id="nfl_buf",
                sport="Football",
                league="NFL",
                full_name="Buffalo Bills",
                short_name="Bills",
                abbreviation="BUF",
                aliases=["Buffalo"],
                conference="AFC",
                division="East",
                city="Buffalo",
                country="USA",
                founded_year=1960,
                primary_color="#00338D",
                secondary_color="#C60C30",
                venue="Highmark Stadium",
                venue_capacity=71608
            ),
            TeamMetadata(
                team_id="nfl_dal",
                sport="Football", 
                league="NFL",
                full_name="Dallas Cowboys",
                short_name="Cowboys",
                abbreviation="DAL",
                aliases=["Dallas"],
                conference="NFC",
                division="East",
                city="Dallas",
                country="USA",
                founded_year=1960,
                primary_color="#003594",
                secondary_color="#869397",
                venue="AT&T Stadium",
                venue_capacity=80000
            ),
            # Add more NFL teams...
        ]
        
        # NBA Teams
        nba_teams = [
            TeamMetadata(
                team_id="nba_lal",
                sport="Basketball",
                league="NBA",
                full_name="Los Angeles Lakers",
                short_name="Lakers",
                abbreviation="LAL",
                aliases=["LA Lakers", "L.A. Lakers"],
                conference="Western",
                division="Pacific",
                city="Los Angeles",
                country="USA",
                founded_year=1947,
                primary_color="#552583",
                secondary_color="#FDB927",
                venue="Crypto.com Arena",
                venue_capacity=18997
            ),
            TeamMetadata(
                team_id="nba_bos",
                sport="Basketball",
                league="NBA",
                full_name="Boston Celtics",
                short_name="Celtics",
                abbreviation="BOS",
                aliases=["Boston"],
                conference="Eastern",
                division="Atlantic",
                city="Boston",
                country="USA",
                founded_year=1946,
                primary_color="#007A33",
                secondary_color="#BA9653",
                venue="TD Garden",
                venue_capacity=19156
            ),
            # Add more NBA teams...
        ]
        
        # Premier League Teams
        epl_teams = [
            TeamMetadata(
                team_id="epl_liv",
                sport="Soccer",
                league="EPL",
                full_name="Liverpool FC",
                short_name="Liverpool",
                abbreviation="LIV",
                aliases=["Liverpool F.C.", "The Reds"],
                conference=None,
                division=None,
                city="Liverpool",
                country="England",
                founded_year=1892,
                primary_color="#C8102E",
                secondary_color="#F6EB61",
                venue="Anfield",
                venue_capacity=53394
            ),
            TeamMetadata(
                team_id="epl_mci",
                sport="Soccer",
                league="EPL",
                full_name="Manchester City",
                short_name="Man City",
                abbreviation="MCI",
                aliases=["Manchester City FC", "Man City", "City"],
                conference=None,
                division=None,
                city="Manchester",
                country="England",
                founded_year=1880,
                primary_color="#6CABDD",
                secondary_color="#FFFFFF",
                venue="Etihad Stadium",
                venue_capacity=53400
            ),
            TeamMetadata(
                team_id="epl_mun",
                sport="Soccer",
                league="EPL",
                full_name="Manchester United",
                short_name="Man United",
                abbreviation="MUN",
                aliases=["Manchester United FC", "Man United", "Man Utd", "United"],
                conference=None,
                division=None,
                city="Manchester",
                country="England",
                founded_year=1878,
                primary_color="#DA020E",
                secondary_color="#FFE500",
                venue="Old Trafford",
                venue_capacity=74310
            ),
            # Add more EPL teams...
        ]
        
        # Store all teams
        all_teams = nfl_teams + nba_teams + epl_teams
        for team in all_teams:
            self.add_team(team)
    
    def add_team(self, team: TeamMetadata):
        """Add or update team metadata."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO team_metadata (
                team_id, sport, league, full_name, short_name, abbreviation,
                aliases, conference, division, city, country, founded_year,
                primary_color, secondary_color, venue, venue_capacity,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            team.team_id,
            team.sport,
            team.league,
            team.full_name,
            team.short_name,
            team.abbreviation,
            json.dumps(team.aliases),
            team.conference,
            team.division,
            team.city,
            team.country,
            team.founded_year,
            team.primary_color,
            team.secondary_color,
            team.venue,
            team.venue_capacity,
            datetime.now(timezone.utc).isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def find_team(self, name: str, sport: Optional[str] = None, 
                  league: Optional[str] = None) -> Optional[TeamMetadata]:
        """
        Find team by name (fuzzy matching).
        
        Args:
            name: Team name to search for
            sport: Optional sport filter
            league: Optional league filter
            
        Returns:
            Best matching TeamMetadata or None
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = """
            SELECT * FROM team_metadata
            WHERE (
                LOWER(full_name) = LOWER(?) OR
                LOWER(short_name) = LOWER(?) OR
                LOWER(abbreviation) = LOWER(?) OR
                EXISTS (
                    SELECT 1 FROM json_each(aliases) 
                    WHERE LOWER(value) = LOWER(?)
                )
            )
        """
        params = [name, name, name, name]
        
        if sport:
            query += " AND LOWER(sport) = LOWER(?)"
            params.append(sport)
            
        if league:
            query += " AND LOWER(league) = LOWER(?)"
            params.append(league)
        
        result = conn.execute(query, params).fetchone()
        conn.close()
        
        if result:
            return self._row_to_team(result)
        
        # Try fuzzy matching
        return self._fuzzy_match_team(name, sport, league)
    
    def _fuzzy_match_team(self, name: str, sport: Optional[str] = None,
                          league: Optional[str] = None) -> Optional[TeamMetadata]:
        """Fuzzy match team name."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all teams
        query = "SELECT * FROM team_metadata WHERE 1=1"
        params = []
        
        if sport:
            query += " AND LOWER(sport) = LOWER(?)"
            params.append(sport)
            
        if league:
            query += " AND LOWER(league) = LOWER(?)"
            params.append(league)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        # Score each team
        best_score = 0
        best_team = None
        name_lower = name.lower()
        
        for row in rows:
            team = self._row_to_team(row)
            score = 0
            
            # Check exact matches
            if name_lower == team.full_name.lower():
                score = 100
            elif name_lower == team.short_name.lower():
                score = 90
            elif name_lower == team.abbreviation.lower():
                score = 85
            elif name_lower in [alias.lower() for alias in team.aliases]:
                score = 80
            # Partial matches
            elif name_lower in team.full_name.lower():
                score = 70
            elif team.short_name.lower() in name_lower:
                score = 60
            elif any(word in name_lower for word in team.full_name.lower().split()):
                score = 50
            
            if score > best_score:
                best_score = score
                best_team = team
        
        return best_team if best_score >= 50 else None
    
    def _row_to_team(self, row) -> TeamMetadata:
        """Convert database row to TeamMetadata."""
        return TeamMetadata(
            team_id=row[0],
            sport=row[1],
            league=row[2],
            full_name=row[3],
            short_name=row[4],
            abbreviation=row[5],
            aliases=json.loads(row[6]) if row[6] else [],
            conference=row[7],
            division=row[8],
            city=row[9],
            country=row[10],
            founded_year=row[11],
            primary_color=row[12],
            secondary_color=row[13],
            venue=row[14],
            venue_capacity=row[15]
        )
    
    def enrich_market_data(self, market_data: Dict) -> Dict:
        """
        Enrich blockchain market data with team metadata.
        
        Args:
            market_data: Raw market data from blockchain
            
        Returns:
            Enriched market data
        """
        enriched = market_data.copy()
        
        # Parse game label
        game_label = market_data.get('game_label', '')
        
        # Common separators
        for sep in [' vs ', ' v ', ' @ ', ' - ']:
            if sep in game_label:
                parts = game_label.split(sep, 1)
                if len(parts) == 2:
                    home_name = parts[0].strip()
                    away_name = parts[1].strip()
                    
                    # Find teams
                    home_team = self.find_team(
                        home_name,
                        sport=market_data.get('sport'),
                        league=market_data.get('league')
                    )
                    
                    away_team = self.find_team(
                        away_name,
                        sport=market_data.get('sport'),
                        league=market_data.get('league')
                    )
                    
                    # Enrich with metadata
                    if home_team:
                        enriched['home_team'] = asdict(home_team)
                        enriched['home_team_id'] = home_team.team_id
                        enriched['home_team_name'] = home_team.full_name
                    else:
                        enriched['home_team_name'] = home_name
                        
                    if away_team:
                        enriched['away_team'] = asdict(away_team)
                        enriched['away_team_id'] = away_team.team_id
                        enriched['away_team_name'] = away_team.full_name
                    else:
                        enriched['away_team_name'] = away_name
                    
                    break
        
        return enriched
    
    def get_teams_by_league(self, league: str) -> List[TeamMetadata]:
        """Get all teams in a league."""
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT * FROM team_metadata
            WHERE LOWER(league) = LOWER(?)
            ORDER BY full_name
        """, (league,)).fetchall()
        
        conn.close()
        
        return [self._row_to_team(row) for row in rows]


def test_metadata_service():
    """Test the metadata service."""
    service = TeamMetadataService()
    
    # Test team finding
    test_cases = [
        ("Liverpool", "Soccer", "EPL"),
        ("Lakers", "Basketball", "NBA"),
        ("Cowboys", "Football", "NFL"),
        ("Man City", "Soccer", "EPL"),
    ]
    
    print("Testing Team Finding:")
    print("-" * 50)
    
    for name, sport, league in test_cases:
        team = service.find_team(name, sport, league)
        if team:
            print(f"{name} -> {team.full_name} ({team.abbreviation})")
            print(f"  Venue: {team.venue} (capacity: {team.venue_capacity:,})")
        else:
            print(f"{name} -> Not found")
    
    print("\nTesting Market Enrichment:")
    print("-" * 50)
    
    # Test market enrichment
    market = {
        'game_label': 'Liverpool vs Manchester City',
        'sport': 'Soccer',
        'league': 'EPL'
    }
    
    enriched = service.enrich_market_data(market)
    print(f"Original: {market['game_label']}")
    print(f"Home: {enriched.get('home_team_name')} - {enriched.get('home_team', {}).get('venue')}")
    print(f"Away: {enriched.get('away_team_name')} - {enriched.get('away_team', {}).get('venue')}")


if __name__ == "__main__":
    test_metadata_service()