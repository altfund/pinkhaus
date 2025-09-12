import requests
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from alembic.config import Config
from alembic import command
from database_utils import upsert_records_orm
from sqlalchemy import create_engine, text
from typing import Dict, Optional


load_dotenv()  # loads from .env if present

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the base API URLs
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
OVERTIME_API_KEY = os.getenv("OVERTIME_API_KEY")

# Base URL and network ID
OVERTIME_BASE_URL = "https://api.overtime.io/overtime-v2"
OVERTIME_NETWORK_ID = os.getenv("OVERTIME_NETWORK_ID", "10")
OVERTIME_API_URL = f"{OVERTIME_BASE_URL}/networks/{OVERTIME_NETWORK_ID}"

DB_NAME = "sport_odds.db"

# Update interval (in seconds)
UPDATE_INTERVAL = 60  # 1-5 minutes


class NormalizedDataIngestion:
    """Data ingestion that uses the normalized schema."""
    
    def __init__(self, db_path="sport_odds.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.lookup_cache = {}
        self._load_lookups()
    
    def _load_lookups(self):
        """Load lookup tables into memory cache."""
        with self.engine.connect() as conn:
            # Load bookmakers
            result = conn.execute(text("SELECT id, name FROM lu_bookmakers"))
            self.lookup_cache['bookmaker'] = {name: id for id, name in result}
            
            # Load sources
            result = conn.execute(text("SELECT id, name FROM lu_sources"))
            self.lookup_cache['source'] = {name: id for id, name in result}
            
            # Load market types
            result = conn.execute(text("SELECT id, name FROM lu_market_types"))
            self.lookup_cache['market_type'] = {name: id for id, name in result}
    
    def _ensure_lookup_value(self, table: str, value: str) -> int:
        """Ensure a value exists in lookup table and return its ID."""
        if value in self.lookup_cache[table]:
            return self.lookup_cache[table][value]
        
        # Insert new value
        with self.engine.begin() as conn:
            if table == 'bookmaker':
                conn.execute(text("INSERT OR IGNORE INTO lu_bookmakers (name) VALUES (:name)"), {"name": value})
            elif table == 'source':
                conn.execute(text("INSERT OR IGNORE INTO lu_sources (name) VALUES (:name)"), {"name": value})
            elif table == 'market_type':
                conn.execute(text("INSERT OR IGNORE INTO lu_market_types (name) VALUES (:name)"), {"name": value})
            
            # Reload cache for this table
            if table == 'bookmaker':
                result = conn.execute(text("SELECT id, name FROM lu_bookmakers WHERE name = :name"), {"name": value})
            elif table == 'source':
                result = conn.execute(text("SELECT id, name FROM lu_sources WHERE name = :name"), {"name": value})
            elif table == 'market_type':
                result = conn.execute(text("SELECT id, name FROM lu_market_types WHERE name = :name"), {"name": value})
            
            row = result.fetchone()
            if row:
                self.lookup_cache[table][value] = row[0]
                return row[0]
            
            raise ValueError(f"Failed to insert {value} into {table}")
    
    def insert_odds_normalized(self, odds_data):
        """Insert odds using the normalized schema."""
        records_to_insert = []
        
        for odd in odds_data:
            try:
                # Get lookup IDs
                bookmaker_id = self._ensure_lookup_value('bookmaker', odd['bookmaker'])
                source_id = self._ensure_lookup_value('source', odd['source'])
                market_type_id = self._ensure_lookup_value('market_type', odd['market_type'])
                
                # Map outcome to ID
                outcome_mapping = {
                    'option_1': 0, 'option_2': 1, 'option_3': 2,
                    'home': 0, 'draw': 1, 'away': 2,
                    'yes': 0, 'no': 1,
                    'over': 0, 'under': 1
                }
                
                outcome = odd.get('outcome', '').lower()
                outcome_id = outcome_mapping.get(outcome)
                
                if outcome_id is None:
                    # Try numeric parsing
                    try:
                        outcome_id = int(outcome.replace('option_', '')) - 1
                    except:
                        logging.warning(f"Unknown outcome: {outcome}")
                        continue
                
                # Convert timestamp
                if isinstance(odd.get('updated_at'), str):
                    timestamp = int(pd.to_datetime(odd['updated_at']).timestamp())
                else:
                    timestamp = int(datetime.now().timestamp())
                
                # Prepare record
                record = {
                    'market_id': odd['source_id'],
                    'bookmaker_id': bookmaker_id,
                    'source_id': source_id,
                    'market_type_id': market_type_id,
                    'outcome_id': outcome_id,
                    'position': odd.get('position', 0),
                    'line_x100': int(odd['line'] * 100) if odd.get('line') else None,
                    'decimal_odds_x1000': int(odd['decimal_odds'] * 1000) if odd.get('decimal_odds') else None,
                    'american_odds': int(odd['american_odds']) if odd.get('american_odds') else None,
                    'implied_x10000': int(odd['normalized_implied'] * 10000) if odd.get('normalized_implied') else None,
                    'updated_at': timestamp
                }
                
                records_to_insert.append(record)
                
            except Exception as e:
                logging.error(f"Error processing odd: {e}, data: {odd}")
                continue
        
        # Bulk insert
        if records_to_insert:
            with self.engine.begin() as conn:
                # Use INSERT OR IGNORE to handle duplicates
                insert_sql = text("""
                    INSERT OR IGNORE INTO odds_normalized 
                    (market_id, bookmaker_id, source_id, market_type_id, outcome_id, 
                     position, line_x100, decimal_odds_x1000, american_odds, implied_x10000, updated_at)
                    VALUES 
                    (:market_id, :bookmaker_id, :source_id, :market_type_id, :outcome_id,
                     :position, :line_x100, :decimal_odds_x1000, :american_odds, :implied_x10000, :updated_at)
                """)
                
                # Insert in chunks to avoid parameter limits
                chunk_size = 1000
                for i in range(0, len(records_to_insert), chunk_size):
                    chunk = records_to_insert[i:i + chunk_size]
                    for record in chunk:
                        conn.execute(insert_sql, record)
                
                logging.info(f"Inserted {len(records_to_insert)} odds records")


# Keep existing helper functions
def last_update_time(database, table_name):
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_updated FROM table_metadata WHERE table_name = ?",
            (table_name,),
        )
        result = cursor.fetchone()
        return pd.to_datetime(result[0]) if result and result[0] else None


def needs_update(last_updated):
    if last_updated is None:
        return True
    return (pd.Timestamp.now() - last_updated).total_seconds() > UPDATE_INTERVAL


def upsert_table(database, df, table_name, key_columns, update_columns=None):
    """
    Insert new records and update existing ones in the database.
    """
    df = df.copy()
    df["updated_at"] = pd.Timestamp.now()

    with sqlite3.connect(database, timeout=10) as conn:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            row_dict = row.fillna("").to_dict()

            # Build the INSERT ... ON CONFLICT ... DO UPDATE statement
            cols = list(row_dict.keys())
            placeholders = ", ".join([f":{col}" for col in cols])
            columns_str = ", ".join(cols)

            # Determine which columns to update on conflict
            if update_columns is None:
                update_cols = [c for c in cols if c not in key_columns]
            else:
                update_cols = update_columns

            set_clause = ", ".join([f"{col} = excluded.{col}" for col in update_cols])

            conflict_clause = ", ".join(key_columns)

            sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT({conflict_clause}) DO UPDATE SET
                {set_clause}
            """

            cursor.execute(sql, row_dict)


# Fetch all Overtime market odds
def get_all_overtime_markets():
    try:
        response = requests.get(
            f"{OVERTIME_API_URL}/markets?active=true",
            headers={"x-api-key": OVERTIME_API_KEY},
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Overtime Markets: {e}")
        return None


def get_overtime_markets_markets(overtime_all_json):
    results = []

    def extract_market(market, parent_game_id=None):
        game_id = market.get("gameId", parent_game_id)
        
        results.append(
            {
                "source": "overtime",
                "source_id": market.get("id"),
                "sport": market.get("sport"),
                "home_team": market.get("homeTeam"),
                "away_team": market.get("awayTeam"),
                "maturity_date": pd.to_datetime(market.get("maturityDate")),
                "league_name": market.get("leagueName"),
                "market_type": market.get("type"),
                "odds": json.dumps(market.get("odds", [])),
            }
        )

        # Recursively extract child markets
        for child in market.get("childMarkets", []):
            extract_market(child, game_id)

    for sport, leagues in overtime_all_json.items():
        for league, markets in leagues.items():
            for market in markets:
                if isinstance(market, dict):
                    extract_market(market)

    return pd.DataFrame(results)


def get_overtime_markets_odds(overtime_all_json):
    results = []

    def extract_odds(market, parent_game_id=None):
        game_id = market.get("gameId", parent_game_id)
        market_type = market.get("type")
        market_id = market.get("id")
        position_names = market.get("positionNames", [])

        for i, odd in enumerate(market.get("odds", [])):
            outcome_label = (
                position_names[i] if i < len(position_names) else f"option_{i + 1}"
            )

            results.append(
                {
                    "source": "overtime",
                    "source_id": market_id,
                    "market_type": market_type,
                    "bookmaker": "overtime",
                    "outcome": outcome_label,
                    "position": i,
                    "american_odds": odd.get("american"),
                    "decimal_odds": odd.get("decimal"),
                    "normalized_implied": odd.get("normalizedImplied"),
                }
            )

        # Recursively extract child market odds
        for child in market.get("childMarkets", []):
            extract_odds(child, game_id)

    for sport, leagues in overtime_all_json.items():
        for league, markets in leagues.items():
            for market in markets:
                if isinstance(market, dict):
                    extract_odds(market)

    return results  # Return list, not DataFrame


# Fetch sports data from Odds API
def fetch_sports(api_key):
    try:
        url = f"{ODDS_API_URL}?apiKey={api_key}"
        logging.info("Fetching sports data from Odds API...")
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sports data from Odds API: {e}")
        return []


def fetch_overtime_games():
    """
    Fetches the full games-info payload from Overtime V2.
    Returns the JSON as a dict: { league_name: [ gameObj, ... ], ... }
    """
    url = f"{OVERTIME_BASE_URL}/games-info"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def initialize_database():
    here = os.path.dirname(__file__)
    cfg = Config(os.path.join(here, "alembic.ini"))
    command.upgrade(cfg, "head")


def main():
    """Main entry point for the data collector using normalized schema."""
    # Initialize database
    initialize_database()
    
    # Initialize normalized ingestion
    ingestion = NormalizedDataIngestion(DB_NAME)
    
    # Run the data update functions
    last_updated = last_update_time(DB_NAME, "market")

    if needs_update(last_updated):
        logging.info("Updating market data...")

        # Fetch Overtime markets
        overtime_market_all_json = get_all_overtime_markets()

        if overtime_market_all_json:
            logging.info("Processing Overtime Markets...")

            # Update markets (still using regular table)
            overtime_markets_markets_df = get_overtime_markets_markets(
                overtime_all_json=overtime_market_all_json
            )
            upsert_table(
                DB_NAME,
                overtime_markets_markets_df,
                "market",
                key_columns=["source_id"],
            )

            # Get odds and insert using normalized schema
            logging.info("Inserting Overtime odds using normalized schema...")
            overtime_odds = get_overtime_markets_odds(
                overtime_all_json=overtime_market_all_json
            )
            ingestion.insert_odds_normalized(overtime_odds)

            # Update games info
            logging.info("Updating Overtime Markets games...")
            games_json = fetch_overtime_games()
            # ... rest of game processing logic remains same ...

            logging.info("Overtime Markets data updated successfully.")

        # Update table metadata
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "REPLACE INTO table_metadata (table_name, last_updated) VALUES (?, ?)",
                ("market", pd.Timestamp.now().isoformat()),
            )
            logging.info("Markets updated.")

    else:
        logging.info("Markets table was updated too recently; no update performed.")


if __name__ == "__main__":
    main()