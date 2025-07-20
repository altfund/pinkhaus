import requests
import pandas as pd
import sqlite3
import json
import time
from contextlib import closing
from fuzzywuzzy import fuzz
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from database import engine
from alembic.config import Config
from alembic import command


load_dotenv()  # loads from .env if present
## TODO
# - 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base API URLs
OVERTIME_API_URL = "https://overtimemarketsv2.xyz/overtime-v2/networks/10"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
OVERTIME_API_KEY = os.getenv("OVERTIME_API_KEY")


DB_NAME = "sport_odds.db"

# Update interval (in seconds)
UPDATE_INTERVAL = 60  # 1-5 minutes

def last_update_time(database, table_name):
    with sqlite3.connect(database) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_updated FROM table_metadata WHERE table_name = ?", (table_name,))
        result = cursor.fetchone()
        return pd.to_datetime(result[0]) if result and result[0] else None


# Helper function to determine if an update is needed
def needs_update(last_updated):
    if last_updated is None:
        return True
    return (pd.Timestamp.now() - last_updated).total_seconds() > UPDATE_INTERVAL

# Create or replace a table in SQLite
def create_or_replace_table(database, df, table_name):
    if df.empty:
        logging.warning(f"No data available to write to {table_name}. Skipping table update.")
        return
    
    with sqlite3.connect(database, timeout=10) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        # Ensure table_name is a valid string and timestamp is formatted correctlyz
        table_name_str = str(table_name)
        last_updated_str = pd.Timestamp.now().isoformat()
        
        conn.execute("REPLACE INTO table_metadata (table_name, last_updated) VALUES (?, ?)", 
                     (table_name_str, last_updated_str))

def upsert_table(database, df, table_name, key_columns):
    """Insert new records and update existing ones in the database."""
    
    df['updated_at'] = pd.Timestamp.now().isoformat()
    
    with sqlite3.connect(database, timeout=10) as conn:
        cursor = conn.cursor()
        
        # Ensure key_columns have a UNIQUE constraint
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        index_columns = []
        for index in indexes:
            cursor.execute(f"PRAGMA index_info({index[1]})")
            index_columns.extend([col[2] for col in cursor.fetchall()])
        
        if not all(col in index_columns for col in key_columns):
            raise ValueError(f"The ON CONFLICT key columns {key_columns} must have a UNIQUE or PRIMARY KEY constraint.")

        for _, row in df.iterrows():
            # Convert timestamps to ISO format and handle None values
            row = row.fillna("")  # Replace NaN with empty string
            row_dict = row.to_dict()
            row_dict["updated_at"] = pd.Timestamp.now().isoformat()

            columns = ", ".join(row_dict.keys())
            placeholders = ", ".join(["?"] * len(row_dict))
            update_columns = ", ".join([f"{col} = ?" for col in row_dict.keys() if col not in key_columns])

            sql = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
                ON CONFLICT ({", ".join(key_columns)}) 
                DO UPDATE SET {update_columns}
            """

            values = tuple(row_dict.values()) + tuple(row_dict[col] for col in row_dict.keys() if col not in key_columns)
            
            try:
                cursor.execute(sql, values)
            except sqlite3.InterfaceError as e:
                logging.error(f"Error binding parameters for {table_name}: {e}")
                logging.error(f"Query: {sql}")
                logging.error(f"Values: {values}")
                raise
        conn.commit()



# Fetch all Overtime market odds
def get_all_overtime_markets():
    try:
        logging.info("Fetching all Overtime Markets data...")
        response = requests.get(f"{OVERTIME_API_URL}/markets")
        response.raise_for_status()

        data = response.json()
        return(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Overtime Markets: {e}")
        
        
def get_overtime_markets_markets(overtime_all_json):
    results = []
    for sport, leagues in overtime_all_json.items():
        for league_id, markets in leagues.items():
            for market in markets:
                if isinstance(market, dict):
                    results.append({
                        "source": "overtime_markets",
                        "source_id": market.get("gameId"),
                        "sport": sport,
                        "league_name": market.get("leagueName"),
                        "market_type": market.get("type"),
                        "home_team": market.get("homeTeam"),
                        "away_team": market.get("awayTeam"),
                        "maturity_date": market.get("maturityDate")
                    })
    return pd.DataFrame(results)


def get_overtime_markets_odds(overtime_all_json):
    results = []

    def extract_odds(market, parent_game_id=None):
        game_id = market.get("gameId", parent_game_id)
        market_type = market.get("type")
        line = market.get("line", None)
        position_names = market.get("positionNames", [])

        for i, odd in enumerate(market.get("odds", [])):
            outcome_label = position_names[i] if i < len(position_names) else f"option_{i+1}"

            results.append({
                "bookmaker": "overtime_markets",
                "source_id": game_id,
                "source": "overtime_markets",
                "market_type": market_type,
                "line": line,
                "outcome": outcome_label,
                "american_odds": odd.get("american"),
                "decimal_odds": odd.get("decimal"),
                "normalized_implied": odd.get("normalizedImplied")
            })

        # Recursively extract child market odds
        for child in market.get("childMarkets", []):
            extract_odds(child, game_id)

    for sport, leagues in overtime_all_json.items():
        for league_id, markets in leagues.items():
            for market in markets:
                if isinstance(market, dict):
                    extract_odds(market)

    return pd.DataFrame(results)

def get_overtime_games_flat(games_json):
    rows = []
    for league, games in games_json.items():
        for g in games:
            # unpack the two teams
            teams = g.get("teams", [])
            home = next((t for t in teams if t["isHome"]), {})
            away = next((t for t in teams if not t["isHome"]), {})

            rows.append({
                "game_id":               g["gameId"],
                "start_time":            g.get("startTime"),
                "last_update":           datetime.fromtimestamp(g["lastUpdate"]/1e3),
                "game_status":           g.get("gameStatus"),
                "is_finished":           g.get("isGameFinished"),
                "tournament":            g.get("tournamentName"),
                "tournament_round":      g.get("tournamentRound"),

                "home_score":            home.get("score"),
                "away_score":            away.get("score"),

                "home_score_by_period":  json.dumps(home.get("scoreByPeriod", [])),
                "away_score_by_period":  json.dumps(away.get("scoreByPeriod", [])),
            })
    return pd.DataFrame(rows)

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

# Fetch events data from Odds API using the /events endpoint
def fetch_events(sport_key, api_key=ODDS_API_KEY):
    try:
        url = f"{ODDS_API_URL}/{sport_key}/events/?apiKey={api_key}&regions=eu"
        logging.info(f"Fetching events using the Odds API /events endpoint for sport_key={sport_key}...")
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 422:
            logging.warning(f"Skipping unsupported sport_key: {sport_key}. Full URL: {url}")
        else:
            logging.error(f"HTTP Error fetching events from Odds API: {http_err}. Full URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching events from Odds API /events endpoint: {e}. Full URL: {url}")
        return None

# Fetch and update Odds API data using /events endpoint
def get_odds_api_markets():
    logging.info("Fetching Odds API markets...")
    sports = fetch_sports(ODDS_API_KEY)
    all_events = []
    for sport in sports:
        if sport.get("active"):
            sport_key = sport.get("key")
            events_data = fetch_events(sport_key)
            if events_data:
                for event in events_data:
                    all_events.append({
                        "source": "odds_api",
                        "source_id": event.get("id"),
                        "sport": event.get("sport_key"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "maturity_date": event.get("commence_time")
                    })
    
    odds_api_df = pd.DataFrame(all_events)
    return(odds_api_df)


def insert_odds(database, df):
    """Insert all odds data without replacing previous entries."""
    with sqlite3.connect(database, timeout=10) as conn:
        df.to_sql("odd", conn, if_exists="append", index=False)
        
def upsert_teams_from_games(conn, games_df):
    # build one row per team
    now = datetime.utcnow()
    teams = []
    for _, g in games_df.iterrows():
        teams.append({
            "team_name":  g["home_team"],
            "league":     g["league"],
            "updated_at": now
        })
        teams.append({
            "team_name":  g["away_team"],
            "league":     g["league"],
            "updated_at": now
        })
    teams_df = pd.DataFrame(teams).drop_duplicates(subset=["team_name"])

    # use your generic upsert helper
    upsert_table(
        conn,
        table_name="team",
        df=teams_df,
        primary_key="team_name",
        # on conflict, update league & timestamp
        update_columns=["league", "updated_at"]
    )
    
def fetch_overtime_games():
    """
    Fetches the full games-info payload from Overtime V2.
    Returns the JSON as a dict: { league_name: [ gameObj, ... ], ... }
    """
    url = f"{OVERTIME_API_URL}/games-info"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def initialize_database():
    here = os.path.dirname(__file__)
    cfg  = Config(os.path.join(here, "alembic.ini"))
    command.upgrade(cfg, "head")


# Main function
if __name__ == "__main__":
    # Initialize database
    initialize_database()

    # Update Overtime Markets
    last_updated = last_update_time(DB_NAME, "market")
    
    if needs_update(last_updated):
        logging.info("Updating Overtime Markets data...")
        
        overtime_market_all_json = get_all_overtime_markets()
        
        if overtime_market_all_json:
            
            logging.info("Updating Overtime Markets markets...")
            
            overtime_markets_markets_df = get_overtime_markets_markets(overtime_all_json=overtime_market_all_json)
            upsert_table(DB_NAME, overtime_markets_markets_df, "market", key_columns=["source_id"])
            
            logging.info("Updating Overtime Markets odds...")
            
            overtime_markets_odds_df = get_overtime_markets_odds(overtime_all_json=overtime_market_all_json) #overtime_market_all_df
            insert_odds(DB_NAME, overtime_markets_odds_df)
            
            logging.info("Updating Overtime Markets games...")
            # 1) Pull games from the API and flatten
            games_json = fetch_overtime_games()
            games_df   = get_overtime_games_flat(games_json)
            with sqlite3.connect(DB_NAME) as conn:
                # 2) Load your existing markets from the DB
                markets_df = pd.read_sql(
                    "SELECT * FROM market",
                    con=conn,
                    parse_dates=["maturity_date", "updated_at"]  # adjust as needed
                )
                
                # 3) Merge games onto markets by source_id == game_id
                enriched = (
                    markets_df
                    .merge(
                        games_df,
                        left_on="source_id",
                        right_on="game_id",
                        how="left",
                        suffixes=("", "_game")  # if you want to keep both versions
                    )
                    .drop(columns=["game_id"])  # drop the duplicate
                )
                
                # 4) Update the timestamp for this enrichment run
                enriched["updated_at"] = datetime.utcnow()
                to_update = enriched[enriched["game_id"].notna()]
                
                logging.info("Updating Overtime Markets game statuses...")
                # 5) Upsert the enriched DataFrame back into your market table
                upsert_table(
                    conn,
                    table_name="market",
                    df=to_update,
                    primary_key="source_id",
                    update_columns=[
                        "start_time",
                        "last_update",
                        "game_status",
                        "is_finished",
                        "tournament",
                        "tournament_round",
                        "home_score",
                        "away_score",
                        "home_score_by_period",
                        "away_score_by_period",
                        "updated_at",
                        # plus any status fields you already have
                    ]
                )
            
                logging.info("Updating Overtime Markets teams...")
                upsert_teams_from_games(conn, games_df)
            
            logging.info("Overtime Markets data updated successfully.")
            
        
        # Update Odds API data
        logging.info("Updating Odds API data...")
        odds_api_df = get_odds_api_markets()
        if not odds_api_df.empty:
            upsert_table(DB_NAME, odds_api_df, "market", key_columns=["source_id"])
            logging.info("Odds API data updated successfully.")
        
        
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("REPLACE INTO table_metadata (table_name, last_updated) VALUES (?, ?)", 
                         ("market", pd.Timestamp.now().isoformat()))
            logging.info("Markets and events updated.")
            
    else:
        logging.info("Markets table was updated too recently; no update performed.")
        #logging.info("Overtime Markets data is up-to-date; no update performed.")
        #logging.info("Odds API data is up-to-date; no update performed.")

 
