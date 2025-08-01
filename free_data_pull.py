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

def upsert_table(database, df, table_name, key_columns, update_columns=None):
    """
    Insert new records and update existing ones in the database.

    :param database: str, path to the sqlite database file
    :param df: pandas.DataFrame, the rows to upsert
    :param table_name: str, name of the target table
    :param key_columns: list of str, columns to use in the ON CONFLICT clause
    :param update_columns: list of str or None, columns to overwrite on conflict;
                           if None, all columns except key_columns will be updated
    """
    # Make a working copy and stamp each row with a new updated_at
    df = df.copy()
    df['updated_at'] = pd.Timestamp.now()

    with sqlite3.connect(database, timeout=10) as conn:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            # 1. Turn the row into a dict, replacing NaN with empty string
            row_dict = row.fillna("").to_dict()

            # 2. Convert any timestamp/datetime to ISO strings
            for col, val in row_dict.items():
                if hasattr(val, "isoformat"):
                    row_dict[col] = val.isoformat()

            cols = list(row_dict.keys())
            placeholders = ", ".join("?" for _ in cols)

            # 3. Decide which columns to update on conflict
            if update_columns is None:
                uc = [c for c in cols if c not in key_columns]
            else:
                uc = update_columns

            update_clause = ", ".join(f"{c} = ?" for c in uc)

            sql = f"""
            INSERT INTO {table_name} ({', '.join(cols)})
            VALUES ({placeholders})
            ON CONFLICT ({', '.join(key_columns)})
            DO UPDATE SET {update_clause}
            """

            # 4. Build the parameter list: first INSERT values, then UPDATE values
            values = [row_dict[c] for c in cols] + [row_dict[c] for c in uc]

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
        url = f"{OVERTIME_API_URL}/markets"
        headers = {"x-api-key": OVERTIME_API_KEY}
        logging.info(f"Fetching Overtime Markets from {url}")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
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
                        "maturity_date": market.get("maturityDate"),
                        "position_names": json.dumps(market.get("positionNames") or [])
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
                "position": i,              # ← new
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
    """
    Flatten the Overtime “games-info” payload (a dict of gameId→info)
    into a DataFrame-friendly list of dicts matching your market schema.
    """
    rows = []

    for game_id, info in games_json.items():
        if not isinstance(info, dict):
            # skip any unexpected values
            continue

        # Convert lastUpdate (ms since epoch) to datetime, if present
        last_update = None
        if info.get("lastUpdate") is not None:
            try:
                last_update = datetime.fromtimestamp(info["lastUpdate"] / 1000)
            except Exception:
                last_update = None

        # Base row
        row = {
            "source_id":            game_id,                   # note: merge into market.source_id
            "last_update":          last_update,
            "game_status":          info.get("gameStatus"),
            "is_finished":          info.get("isGameFinished"),
            "tournament":           info.get("tournamentName") or None,
            "tournament_round":     info.get("tournamentRound") or None,

            # we'll fill these in from the teams array
            "home_team":            None,
            "away_team":            None,
            "home_score":           None,
            "away_score":           None,
            "home_score_by_period": None,
            "away_score_by_period": None,
        }

        # Unpack the two teams
        for t in info.get("teams", []):
            if not isinstance(t, dict) or "name" not in t:
                continue

            name = t["name"]
            score = t.get("score")  # may be None if not finished
            periods = t.get("scoreByPeriod", [])

            if t.get("isHome"):
                row["home_team"] = name
                row["home_score"] = score
                row["home_score_by_period"] = json.dumps(periods)
            else:
                row["away_team"] = name
                row["away_score"] = score
                row["away_score_by_period"] = json.dumps(periods)

        rows.append(row)

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

def upsert_teams_from_markets(conn, markets_df):
    now   = datetime.utcnow()
    teams = []

    for _, m in markets_df.iterrows():
        home, away = m["home_team"], m["away_team"]
        league      = m.get("league_name")

        teams.append({
            "team_name":  home,
            "league":     league,
            "updated_at": now
        })
        teams.append({
            "team_name":  away,
            "league":     league,
            "updated_at": now
        })

    teams_df = (
        pd.DataFrame(teams)
          .drop_duplicates(subset=["team_name"])
    )

    upsert_table(
        database=DB_NAME,
        df=teams_df,
        table_name="team",
        key_columns=["team_name"],
        update_columns=["league","updated_at"]
    )

        
def upsert_teams_from_games(conn, games_df):
    # Deprecated, gettingt
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
        DB_NAME,
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
    url = f"{OVERTIME_BASE_URL}/games-info"
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
            
            logging.info("Updating Overtime teams...")
            upsert_teams_from_markets(DB_NAME, overtime_markets_markets_df)
            
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
                
                
                # strip out any old game columns first
                game_fields = [
                    "last_update", "game_status", "is_finished",
                    "tournament", "tournament_round",
                    "home_score", "away_score",
                    "home_score_by_period", "away_score_by_period"
                ]
                trimmed = markets_df.drop(columns=game_fields, errors="ignore")
                
                # now do a plain left-join by source_id
                enriched = trimmed.merge(games_df[game_fields + ['source_id']], 
                                         on="source_id", how="left")

                
                
                # 4) Update the timestamp for this enrichment run
                enriched["updated_at"] = datetime.utcnow()
                to_update = enriched[enriched["source_id"].notna()]
                
                logging.info("Updating Overtime Markets game statuses...")
                # 5) Upsert the enriched DataFrame back into your market table
                upsert_table(
                    DB_NAME,
                    table_name="market",
                    df=to_update,
                    key_columns=["source_id"],
                    update_columns=game_fields + ["updated_at"]
                )
            
                logging.info("Updating Overtime Markets teams...")
                #upsert_teams_from_games(DB_NAME, games_df)
            
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

 
