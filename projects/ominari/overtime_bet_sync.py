#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 23:20:28 2025

@author: ess
"""

import sqlite3
from pathlib import Path

# Connect to existing database
db_path = Path("sport_odds.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the `bet` table with relations where possible
cursor.execute("""
CREATE TABLE IF NOT EXISTS bet (
    id TEXT PRIMARY KEY,
    account TEXT,
    timestamp INTEGER,
    buy_in_amount REAL,
    fees REAL,
    payout REAL,
    num_of_markets INTEGER,
    collateral TEXT,
    is_resolved BOOLEAN,
    is_user_the_winner BOOLEAN,
    is_exercisable BOOLEAN,
    is_claimable BOOLEAN,
    is_open BOOLEAN,
    final_payout REAL,
    is_live BOOLEAN,
    expiry INTEGER,
    match_id TEXT,
    market_type TEXT,
    position INTEGER,
    line REAL,
    decimal_odds REAL,
    normalized_implied REAL,
    FOREIGN KEY (match_id) REFERENCES match(overtime_source_id)
)
""")

conn.commit()
conn.close()

"Table 'bet' created with foreign key to 'match.overtime_source_id'"

import requests
import sqlite3
from datetime import datetime
from eth_utils import to_checksum_address


# Constants
DB_PATH = "sport_odds.db"
WALLET_ADDRESS = "0xE271ae7C87e87c2Ef897c6113999aE7A462bE2D2"
WALLET_ADDRESS = WALLET_ADDRESS.lower()
WALLET_ADDRESS = to_checksum_address(WALLET_ADDRESS)
network_id = 10
API_URL = f"https://overtimemarketsv2.xyz/overtime-v2/networks/{network_id}/users/{WALLET_ADDRESS}/history"


def fetch_overtime_portfolio():
    response = requests.get(API_URL)
    response.raise_for_status()
    return response.json()


def timestamp_to_iso(ms):
    return datetime.utcfromtimestamp(ms / 1000).isoformat()


def insert_bets(bets_by_type):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for status, bets in bets_by_type.items():
        for bet in bets:
            bet_id = bet["id"]
            ts = timestamp_to_iso(bet["timestamp"])
            expiry = timestamp_to_iso(bet["expiry"])
            payout = bet["payout"]
            buy_in = bet["buyInAmount"]
            fees = bet.get("fees", 0)
            quote = bet.get("totalQuote")
            collateral = bet.get("collateral")
            account = bet.get("account")
            is_winner = bet.get("isUserTheWinner", False)
            is_claimable = bet.get("isClaimable", False)
            is_lost = bet.get("isLost", False)
            final_payout = bet.get("finalPayout", 0)
            is_resolved = bet.get("isResolved", False)

            for market in bet["sportMarkets"]:
                match_id = market["gameId"]
                league = market["leagueName"]
                sport = market["sport"]
                market_type = market["type"]
                position = market.get("position", None)
                line = market.get("line", None)
                odds = market["odd"]["decimal"]
                implied = market["odd"]["normalizedImplied"]
                home_team = market.get("homeTeam")
                away_team = market.get("awayTeam")
                home_score = market.get("homeScore")
                away_score = market.get("awayScore")
                maturity = timestamp_to_iso(market["maturity"])
                resolved = market.get("isResolved", False)
                cancelled = market.get("isCancelled", False)
                game_status = market.get("gameStatus")

                cur.execute(
                    """
                    INSERT OR REPLACE INTO bet (
                        bet_id, bet_timestamp, expiry, buy_in_amount, fees, total_quote, payout,
                        collateral, account, is_user_winner, is_claimable, is_lost, is_resolved, final_payout,
                        market_game_id, market_type, position, odds_decimal, normalized_implied,
                        home_team, away_team, home_score, away_score, maturity_date, market_status,
                        sport, league, match_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        bet_id,
                        ts,
                        expiry,
                        buy_in,
                        fees,
                        quote,
                        payout,
                        collateral,
                        account,
                        int(is_winner),
                        int(is_claimable),
                        int(is_lost),
                        int(is_resolved),
                        final_payout,
                        match_id,
                        market_type,
                        position,
                        odds,
                        implied,
                        home_team,
                        away_team,
                        home_score,
                        away_score,
                        maturity,
                        game_status,
                        sport,
                        league,
                        match_id,  # also link to match(overtime_source_id)
                    ),
                )

    conn.commit()
    conn.close()
    print("✅ Bet data successfully imported into sport_odds.db")


def main():
    print("⏳ Fetching Overtime user portfolio...")
    data = fetch_overtime_portfolio()

    bets_by_type = {
        "open": data.get("open", []),
        "claimable": data.get("claimable", []),
        "closed": data.get("closed", []),
    }

    print(
        f"🔢 Found {sum(len(v) for v in bets_by_type.values())} bets across all states."
    )
    insert_bets(bets_by_type)


if __name__ == "__main__":
    main()
