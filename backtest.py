#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 02:19:02 2025

@author: ess
"""

from datetime import datetime, timezone
from evaluate_open_markets import *
from evaluate_open_markets import _init_external_stub

#_external_stub = _init_external_stub()

as_of = datetime(2025, 7, 24, 14, 0, tzinfo=timezone.utc)
min_break_minutes=5.0
avg_game_duration_minutes=120.0
abs_game_limit = 100

SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    #ExternalGrpcSignal(_external_stub),
]

SIGNAL_WEIGHTS = {
    "implied_raw": 1.0,    # base implied probability
    #"external":    1.0,    # weight for external model
}

# define the chunk size
# print("getting market chunk")
# open_markets = get_upcoming_overtime_markets_with_signals(
#     limit=abs_game_limit,
#     as_of=as_of,
#     min_break_minutes=min_break_minutes,
#     avg_game_duration_minutes=avg_game_duration_minutes
#     )

# print(f"[DEBUG] open_markets → {len(open_markets)} rows")

# match_df = summarize_match_schedule_from_open_markets(open_markets)
# breaks_df = find_upcoming_game_breaks(
#     match_df,
#     min_break_minutes=min_break_minutes,
#     avg_game_duration_minutes=avg_game_duration_minutes,
#     now=as_of
# )
# game_times_df = extract_active_game_periods_from_breaks(match_df, breaks_df)
# # EARLY EXIT if there are no games to report
# if game_times_df.empty:
#     print("No upcoming games.")


# first_chunk = game_times_df.iloc[0]
# start, end = first_chunk["chunk_start"], first_chunk["chunk_end"]

# filtered = open_markets.copy()
# filtered["maturity_date"] = pd.to_datetime(filtered["maturity_date"], errors="coerce")
# filtered = filtered[(filtered["maturity_date"] >= start) & (filtered["maturity_date"] <= end)]

# print(start)
# print(end)
# #print(filtered.rows())

# print(f"[DEBUG] filtered has {len(filtered)} rows") #"; providers = {[p.name for p in signal_providers]}")


generate_betting_session_report_and_save(
            execution_bankroll=1000,
            avg_game_duration_minutes=avg_game_duration_minutes,
            min_break_minutes=min_break_minutes,
            abs_game_limit=abs_game_limit,
            base_dir="backtests",
            signal_providers=SIGNAL_PROVIDERS,
            signal_weights=SIGNAL_WEIGHTS,
            as_of=as_of
            )

# save bets in json

# make market primary key consist of:
    # game_id/source_id
    # type_id
    # line (0 if none)
    # player (0 if none)
    # position (odds position in return list)
    
# make trade/bet/wager table
    # connect to market table
    # if backtest just record and save source and any other data needed
    # if paper trade get quote from overtime and record it
    # if live, get quote and then execute trade
    
# make backtest subfolders
# iterate chunks based on end dates of prior backtest betting reports
# determine dates to backtest within (R&D, out of sample, etc.)

# compose multiple backtests into a weight on them based on performance, correlation etc. per Carver
# 