#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 01:39:20 2025

@author: ess
"""

# performance.py
import pandas as pd
from sqlalchemy import create_engine, text, select, func
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from models import Bet, BettingSession, Market

# adjust to your DB path
from database import engine


def get_latest_session_ids(
    session_type: Optional[str] = None, only_this: Optional[str] = None
) -> List[int]:
    """
    Return the latest (max id) session_id for each unique strategy_name,
    optionally filtering by session_type (e.g. 'backtest').
    """
    with Session(engine) as sess:
        q = sess.query(func.max(BettingSession.id).label("latest_id"))
        if session_type:
            q = q.filter(BettingSession.session_type == session_type)
        q = q.group_by(BettingSession.strategy_name)

        if only_this:
            q = q.filter(BettingSession.strategy_name == only_this)

        return [row.latest_id for row in q.all()]


def load_bets_with_results(session_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Pull bets JOINed to their market’s computed `resolved_outcome`.
    """
    with Session(engine) as sess:
        stmt = (
            select(
                BettingSession.id.label("session_id"),
                BettingSession.as_of.label("as_of"),
                BettingSession.session_type,
                BettingSession.strategy_name,
                Bet.id.label("bet_id"),
                Bet.source_id,
                Bet.odds,
                Bet.execution_stake,
                Bet.fee_amount,
                Bet.normalized_outcome.label("bet_outcome"),
                Market.resolved_outcome.label("market_outcome"),
            )
            .join(Bet, Bet.session_id == BettingSession.id)
            .join(Market, Market.source_id == Bet.source_id, isouter=True)
        )
        if session_ids:
            stmt = stmt.where(BettingSession.id.in_(session_ids))

        df = pd.read_sql(stmt, sess.bind)
    return df


def score_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw bet/result DataFrame, compute:
      - status: 'win' / 'loss' / 'pending'
      - net: P&L per bet, zero for pending
    """

    def _score(row):
        if pd.isna(row.market_outcome):
            return pd.Series({"status": "pending", "net": 0.0})
        elif row.bet_outcome == row.market_outcome:
            gross = row.execution_stake * (row.odds - 1)
            net = gross - row.fee_amount
            return pd.Series({"status": "win", "net": net})
        else:
            net = -row.execution_stake - row.fee_amount
            return pd.Series({"status": "loss", "net": net})

    scored = df.join(df.apply(_score, axis=1))
    return scored


def summarize_performance(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
      1) per-bet: the scored bets
      2) per-session: aggregated metrics by session_id, strategy_name, session_type
    """
    # 1) per-session aggregation
    agg = (
        scored.groupby(
            ["session_id", "strategy_name", "session_type", "as_of"], dropna=False
        )
        .agg(
            total_staked=("execution_stake", "sum"),
            total_fee=("fee_amount", "sum"),
            total_net=("net", "sum"),
            wins=("status", lambda s: (s == "win").sum()),
            losses=("status", lambda s: (s == "loss").sum()),
            pending=("status", lambda s: (s == "pending").sum()),
            roi=(
                "net",
                lambda x: x.sum() / scored.loc[x.index, "execution_stake"].sum()
                if scored.loc[x.index, "execution_stake"].sum()
                else 0.0,
            ),
        )
        .reset_index()
    )

    return scored, agg


def summarize_backtest_performance(
    strat_name: Optional[str] = None,
    session_ids: Optional[List[int]] = None,
    print_per_session: bool = True,
) -> pd.DataFrame:
    """
    Summarizes performance for a backtest. You may supply either:
      - strat_name: to fetch all sessions with that strategy_name and session_type='backtest',
      - session_ids: an explicit list of session IDs.

    Returns per-session performance DataFrame. Also prints summaries if requested.
    """
    # Determine session IDs
    if strat_name:
        with Session(engine) as sess:
            stmt = select(BettingSession.id).where(
                BettingSession.strategy_name == strat_name,
                BettingSession.session_type == "backtest",
            )
            session_ids = sess.execute(stmt).scalars().all()
    if not session_ids:
        raise ValueError("No session IDs provided or found for strat_name")

    # Load, score, summarize
    all_bets_df = load_bets_with_results(session_ids)
    scored_bets_df = score_bets(all_bets_df)
    per_bet_df, per_session_df = summarize_performance(scored_bets_df)

    if print_per_session:
        print("\n=== BACKTEST PERFORMANCE (per session) ===")
        print(per_session_df.to_string(index=False))

        # Overall aggregate
        total_staked = per_session_df["total_staked"].sum()
        total_fee = per_session_df["total_fee"].sum()
        total_net = per_session_df["total_net"].sum()
        wins = per_session_df["wins"].sum()
        losses = per_session_df["losses"].sum()
        pending = per_session_df["pending"].sum()
        overall_roi = (total_net / total_staked) if total_staked else float("nan")

        print("\n=== OVERALL BACKTEST PERFORMANCE ===")
        print(f"Total Staked: {total_staked:.2f}")
        print(f"Total Fee:    {total_fee:.2f}")
        print(f"Total Net:    {total_net:.2f}")
        print(f"Wins:         {wins}")
        print(f"Losses:       {losses}")
        print(f"Pending:      {pending}")
        print(f"Overall ROI:  {overall_roi:.4%}")

    return per_session_df
