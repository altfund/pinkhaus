#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 01:14:39 2025

@author: ess
"""

import pandas as pd
import numpy as np
from performance import score_bets
from evaluate_open_markets import (
    fetch_open_markets_for_as_of,
    fetch_market_outcomes_for_window
)
from kelly_multimarket import calculate_kelly_stakes_with_exclusivity

from signals import *



def compute_strategy_correlation_and_weights(
    perf: pd.DataFrame,
    metric_col: str = "roi",
    group_keys: tuple = ("strategy_name", "source_id"),
) -> tuple:
    """
    Pivot performance DataFrame to compute inter-strategy correlation and inverse-variance weights.

    Returns:
      - corr: DataFrame of strategy×strategy correlations
      - weights: Series of inferred weight per strategy
    """
    pivot = (
        perf
        .pivot_table(index=group_keys[1], columns=group_keys[0], values=metric_col)
        .fillna(0.0)
    )
    corr = pivot.corr()
    variances = np.diag(corr)
    inv_var = 1.0 / variances
    weights = pd.Series(inv_var / inv_var.sum(), index=corr.index)
    return corr, weights


def fetch_markets_with_outcomes(
    as_of: pd.Timestamp,
    until: pd.Timestamp,
    require_outcome: bool = False
) -> pd.DataFrame:
    """
    Retrieve market snapshot at `as_of` and their outcomes by `until` in one combined DataFrame.

    If `require_outcome` is True, only markets with an outcome in the window are returned.

    Output columns include all from the market snapshot plus:
      - result_multiplier: decimal odds or zero if lost
      - outcome_time: timestamp of the outcome
    """
    # 1) get markets open at snapshot
    markets = fetch_open_markets_for_as_of(as_of)
    if markets.empty:
        raise RuntimeError(f"No markets at snapshot {as_of}")

    # 2) fetch outcomes for these markets
    source_ids = markets['source_id'].tolist()
    outcomes = fetch_market_outcomes_for_window(source_ids, as_of, until)

    # 3) merge snapshot and outcomes
    how = 'inner' if require_outcome else 'left'
    df = markets.merge(
        outcomes,
        on='source_id',
        how=how,
        suffixes=('', '_outcome')
    )
    # ensure result_multiplier present
    if 'result_multiplier' not in df.columns:
        df['result_multiplier'] = df.get('decimal_odds', 0.0)
    return df


def vectorized_backtest(
    strategies: list,
    as_of: pd.Timestamp,
    until: pd.Timestamp,
    require_outcome: bool = False
) -> pd.DataFrame:
    """
    Fully vectorized backtest over a single window [as_of, until].

    Steps per strategy:
      - combine market snapshot and outcomes
      - compute combined signal from providers
      - calculate Kelly stakes
      - collate performance

    Parameters:
      - strategies: list of dict specs
      - as_of: pd.Timestamp snapshot
      - until: pd.Timestamp end of window
      - require_outcome: only include markets with outcomes if True
    """
    # validate snapshot availability

    if not strategies:
        raise ValueError("At least one strategy spec required.")
    if as_of >= until:
        raise ValueError("'as_of' must be before 'until'")

    # fetch combined market + outcome data once
    market_df = fetch_markets_with_outcomes(as_of, until, require_outcome=require_outcome)
    if market_df.empty:
        raise RuntimeError(f"No merged market/outcome data for window {as_of} – {until}")

    all_perf = []
    for spec in strategies:
        name = spec.get("name")
        providers = spec.get("providers")
        weights = spec.get("weights")
        bankroll = spec.get("bankroll", 1000)
        corr_matrix = spec.get("correlation_matrix", None)
        risk_adjusted = spec.get("risk_adjusted", True)
        max_bet = spec.get("max_stake_per_bet", None)

        if not name or providers is None or weights is None:
            raise ValueError(f"Invalid strategy spec: {spec}")

        # build combined probability signal
        signals = [prov.get_signals(market_df, as_of) for prov in providers]
        if isinstance(weights, dict):
            weights = [
                weights.get(
                    getattr(p, 'name', type(p).__name__),
                    spec['weights'].get(name, 0.0)
                ) for p in providers
            ]
        prob = sum(w * s for w, s in zip(weights, signals))

        # prepare bets_df for Kelly
        bets_df = pd.DataFrame({
            'match_id': market_df['match_id'],
            'unified_market_type': market_df['market_type'],
            'bet_name': market_df['bet_name'],
            'odds': market_df['decimal_odds'],
            'probability': prob
        })
        stakes = calculate_kelly_stakes_with_exclusivity(
            bets_df,
            bankroll=bankroll,
            correlation_matrix=corr_matrix,
            risk_adjusted=risk_adjusted,
            max_stake_per_bet=max_bet
        )

        # assemble performance
        df = pd.DataFrame({
            'strategy_name': name,
            'source_id': market_df['source_id'],
            'total_staked': stakes,
            'result_multiplier': market_df['result_multiplier']
        })
        df['net'] = df['total_staked'] * df['result_multiplier'] - df['total_staked']
        df['roi'] = df.apply(lambda r: r.net / r.total_staked if r.total_staked else 0.0, axis=1)
        all_perf.append(df)

    perf = pd.concat(all_perf, ignore_index=True)

    # compute correlations & weights
    corr, wts = compute_strategy_correlation_and_weights(perf[['strategy_name','source_id','roi']])
    print("Strategy Correlation Matrix:")
    print(corr)
    print("Inferred Strategy Weights (inverse-variance):")
    print(wts)

    return perf


if __name__ == "__main__":

    strategies = [
        {
            "name": provider.name,
            "providers": provider,
            "weights": 1.0,
            "bankroll": 1000,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None
        } for provider in SIGNAL_PROVIDERS
    ]
    as_of = pd.Timestamp("2025-01-01 00:00:00")
    until = pd.Timestamp("2025-06-30 23:59:59")

    df = vectorized_backtest(
        strategies=strategies,
        as_of=as_of,
        until=until
    )
    print(df.head())
