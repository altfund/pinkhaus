#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 01:14:39 2025

@author: ess
"""

import pandas as pd
import numpy as np
from evaluate_open_markets import (
    fetch_open_markets_for_as_of,
    fetch_market_outcomes_for_window,
    apply_overtime_fees,
)
from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
from backtest import compute_backtest_as_of_list


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
    pivot = perf.pivot_table(
        index=group_keys[1], columns=group_keys[0], values=metric_col
    ).fillna(0.0)
    corr = pivot.corr()
    variances = np.diag(corr)
    inv_var = 1.0 / variances
    weights = pd.Series(inv_var / inv_var.sum(), index=corr.index)
    return corr, weights


def fetch_markets_with_outcomes(
    as_of: pd.Timestamp, until: pd.Timestamp, require_outcome: bool = False
) -> pd.DataFrame:
    """
    Retrieve market snapshot at `as_of` and their outcomes by `until` in one combined DataFrame.

    If `require_outcome` is True, only markets with an outcome in the window are returned.

    Output columns include all from the market snapshot plus:
      - result_multiplier: decimal odds or zero if lost
      - outcome_time: timestamp of the outcome
      - adjusted_odds: fee-adjusted odds for Kelly calculation
      - total_fee_pct: total fee percentage
    """
    # 1) get markets open at snapshot
    markets = fetch_open_markets_for_as_of(as_of)
    if markets.empty:
        raise RuntimeError(f"No markets at snapshot {as_of}")

    # 2) Apply fees to get adjusted odds
    markets = apply_overtime_fees(markets)
    
    # 3) fetch outcomes for these markets
    source_ids = markets["source_id"].tolist()
    outcomes = fetch_market_outcomes_for_window(source_ids, as_of, until)

    # 4) merge snapshot and outcomes
    how = "inner" if require_outcome else "left"
    df = markets.merge(outcomes, on="source_id", how=how, suffixes=("", "_outcome"))
    # ensure result_multiplier present
    if "result_multiplier" not in df.columns:
        df["result_multiplier"] = df.get("decimal_odds", 0.0)
    return df


def vectorized_backtest(
    strategies: list,
    as_of: pd.Timestamp,
    until: pd.Timestamp,
    require_outcome: bool = False,
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
    market_df = fetch_markets_with_outcomes(
        as_of, until, require_outcome=require_outcome
    )
    if market_df.empty:
        raise RuntimeError(
            f"No merged market/outcome data for window {as_of} – {until}"
        )

    # Create match_id if not present
    if "match_id" not in market_df.columns:
        market_df["match_id"] = (
            market_df["home_team"].astype(str)
            + "_vs_"
            + market_df["away_team"].astype(str)
        )

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
        signals = []
        for prov in providers:
            try:
                signal = prov.get_probs(market_df)
                # Handle NaN values
                signal = signal.fillna(0.0)
                signals.append(signal)
            except Exception as e:
                print(f"Error getting signal from {prov.name}: {e}")
                # Create zero signal as fallback
                signals.append(pd.Series(0.0, index=market_df.index))

        # Handle weights - can be a list or dict
        if isinstance(weights, dict):
            weight_list = [weights.get(prov.name, 0.0) for prov in providers]
        elif isinstance(weights, list):
            weight_list = weights
        else:
            # Single weight value, apply to all providers
            weight_list = [weights] * len(providers)

        # Normalize weights
        total_weight = sum(weight_list)
        if total_weight > 0:
            weight_list = [w / total_weight for w in weight_list]
        else:
            weight_list = [1.0 / len(providers)] * len(providers)

        prob = sum(w * s for w, s in zip(weight_list, signals))
        # Ensure probabilities are in valid range
        prob = prob.clip(0.0, 1.0)

        # prepare bets_df for Kelly - use adjusted_odds if available
        odds_column = "adjusted_odds" if "adjusted_odds" in market_df.columns else "decimal_odds"
        bets_df = pd.DataFrame(
            {
                "match_id": market_df["match_id"],
                "unified_market_type": market_df["market_type"],
                "bet_name": market_df["bet_name"],
                "odds": market_df[odds_column],
                "probability": prob,
            }
        )
        stakes = calculate_kelly_stakes_with_exclusivity(
            bets_df,
            bankroll=bankroll,
            correlation_matrix=corr_matrix,
            risk_adjusted=risk_adjusted,
            max_stake_per_bet=max_bet,
        )

        # assemble performance with fee information
        df = pd.DataFrame(
            {
                "strategy_name": name,
                "source_id": market_df["source_id"].values,
                "total_staked": stakes,
                "result_multiplier": market_df["result_multiplier"].fillna(0.0).values,
                "fee_pct": market_df["total_fee_pct"].fillna(0.0).values if "total_fee_pct" in market_df.columns else 0.0,
            }
        )

        # Only include bets with non-zero stakes
        df = df[df["total_staked"] > 0].copy()

        if not df.empty:
            # Calculate execution stake (stake + fees)
            df["fee_amount"] = df["total_staked"] * df["fee_pct"]
            df["execution_stake"] = df["total_staked"] + df["fee_amount"]
            
            # Calculate P&L accounting for fees
            # Win: gross_payout - execution_stake
            # Loss: -execution_stake
            gross_payout = df["total_staked"] * df["result_multiplier"]
            df["net"] = np.where(
                df["result_multiplier"] > 0,
                gross_payout - df["execution_stake"],  # Win case
                -df["execution_stake"]  # Loss case
            )
            df["roi"] = df.apply(
                lambda r: r.net / r.execution_stake if r.execution_stake > 0 else 0.0, axis=1
            )
            all_perf.append(df)

    if not all_perf:
        print("No bets placed in this window")
        return pd.DataFrame()

    perf = pd.concat(all_perf, ignore_index=True)

    # compute correlations & weights only if we have enough data
    if len(perf) > 1 and perf["strategy_name"].nunique() > 1:
        try:
            corr, wts = compute_strategy_correlation_and_weights(
                perf[["strategy_name", "source_id", "roi"]]
            )
            print("Strategy Correlation Matrix:")
            print(corr)
            print("Inferred Strategy Weights (inverse-variance):")
            print(wts)
        except Exception as e:
            print(f"Could not compute correlations: {e}")

    return perf


def run_vectorized_backtest_with_chunks(
    strategies: list,
    min_break_minutes: float = 60.0,
    avg_game_duration_minutes: float = 120.0,
    save_results: bool = True,
):
    """
    Run vectorized backtest over multiple time chunks.
    """
    from datetime import datetime, timezone

    # Get all time chunks
    as_of_pairs = compute_backtest_as_of_list(
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
    )

    if not as_of_pairs:
        print("No backtest chunks found; exiting.")
        return

    # Record runtime for unique backtest ID
    runtime = datetime.now(timezone.utc)
    start_ts, _ = as_of_pairs[0]
    end_ts, _ = as_of_pairs[-1]

    backtest_id = (
        f"vectorized_"
        f"{runtime.strftime('%Y%m%d_%H%M%S')}_"
        f"start={start_ts.strftime('%Y%m%d_%H%M')}_"
        f"end={end_ts.strftime('%Y%m%d_%H%M')}"
    )

    print(f"Running vectorized backtest: {backtest_id}")
    print(f"Processing {len(as_of_pairs)} time windows")

    all_results = []

    # Process each time window
    for i, (as_of, chunk_end) in enumerate(as_of_pairs):
        print(
            f"\n[{i + 1}/{len(as_of_pairs)}] Processing window: {as_of} to {chunk_end}"
        )

        try:
            perf_df = vectorized_backtest(
                strategies=strategies,
                as_of=as_of,
                until=chunk_end,
                require_outcome=True,
            )

            if not perf_df.empty:
                # Add metadata
                perf_df["as_of"] = as_of
                perf_df["chunk_end"] = chunk_end
                perf_df["backtest_id"] = backtest_id

                all_results.append(perf_df)
            else:
                print(f"No bets placed in window {as_of} to {chunk_end}")

            # Print summary for this window if we have data
            if not perf_df.empty:
                summary = perf_df.groupby("strategy_name").agg(
                    {"total_staked": "sum", "net": "sum", "roi": "mean"}
                )
                print(f"Window summary:\n{summary}")

        except Exception as e:
            print(f"Error processing window {as_of}: {e}")
            continue

    if not all_results:
        print("No successful results to process")
        return None

    # Combine all results
    full_results = pd.concat(all_results, ignore_index=True)

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    agg_dict = {
        "total_staked": "sum", 
        "net": "sum", 
        "roi": "mean"
    }
    
    # Add fee aggregations if fees are present
    if "fee_amount" in full_results.columns:
        agg_dict["fee_amount"] = "sum"
    if "execution_stake" in full_results.columns:
        agg_dict["execution_stake"] = "sum"
        
    final_summary = full_results.groupby("strategy_name").agg(agg_dict)
    
    # Calculate total return based on execution stake if available
    if "execution_stake" in final_summary.columns:
        final_summary["total_return"] = final_summary["net"] / final_summary["execution_stake"]
    else:
        final_summary["total_return"] = final_summary["net"] / final_summary["total_staked"]
        
    print(final_summary)

    # Save results if requested
    if save_results:
        output_dir = f"backtests/{backtest_id}"
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save full results
        full_results.to_csv(f"{output_dir}/full_results.csv", index=False)

        # Save summary
        final_summary.to_csv(f"{output_dir}/summary.csv")

        print(f"\nResults saved to: {output_dir}/")

    return full_results


if __name__ == "__main__":
    # Single strategy using all signal providers with their weights
    combined_strategy = {
        "name": "combined_signals",
        "providers": SIGNAL_PROVIDERS,
        "weights": SIGNAL_WEIGHTS,
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None,
    }

    # Individual strategies for each provider
    individual_strategies = [
        {
            "name": provider.name,
            "providers": [provider],
            "weights": [1.0],
            "bankroll": 1000,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None,
        }
        for provider in SIGNAL_PROVIDERS
    ]

    # Use combined strategy + individual strategies
    strategies = [combined_strategy] + individual_strategies

    # Run the chunked backtest
    results = run_vectorized_backtest_with_chunks(
        strategies=strategies,
        min_break_minutes=60.0,
        avg_game_duration_minutes=120.0,
        save_results=True,
    )
