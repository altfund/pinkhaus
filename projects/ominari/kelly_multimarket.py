#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 04:11:34 2025

@author: ess
"""

import numpy as np
from scipy.optimize import minimize
import itertools
from collections import defaultdict


def infer_exclusivity_groups(bets):
    """
    Group mutually exclusive bets based on match_id and unified_market_type.

    Args:
        bets (list of dict): Each bet dict should include at least:
            - 'match_id': a unique identifier for the match or event
            - 'unified_market_type': a string like 'winner', 'total_goals', etc.

    Returns:
        dict: Mapping of exclusivity group labels to indices of bets
    """
    exclusivity_groups = defaultdict(list)

    for i, bet in enumerate(bets):
        match_id = bet.get("match_id")
        market_type = bet.get("unified_market_type")
        line = bet.get("line")

        if not match_id or not market_type or line is None:
            print(f"Warning: Missing fields in bet #{i}: {bet}")
            print(match_id)
            print(market_type)
            print(line)
            continue

        key = f"{match_id}::{market_type}::{line}"
        exclusivity_groups[key].append(i)

    return dict(exclusivity_groups)


def calculate_kelly_utility(
    stakes, odds, probs, bankroll, correlation_matrix=None, exclusivity_groups=None
):
    """
    stakes: array of bet sizes
    odds: array of odds
    probs: array of probabilities
    correlation_matrix: optional correlation matrix (NxN)
    exclusivity_groups: dict of group_name -> list of indices that are exclusive
    """
    end_bankrolls = []
    for i in range(len(stakes)):
        win_return = bankroll - np.sum(stakes) + stakes[i] * odds[i]
        end_bankrolls.append(probs[i] * np.log(win_return))

    total_utility = sum(end_bankrolls)

    # Penalize overbetting mutually exclusive groups
    if exclusivity_groups:
        for group, indices in exclusivity_groups.items():
            total_group_prob = sum([probs[i] for i in indices])
            if total_group_prob > 1.0:
                penalty = 10 * (total_group_prob - 1) ** 2
                total_utility -= penalty

    # Optional: Adjust based on correlation (simplified)
    if correlation_matrix is not None:
        for i, j in itertools.combinations(range(len(stakes)), 2):
            corr = correlation_matrix[i, j]
            stake_prod = stakes[i] * stakes[j]
            total_utility -= (
                0.01 * corr * stake_prod
            )  # penalize correlation overlap slightly

    return -total_utility  # because we're minimizing


def optimize_kelly_multimarket(
    bets, bankroll, exclusivity_groups=None, correlation_matrix=None
):
    """
    bets: list of dicts with keys: name, odds, prob
    bankroll: total bankroll available
    """
    odds = np.array([b["odds"] for b in bets])
    probs = np.array([b["prob"] for b in bets])
    names = [b["name"] for b in bets]

    x0 = np.zeros(len(bets)) + bankroll / (len(bets) * 10)
    bounds = [(0, bankroll * 0.1) for _ in bets]  # Cap to 10% per bet

    cons = {"type": "ineq", "fun": lambda x: bankroll - sum(x)}

    # Filter exclusive groups to best option
    if exclusivity_groups:
        filtered_bets = []
        for group, indices in exclusivity_groups.items():
            best_ev = -np.inf
            best_index = None
            for i in indices:
                ev = bets[i]["prob"] * (bets[i]["odds"] - 1) - (
                    1 - bets[i]["prob"]
                )  # basic EV
                if ev > best_ev:
                    best_ev = ev
                    best_index = i
            if best_index is not None:
                filtered_bets.append(bets[best_index])

        # Add back non-exclusive bets
        exclusive_indices = set(i for idxs in exclusivity_groups.values() for i in idxs)
        non_exclusive = [b for i, b in enumerate(bets) if i not in exclusive_indices]
        bets = filtered_bets + non_exclusive

    result = minimize(
        calculate_kelly_utility,
        x0=x0,
        args=(odds, probs, bankroll, correlation_matrix, exclusivity_groups),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    allocations = result.x
    output = []
    for i in range(len(bets)):
        if allocations[i] > 1e-4:
            output.append(
                {
                    "name": names[i],
                    "odds": odds[i],
                    "prob": probs[i],
                    "stake": round(allocations[i], 2),
                }
            )

    return output


import pandas as pd


def calculate_kelly_stakes_with_exclusivity(
    bets_df,
    bankroll=1000,
    correlation_matrix=None,
    risk_adjusted=True,
    max_stake_per_bet=None,
):
    """
    Calculate optimal stake allocations using Kelly Criterion with mutual exclusivity and correlation.

    Parameters:
    - bets_df: DataFrame with columns ['match_id', 'unified_market_type', 'bet_name', 'odds', 'probability']
    - bankroll: Total bankroll available
    - correlation_matrix: Optional correlation matrix of outcome dependencies
    - risk_adjusted: Penalize for variance if True
    - max_stake_per_bet: Optional cap on any single bet

    Returns:
    - DataFrame with original data and 'stake' column indicating optimal bet amount
    """

    def kelly_objective(fractions):
        """
        Kelly objective function to maximize expected log growth.
        Adjusts for correlation and mutual exclusivity.
        """
        expected_returns = bets_df["probability"].values * np.log1p(
            fractions * (bets_df["odds"].values - 1)
        )
        if risk_adjusted and correlation_matrix is not None:
            # Quadratic penalty term for correlated outcomes
            portfolio_variance = fractions.T @ correlation_matrix @ fractions
            return -np.sum(expected_returns) + 0.5 * portfolio_variance
        else:
            return -np.sum(expected_returns)

    # Dynamically determine mutual exclusivity groups based on match_id and unified_market_type
    exclusivity_groups = {}
    group_ids = bets_df.groupby(
        ["source_id", "unified_market_type", "normalized_line"]
    ).indices
    for group_idx, indices in enumerate(group_ids.values()):
        exclusivity_groups[group_idx] = list(indices)

    n = len(bets_df)
    x0 = np.zeros(n)  # Start with no bets
    bounds = [(0, 1) for _ in range(n)]  # No negative stakes

    if max_stake_per_bet:
        bounds = [(0, min(1, max_stake_per_bet / bankroll)) for _ in range(n)]

    # Constraint: Sum of stakes in any exclusivity group must be <= 1
    constraints = []
    for indices in exclusivity_groups.values():
        constraints.append(
            {"type": "ineq", "fun": lambda f, idx=indices: 1 - np.sum(f[idx])}
        )

    # Total investment must be <= 1 (i.e., total bankroll)
    constraints.append({"type": "ineq", "fun": lambda f: 1 - np.sum(f)})

    # If no correlation matrix provided, assume independence
    if correlation_matrix is None:
        correlation_matrix = np.eye(n)

    result = minimize(
        kelly_objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"disp": False},
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    bets_df = bets_df.copy()
    bets_df["stake_fraction"] = result.x
    bets_df["stake"] = bankroll * bets_df["stake_fraction"]

    return bets_df[
        [
            "source_id",
            "unified_market_type",
            "normalized_outcome",
            "normalized_line",
            "market_name",
            "league_name",
            "bookmaker",
            "odds",
            "probability",
            "stake",
            "stake_fraction",
        ]
    ]


# ===== Example usage =====
if __name__ == "__main__":
    # EXAMPLE TEST CASE
    example_bets = pd.DataFrame(
        {
            "match_id": ["game1", "game1", "game2", "game2"],
            "unified_market_type": ["winner", "winner", "total", "total"],
            "bet_name": ["Team A", "Team B", "Over 2.5", "Under 2.5"],
            "odds": [2.1, 1.9, 2.0, 2.0],
            "probability": [0.48, 0.52, 0.50, 0.50],
        }
    )

    # Example usage
    results = calculate_kelly_stakes_with_exclusivity(
        example_bets,
        bankroll=1000,
        correlation_matrix=np.eye(len(example_bets)),  # No correlation
        risk_adjusted=True,
    )

    results

    bankroll = 1000

    bets = [
        {
            "name": "Over 2.5 goals",
            "odds": 2.0,
            "prob": 0.5,
            "match_id": "match123",
            "unified_market_type": "total_goals",
            "line": 2.5,
        },
        {
            "name": "Under 2.5 goals",
            "odds": 2.0,
            "prob": 0.5,
            "match_id": "match123",
            "unified_market_type": "total_goals",
            "line": 2.5,
        },
        {
            "name": "Team A to win",
            "odds": 3.0,
            "prob": 0.3,
            "match_id": "match123",
            "unified_market_type": "winner",
            "line": 0.0,
        },
        {
            "name": "Team B to win",
            "odds": 2.5,
            "prob": 0.4,
            "match_id": "match123",
            "unified_market_type": "winner",
            "line": 0.0,
        },
        {
            "name": "Draw",
            "odds": 3.5,
            "prob": 0.3,
            "match_id": "match123",
            "unified_market_type": "winner",
            "line": 0.0,
        },
    ]

    exclusivity_groups = infer_exclusivity_groups(bets)
    print(exclusivity_groups)

    correlation_matrix = np.array(
        [
            [1.0, 0.5, -0.3, 0.2, -0.2],
            [0.5, 1.0, 0.2, 0.1, -0.1],
            [-0.3, 0.2, 1.0, -0.2, 0.3],
            [0.2, 0.1, -0.2, 1.0, -0.9],
            [-0.2, -0.1, 0.3, -0.9, 1.0],
        ]
    )

    result = optimize_kelly_multimarket(
        bets, bankroll, exclusivity_groups, correlation_matrix
    )
    print("Recommended Bets:")
    for r in result:
        print(
            f" - {r['name']} @ {r['odds']} | Prob: {r['prob']} | Stake: ${r['stake']}"
        )
