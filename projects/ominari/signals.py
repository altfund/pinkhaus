#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 01:15:51 2025

@author: ess
"""

import pandas as pd
import time
import grpc
from datetime import datetime, timezone
from pinkhaus_models.proto.ominari.external_pb2 import SignalBatchRequest
from pinkhaus_models.proto.ominari.external_pb2_grpc import SignalServiceStub


class SignalProvider:
    """Base class: must return a pd.Series of probabilities (0–1)."""

    name: str

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ImpliedRawSignal(SignalProvider):
    name = "implied_probability"

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # Check if implied_raw column exists
        if "implied_raw" not in df.columns:
            print("[ImpliedRawSignal] WARNING: implied_raw column not found, falling back to odds calculation")
            # Calculate from odds if available
            if "odds" in df.columns:
                base_probs = 1.0 / df["odds"].replace(0, float('inf'))
            else:
                print("[ImpliedRawSignal] ERROR: Neither implied_raw nor odds column found")
                return pd.Series([0.5] * len(df), index=df.index)
        else:
            # assumes df["implied_raw"] is in percent
            base_probs = df["implied_raw"].astype(float).div(100.0)
        
        # Handle NaN, inf, and invalid values
        base_probs = base_probs.replace([float('inf'), -float('inf')], pd.NA)
        base_probs = base_probs.fillna(0.5)  # Default to 50% for missing values
        
        # Apply small biases based on known market inefficiencies
        adjusted_probs = base_probs.copy()
        
        # Calculate market overround/vig for each match group if possible
        if all(col in df.columns for col in ["source_id", "unified_market_type", "normalized_line"]):
            # Group by match and market type to calculate overround
            grouped = df.groupby(["source_id", "unified_market_type", "normalized_line"])
            
            for group_key, group_df in grouped:
                # Calculate total probability for this market
                group_indices = group_df.index
                group_base_probs = base_probs.loc[group_indices]
                total_prob = group_base_probs.sum()
                
                # Market overround indicates confidence - higher overround = less confident market
                # Typical overround is 5-10%, anything above 15% is high
                if total_prob > 1.0:
                    overround = total_prob - 1.0
                    
                    # Adjust probabilities based on overround
                    # Higher overround = scale down probabilities more aggressively
                    if overround > 0.15:  # High overround (>15%)
                        # Scale down probabilities by extra 2%
                        adjusted_probs.loc[group_indices] = base_probs.loc[group_indices] * 0.98
                    elif overround > 0.10:  # Medium-high overround (10-15%)
                        # Scale down probabilities by 1%
                        adjusted_probs.loc[group_indices] = base_probs.loc[group_indices] * 0.99
                    elif overround < 0.05:  # Low overround (<5%)
                        # Market is very competitive, trust probabilities more
                        # Scale up slightly by 0.5%
                        adjusted_probs.loc[group_indices] = base_probs.loc[group_indices] * 1.005
        
        # Get odds if available (use 'odds' column instead of 'decimal_odds')
        if "odds" in df.columns:
            odds = df["odds"]
            
            # Favorite-longshot bias: public overvalues favorites, undervalues longshots
            # Apply -1% adjustment to heavy favorites (odds < 2.0)
            favorite_mask = (odds > 1.0) & (odds < 2.0) & odds.notna()
            adjusted_probs.loc[favorite_mask] = adjusted_probs.loc[favorite_mask] - 0.01
            
            # Apply +1% adjustment to longshots (odds > 4.0)
            longshot_mask = (odds > 4.0) & odds.notna()
            adjusted_probs.loc[longshot_mask] = adjusted_probs.loc[longshot_mask] + 0.01
            
            # Draw bias: public slightly undervalues draws in soccer
            if "normalized_outcome" in df.columns:
                draw_mask = df["normalized_outcome"].str.contains("Draw|draw|option_3", na=False)
                adjusted_probs.loc[draw_mask] = adjusted_probs.loc[draw_mask] + 0.005  # +0.5% for draws
        
        # Ensure probabilities stay in valid range [0.01, 0.99]
        adjusted_probs = adjusted_probs.clip(0.01, 0.99)
        
        return adjusted_probs


class ExternalGrpcSignal(SignalProvider):
    name = "coin_flip"

    def __init__(self, stub, timeout: float = 2.0):
        self.stub = stub
        self.timeout = timeout

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # 1) Always log entry & DataFrame size
        print(f"[CLIENT] ExternalGrpcSignal.get_probs: {len(df)} rows", flush=True)

        # 2) If empty, bail early (but log it)
        if df.empty:
            print("[CLIENT]  → empty df → returning empty Series", flush=True)
            return pd.Series([], index=df.index)

        # 3) Build & log the batch request
        req = SignalBatchRequest()
        for idx, row in df.iterrows():
            r = req.requests.add()
            r.source_id = row["source_id"]
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time = row["time"].isoformat() if row.get("time") else datetime.now(timezone.utc).isoformat()
            r.query = "?"
            r.model = ""
        print(f"[CLIENT]  → sending {len(req.requests)} RPC requests", flush=True)

        # 4) Actually call the RPC, but don’t hide exceptions
        try:
            t0 = time.time()
            resp = self.stub.GetProbabilities(req, timeout=self.timeout)
            took = time.time() - t0
            print(
                f"[CLIENT]  → RPC returned in {took:.3f}s, {len(resp.probabilities)} probs",
                flush=True,
            )
            probs = list(resp.probabilities)
        except grpc.RpcError as e:
            # log the error before fallback
            print(f"[CLIENT]  ! RPC error: {e.code()} {e.details()}", flush=True)
            probs = [-1.0] * len(df)

        # 5) Return and log
        series = pd.Series(probs, index=df.index)
        print(f"[CLIENT]  → returning series head:\n{series.head()}", flush=True)
        return series


class GrantSignal(SignalProvider):
    name = "grant"

    def __init__(self, stub, timeout: float = 2.0):
        self.stub = stub
        self.timeout = timeout

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # 1) Always log entry & DataFrame size
        print(f"[CLIENT] GrantSignal.get_probs: {len(df)} rows", flush=True)

        # 2) If empty, bail early (but log it)
        if df.empty:
            print("[CLIENT]  → empty df → returning empty Series", flush=True)
            return pd.Series([], index=df.index)

        def compose_query(row):
            query = f"""
                You are trying to provide likelihoods between 0 and 1 of win/loss/draw outcomes for soccer matches.
                What is the probability of {row["normalized_outcome"]} 
                in the match {row["home_team"]} (home) vs {row["away_team"]} (away) 
                in the {row["league_name"]} league 
                on {row["maturity_date"]}?
            """
            return query

        # ± uv run grant list
        # Available models:
        #  - gpt-oss:20b
        #  - mistral-nemo:12b
        #  - nomic-embed-text:latest
        #  - deepseek-r1:8b
        #  - llama3.3:latest
        #  - llama3.2:latest

        # 3) Build & log the batch request
        req = SignalBatchRequest()
        for idx, row in df.iterrows():
            r = req.requests.add()
            r.source_id = row["source_id"]
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time = row["time"].isoformat() if row.get("time") else datetime.now(timezone.utc).isoformat()
            r.query = compose_query(row)
            r.model = "llama3.2:latest"
        print(f"[CLIENT]  → sending {len(req.requests)} RPC requests", flush=True)

        # 4) Actually call the RPC, but don’t hide exceptions
        try:
            t0 = time.time()
            resp = self.stub.GetProbabilities(req, timeout=self.timeout)
            took = time.time() - t0
            print(
                f"[CLIENT]  → RPC returned in {took:.3f}s, {len(resp.probabilities)} probs",
                flush=True,
            )
            probs = list(resp.probabilities)
        except grpc.RpcError as e:
            # log the error before fallback
            print(f"[CLIENT]  ! RPC error: {e.code()} {e.details()}", flush=True)
            probs = [-1.0] * len(df)

        # 5) Return and log
        series = pd.Series(probs, index=df.index)
        print(f"[CLIENT]  → returning series head:\n{series.head()}", flush=True)
        return series


# ─── 2. CONFIGURE SIGNALS & WIRING INTO prepare_kelly_input ─────────────────
# Initialize external gRPC stub once at module load
# in evaluate_open_markets.py, near top:
def _init_stub(host: str = "localhost:50050") -> SignalServiceStub:
    """
    Returns a gRPC stub pointing at your external SignalService.
    """
    channel = grpc.insecure_channel(host)
    return SignalServiceStub(channel)


_external_stub = _init_stub(host="localhost:50051")
_internal_stub = _init_stub()


def get_signal_providers():
    """Get list of available signal providers."""
    return SIGNAL_PROVIDERS

SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    # ExternalGrpcSignal(_internal_stub),  # Disabled - gRPC not running
    # GrantSignal(_external_stub),          # Disabled - gRPC not running
]

# Import blockchain signal providers
try:
    from blockchain_signal_provider import BlockchainEnhancedSignal, BlockchainMarketScout
    # Add blockchain signals to providers list
    SIGNAL_PROVIDERS.extend([
        BlockchainEnhancedSignal(),
        BlockchainMarketScout(),
    ])
    print("✅ Loaded blockchain signal providers")
except ImportError as e:
    print(f"⚠️  Could not load blockchain signal providers: {e}")

SIGNAL_WEIGHTS = {
    "implied_probability": 1.0,  # base implied probability
    "coin_flip": 1.0,  # weight for external model
    "grant": 1.0,  # Grant's LLM model
    "blockchain_enhanced_signal": 1.5,  # Enhanced blockchain signal (higher weight)
    "blockchain_market_scout": 1.2,  # Market scout signal
}
