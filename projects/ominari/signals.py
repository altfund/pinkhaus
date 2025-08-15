#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 01:15:51 2025

@author: ess
"""

import pandas as pd
import time
import grpc
from external_pb2 import SignalBatchRequest
from external_pb2_grpc import SignalServiceStub

class SignalProvider:
    """Base class: must return a pd.Series of probabilities (0–1)."""

    name: str

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ImpliedRawSignal(SignalProvider):
    name = "implied_probability"

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # assumes df["implied_raw"] is in percent
        return df["implied_raw"].astype(float).div(100.0)


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
            r.as_of_time = row["time"].isoformat()
            r.query = "?"
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

        # 3) Build & log the batch request
        req = SignalBatchRequest()
        for idx, row in df.iterrows():
            r = req.requests.add()
            r.source_id = row["source_id"]
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time = row["time"].isoformat()
            r.query = compose_query(row)
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
def _init_external_stub(host: str = "localhost:50051") -> SignalServiceStub:
    """
    Returns a gRPC stub pointing at your external SignalService.
    """
    channel = grpc.insecure_channel(host)
    return SignalServiceStub(channel)


_external_stub = _init_external_stub()


SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    ExternalGrpcSignal(_external_stub),
    GrantSignal(_external_stub)
]

SIGNAL_WEIGHTS = {
    "implied_probability":1.0,    # base implied probability
    "coin_flip":       1.0,    # weight for external model
    "grant":        1.0,    # 
}
