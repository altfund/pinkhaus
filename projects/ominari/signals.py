#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 01:15:51 2025

@author: ess
"""

import pandas as pd
import time
import grpc

class SignalProvider:
    """Base class: must return a pd.Series of probabilities (0–1)."""

    name: str

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class ImpliedRawSignal(SignalProvider):
    name = "implied_raw"

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # assumes df["implied_raw"] is in percent
        return df["implied_raw"].astype(float).div(100.0)


class ExternalGrpcSignal(SignalProvider):
    name = "external"

    def __init__(self, stub, timeout: float = 2.0):
        self.stub = stub
        self.timeout = timeout

    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        import grpc
        from external_pb2 import SignalBatchRequest

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
            # teams
            # odds?
            # make this a meta query or is that what this is?
            r.normalized_outcome = row["normalized_outcome"]
            r.as_of_time = row["time"].isoformat()
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
            probs = [0.0] * len(df)

        # 5) Return and log
        series = pd.Series(probs, index=df.index)
        print(f"[CLIENT]  → returning series head:\n{series.head()}", flush=True)
        return series
    
# ─── 2. CONFIGURE SIGNALS & WIRING INTO prepare_kelly_input ─────────────────
# Initialize external gRPC stub once at module load
# in evaluate_open_markets.py, near top:
def _init_external_stub():
    # for local testing, use your dummy server:
    channel = grpc.insecure_channel("localhost:50051")
    return __import__("external_pb2_grpc").SignalServiceStub(channel)


_external_stub = _init_external_stub()


SIGNAL_PROVIDERS = [
    ImpliedRawSignal(),
    ExternalGrpcSignal(_external_stub),
]

SIGNAL_WEIGHTS = {
    "implied_kelly": 1.0,    # base implied probability
    "random":    1.0,    # weight for external model
}
