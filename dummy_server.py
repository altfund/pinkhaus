#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 01:06:25 2025

@author: ess
"""

# dummy_server.py

import time
import grpc
from concurrent import futures

# import the generated classes
from external_pb2 import SignalBatchResponse
from external_pb2_grpc import SignalServiceServicer, add_SignalServiceServicer_to_server

class DummySignalServer(SignalServiceServicer):
    """A test server that returns 0.5 for every probability."""
    def GetProbabilities(self, request, context):
        print(f"[SERVER] Got {len(request.requests)} requests")
        n = len(request.requests)
        # build a response with 0.5 for each incoming request
        return SignalBatchResponse(probabilities=[0.5] * n)

def serve(host: str = '[::]:50051', max_workers: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_SignalServiceServicer_to_server(DummySignalServer(), server)
    server.add_insecure_port(host)
    server.start()
    print(f"▶ DummySignalServer listening on {host}")
    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        server.stop(0)
        print("\n🛑 Server stopped")

if __name__ == '__main__':
    serve()
