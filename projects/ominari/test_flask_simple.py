#!/usr/bin/env python3
try:
    import flask
    print("Flask is available!")
except ImportError:
    print("Flask not available")