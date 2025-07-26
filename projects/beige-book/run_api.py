#!/usr/bin/env python3
"""
Run the Beige Book Transcription API server.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import uvicorn
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Run the Beige Book Transcription API server"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Starting Beige Book Transcription API server...")
    print("Documentation will be available at:")
    print(f"  - Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"  - ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"  - OpenAPI JSON: http://{args.host}:{args.port}/openapi.json")

    # Run the server
    uvicorn.run(
        "beige_book.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
