#!/usr/bin/env python3
"""Test script to reproduce the feed processing error"""

import logging
import tempfile
from beige_book import TranscriptionService, TranscriptionDatabase
from beige_book.models import TranscriptionRequest, InputConfig, ProcessingConfig, OutputConfig, DatabaseConfig, FeedOptions

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create a minimal test
try:
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"Using temp database: {db_path}")
    
    # Create database
    db = TranscriptionDatabase(db_path)
    db.create_tables()
    
    # Create service
    service = TranscriptionService()
    
    # Create request
    request = TranscriptionRequest(
        input=InputConfig(
            type="feed",
            source="../../resources/fc/feeds.toml"
        ),
        processing=ProcessingConfig(
            model="tiny",  # Use tiny for faster testing
            feed_options=FeedOptions(limit=1)  # Process only 1 item
        ),
        output=OutputConfig(
            format="sqlite",
            database=DatabaseConfig(
                db_path=db_path
            )
        )
    )
    
    # Process
    print("Starting feed processing...")
    response = service.process_request(request)
    
    print(f"Response: {response}")
    print(f"Summary: {response.summary}")
    
    if response.errors:
        print(f"Errors: {response.errors}")
        for error in response.errors:
            print(f"  - {error.source}: {error.error_type} - {error.message}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(f"Type: {type(e).__name__}")
    print(f"Traceback:\n{traceback.format_exc()}")