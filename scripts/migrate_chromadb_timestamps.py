#!/usr/bin/env python3
"""
Migrate existing ChromaDB data to include Unix timestamps for date filtering.

This script adds published_timestamp fields to existing chunks in the vector store.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.grant.grant.vector_store import PodcastVectorStore


def migrate_timestamps(vector_store_path: str = "./grant_chroma_db"):
    """Add Unix timestamps to existing chunks in the vector store."""
    
    print(f"Migrating timestamps in vector store: {vector_store_path}")
    
    # Initialize vector store
    vector_store = PodcastVectorStore(persist_directory=vector_store_path)
    
    # Get all chunks
    print("Fetching all chunks...")
    all_data = vector_store.collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    
    if not all_data["ids"]:
        print("No chunks found in vector store.")
        return
    
    total_chunks = len(all_data["ids"])
    print(f"Found {total_chunks} chunks to process")
    
    # Process chunks that have published dates but no timestamps
    updated_count = 0
    for i, (chunk_id, metadata) in enumerate(zip(all_data["ids"], all_data["metadatas"])):
        if i % 100 == 0:
            print(f"Processing chunk {i+1}/{total_chunks}...")
        
        # Check if chunk has published date but no timestamp
        if metadata.get("published") and not metadata.get("published_timestamp"):
            try:
                # Parse the ISO date and convert to Unix timestamp
                published_dt = datetime.fromisoformat(
                    metadata["published"].replace('Z', '+00:00')
                )
                published_timestamp = int(published_dt.timestamp())
                
                # Update the metadata
                metadata["published_timestamp"] = published_timestamp
                
                # Update in ChromaDB
                vector_store.collection.update(
                    ids=[chunk_id],
                    metadatas=[metadata]
                )
                
                updated_count += 1
                
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
    
    print(f"\nMigration complete!")
    print(f"Updated {updated_count} chunks with timestamps")
    print(f"Total chunks: {total_chunks}")


def verify_migration(vector_store_path: str = "./grant_chroma_db"):
    """Verify that the migration was successful."""
    
    print(f"\nVerifying migration in: {vector_store_path}")
    
    vector_store = PodcastVectorStore(persist_directory=vector_store_path)
    
    # Sample some chunks to check
    sample = vector_store.collection.get(
        limit=10,
        include=["metadatas"]
    )
    
    has_timestamp = 0
    has_published = 0
    
    for metadata in sample["metadatas"]:
        if metadata.get("published"):
            has_published += 1
            if metadata.get("published_timestamp"):
                has_timestamp += 1
    
    print(f"Sampled 10 chunks:")
    print(f"  - {has_published} have published dates")
    print(f"  - {has_timestamp} have timestamp fields")
    
    # Test date filtering
    if has_timestamp > 0:
        print("\nTesting date filtering...")
        # Get a sample timestamp
        test_timestamp = next(
            m["published_timestamp"] 
            for m in sample["metadatas"] 
            if m.get("published_timestamp")
        )
        
        # Test filtering
        filtered = vector_store.collection.get(
            where={"published_timestamp": {"$gte": test_timestamp}},
            limit=5,
            include=["metadatas"]
        )
        
        print(f"Found {len(filtered['ids'])} chunks with timestamp >= {test_timestamp}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB to include Unix timestamps"
    )
    parser.add_argument(
        "--vector-store",
        default="./grant_chroma_db",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the migration, don't perform it"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_migration(args.vector_store)
    else:
        migrate_timestamps(args.vector_store)
        verify_migration(args.vector_store)


if __name__ == "__main__":
    main()