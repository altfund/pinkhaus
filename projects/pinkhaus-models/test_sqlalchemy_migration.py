#!/usr/bin/env python3
"""Test script to verify SQLAlchemy migration works correctly."""

import tempfile
import os
from pinkhaus_models import TranscriptionDatabase, VectorStore, TranscriptionResult, Segment
import numpy as np


def test_transcription_database():
    """Test TranscriptionDatabase with SQLAlchemy backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = TranscriptionDatabase(db_path=db_path)
        
        # Create test data
        segments = [
            Segment(start=0.0, end=5.0, text="Hello world"),
            Segment(start=5.0, end=10.0, text="This is a test"),
        ]
        
        result = TranscriptionResult(
            filename="test.mp3",
            file_hash="abc123",
            language="en",
            segments=segments,
            full_text="Hello world. This is a test.",
        )
        
        # Test save
        transcription_id = db.save_transcription(
            result=result,
            model_name="whisper-1",
            feed_url="https://example.com/feed",
            feed_item_id="item123",
            feed_item_title="Test Episode",
            feed_item_published="2024-01-01T00:00:00",
        )
        print(f"✓ Saved transcription with ID: {transcription_id}")
        
        # Test retrieval
        metadata = db.get_transcription_metadata(transcription_id)
        assert metadata is not None
        assert metadata.filename == "test.mp3"
        assert metadata.file_hash == "abc123"
        print(f"✓ Retrieved metadata: {metadata.filename}")
        
        # Test segments
        segments = db.get_segments_for_transcription(transcription_id)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        print(f"✓ Retrieved {len(segments)} segments")
        
        # Test search
        results = db.search_transcriptions("test")
        assert len(results) > 0
        print(f"✓ Search found {len(results)} results")
        
        # Test feed item exists
        exists = db.check_feed_item_exists("https://example.com/feed", "item123")
        assert exists
        print("✓ Feed item check works")
        
        print("\n✅ TranscriptionDatabase tests passed!")


def test_vector_store():
    """Test VectorStore with SQLAlchemy backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "vectors.db")
        store = VectorStore(db_path=db_path)
        
        # Create test chunks
        chunks = []
        for i in range(3):
            chunk = {
                "id": f"chunk_{i}",
                "text": f"This is test chunk number {i}",
                "metadata": {"index": i, "source": "test"},
                "embedding": np.random.rand(384).astype(np.float32),
            }
            chunks.append(chunk)
        
        # Test add chunks
        store.add_chunks(chunks)
        print(f"✓ Added {len(chunks)} chunks")
        
        # Test retrieval
        retrieved = store.get_all_chunks()
        assert len(retrieved) == 3
        assert retrieved[0]["text"] == "This is test chunk number 0"
        print(f"✓ Retrieved {len(retrieved)} chunks")
        
        # Test similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = store.search_similar(query_embedding, top_k=2)
        assert len(results) == 2
        print(f"✓ Similarity search returned {len(results)} results")
        
        # Test chunk exists
        exists = store.chunk_exists("chunk_1")
        assert exists
        print("✓ Chunk existence check works")
        
        print("\n✅ VectorStore tests passed!")


def test_database_backends():
    """Test that we can use different database backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with SQLite URL format
        db_url = f"sqlite:///{tmpdir}/test_url.db"
        db = TranscriptionDatabase(database_url=db_url)
        db.create_tables()
        print("✓ SQLite URL format works")
        
        # Test backward compatibility with path
        db_path = os.path.join(tmpdir, "test_path.db")
        db2 = TranscriptionDatabase(db_path=db_path)
        db2.create_tables()
        print("✓ Legacy path format works")
        
        print("\n✅ Database backend tests passed!")


if __name__ == "__main__":
    print("Testing SQLAlchemy migration...\n")
    test_transcription_database()
    test_vector_store()
    test_database_backends()
    print("\n🎉 All tests passed! SQLAlchemy migration is working correctly.")