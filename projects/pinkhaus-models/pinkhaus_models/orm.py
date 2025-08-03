"""SQLAlchemy ORM models for pinkhaus projects."""

from sqlalchemy import (
    Column, Integer, String, Text, Float, ForeignKey, 
    DateTime, Index, UniqueConstraint, LargeBinary, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()


class TranscriptionMetadataORM(Base):
    """ORM model for transcription metadata."""
    
    __tablename__ = "transcription_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(Text, nullable=False)
    file_hash = Column(Text, nullable=False)
    language = Column(Text, nullable=False)
    full_text = Column(Text, nullable=False)
    model_name = Column(Text)
    feed_url = Column(Text)
    feed_item_id = Column(Text)
    feed_item_title = Column(Text)
    feed_item_published = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    segments = relationship("TranscriptionSegmentORM", back_populates="transcription", 
                          cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('file_hash', 'model_name'),
        Index('idx_transcription_metadata_file_hash', 'file_hash'),
        Index('idx_transcription_metadata_feed_item', 'feed_url', 'feed_item_id',
              unique=True, postgresql_where=(feed_url.isnot(None) & feed_item_id.isnot(None))),
    )


class TranscriptionSegmentORM(Base):
    """ORM model for transcription segments."""
    
    __tablename__ = "transcription_segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transcription_id = Column(Integer, ForeignKey('transcription_metadata.id'), nullable=False)
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    
    # Relationships
    transcription = relationship("TranscriptionMetadataORM", back_populates="segments")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('transcription_id', 'segment_index'),
        Index('idx_transcription_segments_transcription_id', 'transcription_id'),
    )


class TextChunkORM(Base):
    """ORM model for text chunks with embeddings (for vector store)."""
    
    __tablename__ = "text_chunks"
    
    id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    metadata_json = Column('metadata', JSON, nullable=False)  # Use different attribute name
    embedding = Column(LargeBinary)  # Store numpy array as binary
    created_at = Column(DateTime, server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_created', 'created_at'),
    )