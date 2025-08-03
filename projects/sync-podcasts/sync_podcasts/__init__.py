"""Sync Podcasts - Orchestrates podcast transcription and indexing pipeline."""

__version__ = "0.1.0"

from .sync import PodcastSyncer, SyncConfig

__all__ = ["PodcastSyncer", "SyncConfig"]
