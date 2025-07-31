"""
Database import/export utilities for transcriptions.

Provides functionality to export database transcriptions to JSON/TOML files
and import them back, preserving all metadata including feed information.
"""

import json
import toml
from pathlib import Path
from typing import List, Dict, Any, Optional
from pinkhaus_models import TranscriptionDatabase, TranscriptionResult


class DatabaseIO:
    """Utility class for database import/export operations."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db = TranscriptionDatabase(db_path)
    
    def export_all_to_json(self, output_path: str) -> int:
        """
        Export all transcriptions to a JSON file.
        
        Args:
            output_path: Path to the output JSON file
            
        Returns:
            Number of transcriptions exported
        """
        all_transcriptions = self.db.get_all_transcriptions()
        export_data = []
        
        for metadata in all_transcriptions:
            # Get full transcription data with segments
            full_data = self.db.get_transcription(metadata.id)
            if not full_data:
                continue
            
            # Export transcription result
            result = self.db.export_to_dict(metadata.id)
            if not result:
                continue
            
            # Build export structure with all metadata
            export_item = {
                "transcription": result.to_dict(),
                "metadata": {
                    "model_name": metadata.model_name,
                    "feed_url": metadata.feed_url,
                    "feed_item_id": metadata.feed_item_id,
                    "feed_item_title": metadata.feed_item_title,
                    "feed_item_published": metadata.feed_item_published.isoformat() if metadata.feed_item_published else None,
                    "created_at": metadata.created_at.isoformat() if metadata.created_at else None
                }
            }
            export_data.append(export_item)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                {"transcriptions": export_data, "version": "1.0"},
                f,
                indent=2,
                ensure_ascii=False
            )
        
        return len(export_data)
    
    def export_all_to_toml(self, output_path: str) -> int:
        """
        Export all transcriptions to a TOML file.
        
        Args:
            output_path: Path to the output TOML file
            
        Returns:
            Number of transcriptions exported
        """
        all_transcriptions = self.db.get_all_transcriptions()
        export_data = {"version": "1.0", "transcriptions": []}
        
        for metadata in all_transcriptions:
            # Get full transcription data
            result = self.db.export_to_dict(metadata.id)
            if not result:
                continue
            
            # Convert to dict for TOML
            transcription_dict = result.to_dict()
            
            # Add metadata
            transcription_dict["metadata"] = {
                "model_name": metadata.model_name or "",
                "feed_url": metadata.feed_url or "",
                "feed_item_id": metadata.feed_item_id or "",
                "feed_item_title": metadata.feed_item_title or "",
                "feed_item_published": metadata.feed_item_published.isoformat() if metadata.feed_item_published else "",
                "created_at": metadata.created_at.isoformat() if metadata.created_at else ""
            }
            
            export_data["transcriptions"].append(transcription_dict)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            toml.dump(export_data, f)
        
        return len(export_data["transcriptions"])
    
    def import_from_json(self, input_path: str, skip_duplicates: bool = True) -> Dict[str, int]:
        """
        Import transcriptions from a JSON file.
        
        Args:
            input_path: Path to the input JSON file
            skip_duplicates: Whether to skip transcriptions that already exist
            
        Returns:
            Dictionary with 'imported' and 'skipped' counts
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transcriptions = data.get("transcriptions", [])
        imported = 0
        skipped = 0
        
        for item in transcriptions:
            # Extract transcription and metadata
            transcription_data = item["transcription"]
            metadata = item.get("metadata", {})
            
            # Check for duplicates if requested
            if skip_duplicates:
                file_hash = transcription_data.get("file_hash")
                model_name = metadata.get("model_name", "unknown")
                
                # Check if already exists
                with self.db._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT id FROM transcription_metadata WHERE file_hash = ? AND model_name = ?",
                        (file_hash, model_name)
                    )
                    existing = cursor.fetchone()
                
                if existing:
                    skipped += 1
                    continue
            
            # Create TranscriptionResult from dict
            result = TranscriptionResult.from_dict(transcription_data)
            
            # Save to database
            self.db.save_transcription(
                result,
                model_name=metadata.get("model_name", "unknown"),
                feed_url=metadata.get("feed_url"),
                feed_item_id=metadata.get("feed_item_id"),
                feed_item_title=metadata.get("feed_item_title"),
                feed_item_published=metadata.get("feed_item_published")
            )
            imported += 1
        
        return {"imported": imported, "skipped": skipped}
    
    def import_from_toml(self, input_path: str, skip_duplicates: bool = True) -> Dict[str, int]:
        """
        Import transcriptions from a TOML file.
        
        Args:
            input_path: Path to the input TOML file
            skip_duplicates: Whether to skip transcriptions that already exist
            
        Returns:
            Dictionary with 'imported' and 'skipped' counts
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        
        transcriptions = data.get("transcriptions", [])
        imported = 0
        skipped = 0
        
        for item in transcriptions:
            # Extract metadata
            metadata = item.pop("metadata", {})
            
            # Check for duplicates if requested
            if skip_duplicates:
                file_hash = item.get("file_hash")
                model_name = metadata.get("model_name", "unknown")
                
                # Check if already exists
                with self.db._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT id FROM transcription_metadata WHERE file_hash = ? AND model_name = ?",
                        (file_hash, model_name)
                    )
                    existing = cursor.fetchone()
                
                if existing:
                    skipped += 1
                    continue
            
            # Create TranscriptionResult from dict
            result = TranscriptionResult.from_dict(item)
            
            # Save to database
            self.db.save_transcription(
                result,
                model_name=metadata.get("model_name", "unknown"),
                feed_url=metadata.get("feed_url") or None,
                feed_item_id=metadata.get("feed_item_id") or None,
                feed_item_title=metadata.get("feed_item_title") or None,
                feed_item_published=metadata.get("feed_item_published") or None
            )
            imported += 1
        
        return {"imported": imported, "skipped": skipped}
    
    def export_transcription_to_json(self, transcription_id: int, output_path: str) -> bool:
        """
        Export a single transcription to JSON file.
        
        Args:
            transcription_id: ID of the transcription to export
            output_path: Path to the output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        # Get full data
        db_data = self.db.get_transcription(transcription_id)
        if not db_data:
            return False
        
        result = self.db.export_to_dict(transcription_id)
        if not result:
            return False
        
        metadata = db_data["metadata"]
        
        export_data = {
            "transcription": result.to_dict(),
            "metadata": {
                "model_name": metadata["model_name"],
                "feed_url": metadata["feed_url"],
                "feed_item_id": metadata["feed_item_id"],
                "feed_item_title": metadata["feed_item_title"],
                "feed_item_published": metadata["feed_item_published"],
                "created_at": metadata["created_at"]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def export_transcription_to_toml(self, transcription_id: int, output_path: str) -> bool:
        """
        Export a single transcription to TOML file.
        
        Args:
            transcription_id: ID of the transcription to export
            output_path: Path to the output TOML file
            
        Returns:
            True if successful, False otherwise
        """
        # Get full data
        db_data = self.db.get_transcription(transcription_id)
        if not db_data:
            return False
        
        result = self.db.export_to_dict(transcription_id)
        if not result:
            return False
        
        metadata = db_data["metadata"]
        
        # Convert to dict and add metadata
        export_data = result.to_dict()
        export_data["metadata"] = {
            "model_name": metadata["model_name"] or "",
            "feed_url": metadata["feed_url"] or "",
            "feed_item_id": metadata["feed_item_id"] or "",
            "feed_item_title": metadata["feed_item_title"] or "",
            "feed_item_published": metadata["feed_item_published"] or "",
            "created_at": metadata["created_at"] or ""
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            toml.dump(export_data, f)
        
        return True