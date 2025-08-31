"""
Speaker matching service for identifying speakers across recordings.

This module matches temporary speaker labels from diarization to persistent
speaker profiles using voice embeddings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from pinkhaus_models.database import TranscriptionDatabase
from .voice_embeddings import (
    VoiceEmbeddingExtractor,
    cosine_similarity,
    serialize_embedding,
    deserialize_embedding,
)


class SpeakerMatcher:
    """Match temporary speakers to persistent profiles using voice embeddings."""

    def __init__(
        self,
        db: TranscriptionDatabase,
        threshold: float = 0.85,
        embedding_method: str = "speechbrain",
        device: Optional[str] = None,
    ):
        """
        Initialize the speaker matcher.

        Args:
            db: Database instance
            threshold: Minimum cosine similarity for matching (0-1)
            embedding_method: Method for embedding extraction
            device: Device for embedding extraction
        """
        self.db = db
        self.threshold = threshold
        self.extractor = VoiceEmbeddingExtractor(method=embedding_method, device=device)

    def find_best_match(
        self, embedding: np.ndarray, feed_url: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Find best matching speaker profiles for an embedding.

        Args:
            embedding: Voice embedding to match
            feed_url: Optional feed URL to scope the search
            top_k: Number of top matches to return

        Returns:
            List of (profile_id, similarity, profile_info) tuples
        """
        # Get candidate profiles
        if feed_url:
            profiles = self.db.get_speaker_profiles_for_feed(feed_url)
        else:
            # Get all profiles (would need to add this method)
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM speaker_profiles ORDER BY total_appearances DESC LIMIT 100"
                )
                profiles = [dict(row) for row in cursor.fetchall()]

        if not profiles:
            return []

        # Calculate similarities
        matches = []
        for profile in profiles:
            profile_id = profile["id"]

            # Get embeddings for this profile
            embeddings_data = self.db.get_speaker_embeddings(profile_id)
            if not embeddings_data:
                continue

            # Calculate similarity with each embedding
            similarities = []
            for emb_data in embeddings_data:
                stored_embedding = deserialize_embedding(
                    emb_data["embedding"], emb_data["embedding_dimension"]
                )
                sim = cosine_similarity(embedding, stored_embedding)
                similarities.append(sim)

            # Use max similarity
            if similarities:
                max_sim = max(similarities)
                matches.append((profile_id, max_sim, profile))

        # Sort by similarity and return top K
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def match_speaker(
        self,
        embedding: np.ndarray,
        feed_url: Optional[str] = None,
        create_if_not_found: bool = True,
        speaker_hint: Optional[str] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Match a speaker embedding to a profile.

        Args:
            embedding: Voice embedding
            feed_url: Optional feed URL for scoping
            create_if_not_found: Whether to create new profile if no match
            speaker_hint: Optional hint for speaker name

        Returns:
            Tuple of (profile_id, confidence)
        """
        # Find best matches
        matches = self.find_best_match(embedding, feed_url)

        if matches and matches[0][1] >= self.threshold:
            # Found a good match
            return matches[0][0], matches[0][1]

        if create_if_not_found:
            # Create new profile
            display_name = (
                speaker_hint
                or f"Unknown Speaker {datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            profile_id = self.db.create_speaker_profile(
                display_name=display_name, feed_url=feed_url
            )
            return profile_id, 1.0

        return None, 0.0

    def identify_speakers_in_transcription(
        self,
        transcription_id: int,
        audio_path: str,
        embeddings: Optional[Dict[str, Tuple[np.ndarray, float, List[int]]]] = None,
        feed_url: Optional[str] = None,
    ) -> Dict[str, Tuple[int, float]]:
        """
        Identify all speakers in a transcription and link to profiles.

        Args:
            transcription_id: ID of the transcription
            audio_path: Path to audio file
            embeddings: Pre-extracted embeddings or None to extract
            feed_url: Feed URL for scoping

        Returns:
            Dict mapping temporary_label to (profile_id, confidence)
        """
        # Get transcription data
        trans_data = self.db.get_transcription(transcription_id)
        if not trans_data or not trans_data["metadata"]["has_speaker_labels"]:
            return {}

        # Extract embeddings if not provided
        if embeddings is None:
            # Would need to reconstruct TranscriptionResult from database
            # For now, assume embeddings are provided
            raise ValueError("Embeddings must be provided")

        # Match each speaker
        speaker_mappings = {}

        for temp_label, (embedding, duration, indices) in embeddings.items():
            # Try to match speaker
            profile_id, confidence = self.match_speaker(
                embedding,
                feed_url=feed_url,
                speaker_hint=f"Speaker from {trans_data['metadata']['filename']}",
            )

            if profile_id:
                # Store embedding
                self.db.add_speaker_embedding(
                    profile_id=profile_id,
                    embedding=serialize_embedding(embedding),
                    embedding_dimension=len(embedding),
                    source_transcription_id=transcription_id,
                    source_segment_indices=indices,
                    duration_seconds=duration,
                    quality_score=min(1.0, duration / 10.0),
                    extraction_method=self.extractor.method,
                )

                # Link occurrence
                self.db.link_speaker_occurrence(
                    transcription_id=transcription_id,
                    temporary_label=temp_label,
                    profile_id=profile_id,
                    confidence=confidence,
                )

                speaker_mappings[temp_label] = (profile_id, confidence)

        # Update segment profile_ids
        self._update_segment_profiles(transcription_id, speaker_mappings)

        return speaker_mappings

    def _update_segment_profiles(
        self, transcription_id: int, speaker_mappings: Dict[str, Tuple[int, float]]
    ):
        """Update segment profile_ids based on speaker mappings."""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            # Get temporary speaker assignments
            cursor.execute(
                """
                SELECT seg.id, seg.speaker_id, spk.speaker_label
                FROM transcription_segments seg
                JOIN speakers spk ON seg.speaker_id = spk.id
                WHERE seg.transcription_id = ?
            """,
                (transcription_id,),
            )

            segments = cursor.fetchall()

            # Update each segment with profile_id
            for seg in segments:
                temp_label = seg["speaker_label"]
                if temp_label in speaker_mappings:
                    profile_id, _ = speaker_mappings[temp_label]
                    cursor.execute(
                        """
                        UPDATE transcription_segments
                        SET profile_id = ?
                        WHERE id = ?
                    """,
                        (profile_id, seg["id"]),
                    )

    def merge_speaker_profiles(
        self, profile_id_keep: int, profile_id_merge: int
    ) -> bool:
        """
        Merge two speaker profiles (for fixing misidentifications).

        Args:
            profile_id_keep: Profile to keep
            profile_id_merge: Profile to merge into keep

        Returns:
            Success status
        """
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Update embeddings
                cursor.execute(
                    """
                    UPDATE speaker_embeddings
                    SET profile_id = ?
                    WHERE profile_id = ?
                """,
                    (profile_id_keep, profile_id_merge),
                )

                # Update occurrences
                cursor.execute(
                    """
                    UPDATE speaker_occurrences
                    SET profile_id = ?
                    WHERE profile_id = ?
                """,
                    (profile_id_keep, profile_id_merge),
                )

                # Update segments
                cursor.execute(
                    """
                    UPDATE transcription_segments
                    SET profile_id = ?
                    WHERE profile_id = ?
                """,
                    (profile_id_keep, profile_id_merge),
                )

                # Update profile statistics
                cursor.execute(
                    """
                    UPDATE speaker_profiles
                    SET total_appearances = (
                        SELECT COUNT(DISTINCT transcription_id)
                        FROM speaker_occurrences
                        WHERE profile_id = ?
                    ),
                    total_duration_seconds = (
                        SELECT COALESCE(SUM(duration), 0)
                        FROM transcription_segments
                        WHERE profile_id = ?
                    ),
                    updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (profile_id_keep, profile_id_keep, profile_id_keep),
                )

                # Delete merged profile
                cursor.execute(
                    """
                    DELETE FROM speaker_profiles
                    WHERE id = ?
                """,
                    (profile_id_merge,),
                )

                conn.commit()
                return True

            except Exception as e:
                print(f"Error merging profiles: {e}")
                conn.rollback()
                return False

    def verify_speaker_occurrence(
        self, transcription_id: int, temporary_label: str, profile_id: int
    ) -> bool:
        """
        Mark a speaker occurrence as human-verified.

        Args:
            transcription_id: Transcription ID
            temporary_label: Temporary speaker label
            profile_id: Verified profile ID

        Returns:
            Success status
        """
        try:
            self.db.link_speaker_occurrence(
                transcription_id=transcription_id,
                temporary_label=temporary_label,
                profile_id=profile_id,
                confidence=1.0,
                is_verified=True,
            )
            return True
        except Exception as e:
            print(f"Error verifying speaker: {e}")
            return False
