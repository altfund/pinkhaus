#!/usr/bin/env python3
"""
High-level audio processing with automatic speaker identification.

This module provides a unified interface for processing audio files with:
- Transcription
- Speaker diarization  
- Voice embedding extraction
- Speaker profile creation/matching
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from .transcriber import AudioTranscriber
from .speaker_diarizer import SpeakerDiarizer
from .voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding
from .speaker_matcher import SpeakerMatcher
from pinkhaus_models.database import TranscriptionDatabase

logger = logging.getLogger(__name__)


class AudioProcessor:
    """High-level audio processor that handles all voice-related tasks automatically."""
    
    def __init__(
        self,
        db: TranscriptionDatabase,
        model_name: str = "base",
        hf_token: Optional[str] = None,
        embedding_method: str = "speechbrain",
        matcher_threshold: float = 0.85
    ):
        """
        Initialize the audio processor.
        
        Args:
            db: Database instance for storing results
            model_name: Whisper model size (tiny, base, small, medium, large)
            hf_token: Hugging Face token for pyannote models
            embedding_method: Method for voice embeddings (speechbrain, pyannote, mock)
            matcher_threshold: Threshold for speaker matching (0.0-1.0)
        """
        self.db = db
        self.transcriber = AudioTranscriber(model_name=model_name)
        self.diarizer = SpeakerDiarizer(auth_token=hf_token) if hf_token else None
        self.extractor = VoiceEmbeddingExtractor(method=embedding_method)
        self.matcher = SpeakerMatcher(
            db, 
            threshold=matcher_threshold, 
            embedding_method=embedding_method
        )
        self.hf_token = hf_token
        
    def process_audio_file(
        self,
        audio_path: str,
        feed_url: str,
        enable_diarization: bool = True,
        create_new_profiles: bool = True,
        profile_prefix: str = "Speaker",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process an audio file with full speaker identification pipeline.
        
        This method:
        1. Transcribes the audio
        2. Performs speaker diarization (if enabled)
        3. Extracts voice embeddings for each speaker
        4. Matches speakers to existing profiles or creates new ones
        5. Saves everything to the database
        
        Args:
            audio_path: Path to the audio file
            feed_url: Feed URL for scoping speaker profiles
            enable_diarization: Whether to perform speaker diarization
            create_new_profiles: Whether to create new profiles for unmatched speakers
            profile_prefix: Prefix for new speaker profile names
            verbose: Whether to print progress messages
            
        Returns:
            Dict containing:
                - transcription_id: Database ID of saved transcription
                - transcription: TranscriptionResult object
                - num_speakers: Number of speakers detected
                - speaker_profiles: Dict mapping speaker labels to profile IDs
                - embeddings: Dict of extracted embeddings
                - matches: Dict of speaker matching results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if verbose:
            print(f"Processing: {audio_path.name}")
            
        # Step 1: Transcribe with diarization
        if enable_diarization and self.diarizer:
            if verbose:
                print("  1. Transcribing with speaker diarization...")
            result = self.transcriber.transcribe_file(
                str(audio_path),
                enable_diarization=True,
                hf_token=self.hf_token,
                verbose=False
            )
        else:
            if verbose:
                print("  1. Transcribing audio...")
            result = self.transcriber.transcribe_file(
                str(audio_path),
                verbose=False
            )
            
        if verbose:
            print(f"     ✓ Transcribed {len(result.segments)} segments")
            if result.has_speaker_labels:
                print(f"     ✓ Detected {result.num_speakers} speakers")
                
        # Step 2: Extract voice embeddings if we have speaker labels
        embeddings = {}
        if result.has_speaker_labels:
            if verbose:
                print("  2. Extracting voice embeddings...")
            embeddings = self.extractor.extract_embeddings_for_transcription(
                str(audio_path),
                result
            )
            if verbose:
                print(f"     ✓ Extracted embeddings for {len(embeddings)} speakers")
        else:
            if verbose:
                print("  2. No speaker labels - skipping embedding extraction")
                
        # Step 3: Match speakers and create/update profiles
        speaker_profiles = {}
        matches = {}
        
        if embeddings:
            if verbose:
                print("  3. Matching speakers to profiles...")
                
            for speaker_label, (embedding, duration, segment_indices) in embeddings.items():
                # Try to match to existing profile
                match_results = self.matcher.find_best_match(embedding, feed_url=feed_url)
                
                if match_results and match_results[0][1] >= self.matcher.threshold:
                    # Found a match
                    profile_id = match_results[0][0]
                    confidence = match_results[0][1]
                    profile_data = match_results[0][2]
                    
                    if verbose:
                        print(f"     {speaker_label} → {profile_data['display_name']} (confidence: {confidence:.3f})")
                    
                    speaker_profiles[speaker_label] = profile_id
                    matches[speaker_label] = {
                        'profile_id': profile_id,
                        'confidence': confidence,
                        'is_new': False
                    }
                    
                    # Add new embedding to profile
                    self.db.add_speaker_embedding(
                        profile_id,
                        serialize_embedding(embedding),
                        256,
                        quality_score=confidence
                    )
                    
                elif create_new_profiles:
                    # Create new profile
                    speaker_num = speaker_label.split('_')[1] if 'SPEAKER_' in speaker_label else '?'
                    display_name = f"{profile_prefix} {speaker_num}"
                    
                    profile_id = self.db.create_speaker_profile(
                        display_name=display_name,
                        feed_url=feed_url,
                        canonical_label=speaker_label
                    )
                    
                    if verbose:
                        print(f"     {speaker_label} → NEW: {display_name} (ID: {profile_id})")
                    
                    speaker_profiles[speaker_label] = profile_id
                    matches[speaker_label] = {
                        'profile_id': profile_id,
                        'confidence': 1.0,
                        'is_new': True
                    }
                    
                    # Store embedding
                    self.db.add_speaker_embedding(
                        profile_id,
                        serialize_embedding(embedding),
                        256,
                        quality_score=0.90
                    )
        else:
            if verbose:
                print("  3. No embeddings - skipping speaker matching")
                
        # Step 4: Save transcription to database
        if verbose:
            print("  4. Saving to database...")
            
        trans_id = self.db.save_transcription(result, feed_url=feed_url)
        
        # Link speaker occurrences
        for speaker_label, profile_id in speaker_profiles.items():
            match_info = matches.get(speaker_label, {})
            self.db.link_speaker_occurrence(
                transcription_id=trans_id,
                temporary_label=speaker_label,
                profile_id=profile_id,
                confidence=match_info.get('confidence', 0.90),
                is_verified=not match_info.get('is_new', True)
            )
            
        if verbose:
            print(f"     ✓ Saved transcription (ID: {trans_id})")
            print(f"     ✓ Linked {len(speaker_profiles)} speaker occurrences")
            
        return {
            'transcription_id': trans_id,
            'transcription': result,
            'num_speakers': result.num_speakers if result.has_speaker_labels else 0,
            'speaker_profiles': speaker_profiles,
            'embeddings': embeddings,
            'matches': matches
        }
        
    def process_episode_batch(
        self,
        audio_files: List[str],
        feed_url: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files from the same feed.
        
        This is useful for processing multiple episodes and building up
        speaker profiles over time.
        
        Args:
            audio_files: List of audio file paths
            feed_url: Feed URL for all episodes
            **kwargs: Additional arguments passed to process_audio_file
            
        Returns:
            List of results from process_audio_file for each episode
        """
        results = []
        
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\n=== Episode {i}/{len(audio_files)} ===")
            try:
                result = self.process_audio_file(audio_path, feed_url, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append({
                    'error': str(e),
                    'audio_path': audio_path
                })
                
        return results
        
    def get_speaker_summary(self, feed_url: str) -> Dict[str, Any]:
        """
        Get a summary of all speakers for a feed.
        
        Args:
            feed_url: Feed URL to get speakers for
            
        Returns:
            Dict with speaker statistics and information
        """
        profiles = self.db.get_speaker_profiles_for_feed(feed_url)
        
        summary = {
            'feed_url': feed_url,
            'total_speakers': len(profiles),
            'speakers': []
        }
        
        for profile in profiles:
            speaker_info = {
                'id': profile['id'],
                'name': profile['display_name'],
                'label': profile['canonical_label'],
                'appearances': profile['total_appearances'],
                'duration_seconds': profile['total_duration_seconds'],
                'embeddings_count': len(self.db.get_speaker_embeddings(profile['id']))
            }
            
            # Get sample statements
            statements = self.db.get_speaker_statements(profile['id'])
            if statements:
                speaker_info['sample_statements'] = [
                    stmt['text'][:100] + '...' if len(stmt['text']) > 100 else stmt['text']
                    for stmt in statements[:3]
                ]
                
            summary['speakers'].append(speaker_info)
            
        # Sort by total speaking time
        summary['speakers'].sort(key=lambda x: x['duration_seconds'], reverse=True)
        
        return summary