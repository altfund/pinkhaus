#!/usr/bin/env python3
"""
Utility to update existing transcriptions with speaker diarization data.

This script processes existing transcriptions in the database and adds:
- Speaker diarization
- Voice embeddings
- Speaker profiles
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from pinkhaus_models.database import TranscriptionDatabase
from .audio_processor import AudioProcessor
from .transcriber import TranscriptionResult


def update_transcription_with_speakers(
    db: TranscriptionDatabase,
    transcription_id: int,
    audio_path: str,
    processor: AudioProcessor,
    feed_url: Optional[str] = None
) -> bool:
    """
    Update a single transcription with speaker data.
    
    This updates the EXISTING transcription with speaker labels,
    rather than creating a new one.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get existing transcription from database
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get transcription metadata
            cursor.execute(
                "SELECT filename, file_hash, language, full_text, feed_url FROM transcription_metadata WHERE id = ?",
                (transcription_id,)
            )
            trans_data = cursor.fetchone()
            
            if not trans_data:
                print(f"Transcription {transcription_id} not found")
                return False
                
            filename, file_hash, language, full_text, existing_feed_url = trans_data
            
            # Use existing feed_url if not provided
            if not feed_url and existing_feed_url:
                feed_url = existing_feed_url
            elif not feed_url:
                feed_url = f"file://{audio_path}"
                
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
            
        print(f"Processing transcription {transcription_id}: {filename}")
        
        # First, perform diarization to get speaker labels
        from .speaker_diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer(auth_token=processor.hf_token)
        diarization = diarizer.diarize_file(audio_path, use_mock=False)
        
        print(f"  - Detected {diarization.num_speakers} speakers")
        
        # Get existing segments
        from .transcriber import TranscriptionResult, Segment
        result = TranscriptionResult()
        result.filename = filename
        result.file_hash = file_hash
        result.language = language
        result.full_text = full_text
        
        # Load segments from database
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT start_time, end_time, text FROM transcription_segments WHERE transcription_id = ? ORDER BY start_time",
                (transcription_id,)
            )
            
            for start_time, end_time, text in cursor.fetchall():
                # Convert from seconds to milliseconds for Segment
                seg = Segment(start_ms=int(start_time * 1000), end_ms=int(end_time * 1000), text=text)
                result.segments.append(seg)
        
        # Align diarization with existing segments
        segments_list = [
            {
                "start": seg.start_ms / 1000.0,
                "end": seg.end_ms / 1000.0,
                "text": seg.text
            }
            for seg in result.segments
        ]
        
        enhanced_segments = diarizer.align_with_transcription(diarization, segments_list)
        
        # Update segments with speaker info
        result.has_speaker_labels = True
        result.num_speakers = diarization.num_speakers
        
        for i, enhanced in enumerate(enhanced_segments):
            if i < len(result.segments):
                result.segments[i].speaker = enhanced.get("speaker", "UNKNOWN")
                result.segments[i].confidence = enhanced.get("confidence", 0.0)
        
        # Extract voice embeddings
        embeddings = processor.extractor.extract_embeddings_for_transcription(audio_path, result)
        
        # Update database with speaker information
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update transcription metadata
            cursor.execute(
                "UPDATE transcription_metadata SET num_speakers = ?, has_speaker_labels = ? WHERE id = ?",
                (diarization.num_speakers, True, transcription_id)
            )
            
            # First, we need to create speaker records in the speakers table
            speaker_id_map = {}
            for speaker_label in set(seg.speaker for seg in result.segments if hasattr(seg, 'speaker') and seg.speaker):
                cursor.execute(
                    "INSERT INTO speakers (transcription_id, speaker_label) VALUES (?, ?)",
                    (transcription_id, speaker_label)
                )
                speaker_id_map[speaker_label] = cursor.lastrowid
            
            # Update segments with speaker labels
            for i, seg in enumerate(result.segments):
                if hasattr(seg, 'speaker') and seg.speaker and seg.speaker in speaker_id_map:
                    speaker_id = speaker_id_map[seg.speaker]
                    cursor.execute(
                        "UPDATE transcription_segments SET speaker_id = ?, speaker_confidence = ? WHERE transcription_id = ? AND start_time = ? AND end_time = ?",
                        (speaker_id, getattr(seg, 'confidence', None), transcription_id, seg.start_ms / 1000.0, seg.end_ms / 1000.0)
                    )
            
            conn.commit()
        
        # Now handle speaker profiles and embeddings
        speaker_profiles = {}
        for speaker_label, (embedding, duration, segment_indices) in embeddings.items():
            # Try to match to existing profile
            matches = processor.matcher.find_best_match(embedding, feed_url=feed_url)
            
            if matches and matches[0][1] >= processor.matcher.threshold:
                # Found a match
                profile_id = matches[0][0]
                confidence = matches[0][1]
                print(f"  - {speaker_label} matched to existing profile (confidence: {confidence:.3f})")
            else:
                # Create new profile
                profile_id = db.create_speaker_profile(
                    display_name=f"Speaker {speaker_label.split('_')[1]}" if "SPEAKER_" in speaker_label else speaker_label,
                    feed_url=feed_url,
                    canonical_label=speaker_label
                )
                confidence = 1.0
                print(f"  - {speaker_label} created new profile (ID: {profile_id})")
            
            # Store embedding
            from .voice_embeddings import serialize_embedding
            db.add_speaker_embedding(
                profile_id,
                serialize_embedding(embedding),
                256,
                quality_score=confidence * 0.9
            )
            
            speaker_profiles[speaker_label] = (profile_id, confidence)
        
        # Link speaker occurrences
        for speaker_label, (profile_id, confidence) in speaker_profiles.items():
            db.link_speaker_occurrence(
                transcription_id=transcription_id,
                temporary_label=speaker_label,
                profile_id=profile_id,
                confidence=confidence,
                is_verified=False
            )
        
        print(f"  - Created/matched {len(speaker_profiles)} profiles")
        
        return True
        
    except Exception as e:
        print(f"Error processing transcription {transcription_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update existing transcriptions with speaker diarization"
    )
    parser.add_argument(
        "db_path",
        help="Path to the database containing transcriptions"
    )
    parser.add_argument(
        "--audio-dir",
        help="Directory containing audio files (searches recursively)"
    )
    parser.add_argument(
        "--audio-map",
        help="CSV file mapping transcription IDs to audio paths (id,path)"
    )
    parser.add_argument(
        "--transcription-id",
        type=int,
        help="Update only a specific transcription ID"
    )
    parser.add_argument(
        "--feed-url",
        help="Feed URL for speaker profile scoping"
    )
    parser.add_argument(
        "--embedding-method",
        choices=["speechbrain", "pyannote", "mock"],
        default="speechbrain",
        help="Voice embedding extraction method"
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required")
        print("Please set: export HF_TOKEN='hf_...'")
        print("And accept conditions at:")
        print("  - https://hf.co/pyannote/speaker-diarization-3.1")
        print("  - https://hf.co/pyannote/segmentation-3.0")
        sys.exit(1)
    
    # Initialize database
    db = TranscriptionDatabase(args.db_path)
    
    # Check if speaker tables exist
    try:
        db.get_speaker_profiles_for_feed("")
    except:
        print("Creating speaker identity tables...")
        db.create_speaker_identity_tables()
    
    # Initialize AudioProcessor
    processor = AudioProcessor(
        db=db,
        model_name=args.model,
        hf_token=hf_token,
        embedding_method=args.embedding_method,
        matcher_threshold=0.85
    )
    
    # Build audio file mapping
    audio_map = {}
    
    if args.audio_map:
        # Load from CSV (support stdin with "-")
        if args.audio_map == "-":
            import sys
            for line in sys.stdin:
                if line.strip():
                    trans_id, audio_path = line.strip().split(',', 1)
                    audio_map[int(trans_id)] = audio_path
        else:
            with open(args.audio_map, 'r') as f:
                for line in f:
                    if line.strip():
                        trans_id, audio_path = line.strip().split(',', 1)
                        audio_map[int(trans_id)] = audio_path
                    
    elif args.audio_dir:
        # Search directory for audio files
        audio_dir = Path(args.audio_dir)
        audio_files = {}
        
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']:
            for audio_file in audio_dir.rglob(ext):
                audio_files[audio_file.name] = str(audio_file)
        
        # Match with transcriptions
        if args.transcription_id:
            trans = db.get_transcription(args.transcription_id)
            if trans and trans['metadata']['filename'] in audio_files:
                audio_map[args.transcription_id] = audio_files[trans['metadata']['filename']]
        else:
            # Get all transcriptions without speaker data
            # This would need a new database method - for now just get recent ones
            print(f"Found {len(audio_files)} audio files in {args.audio_dir}")
            print("Manual mapping may be required. Use --audio-map option.")
            
    if args.dry_run:
        print("\nDRY RUN - No changes will be made")
        print(f"\nWould update {len(audio_map)} transcriptions:")
        for trans_id, audio_path in audio_map.items():
            print(f"  - Transcription {trans_id} -> {audio_path}")
        return
    
    # Process transcriptions
    print(f"\nUpdating {len(audio_map)} transcriptions...")
    
    success = 0
    failed = 0
    
    for trans_id, audio_path in audio_map.items():
        if update_transcription_with_speakers(
            db, trans_id, audio_path, processor, args.feed_url
        ):
            success += 1
        else:
            failed += 1
    
    print(f"\nComplete! Updated: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()