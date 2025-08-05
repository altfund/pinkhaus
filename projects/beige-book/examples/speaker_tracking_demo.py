#!/usr/bin/env python3
"""
Complete demo of speaker tracking across podcast episodes.

This example shows how to:
1. Transcribe multiple episodes with speaker diarization
2. Identify recurring speakers across episodes
3. Query speaker information over time
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinkhaus_models.database import TranscriptionDatabase
from beige_book.transcriber import AudioTranscriber
from beige_book.speaker_matcher import SpeakerMatcher
from beige_book.voice_embeddings import VoiceEmbeddingExtractor, serialize_embedding
from beige_book.speaker_diarizer import SpeakerDiarizer


def main():
    """Demonstrate speaker tracking across episodes."""
    
    # Require HF token for real diarization
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for real speaker diarization.\n"
            "Please set: export HF_TOKEN='hf_...'\n"
            "And accept conditions at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    
    # Create a temporary database for the demo
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print(f"\n=== SPEAKER TRACKING DEMO ===")
    print(f"Database: {db_path}")
    print()
    
    # Initialize database
    db = TranscriptionDatabase(db_path)
    db.create_tables()
    db.create_speaker_identity_tables()
    
    # Setup components
    feed_url = "https://example-podcast.com/feed.rss"
    transcriber = AudioTranscriber(model_name="tiny")
    
    # Path to test audio (using harvard.wav for demo)
    audio_path = Path(__file__).parent.parent.parent.parent / "resources" / "audio" / "harvard.wav"
    
    if not audio_path.exists():
        print(f"Error: Test audio not found at {audio_path}")
        return
    
    print("=== EPISODE 1: Initial Transcription ===\n")
    
    # Transcribe first "episode" with real diarization
    print("Transcribing with speaker diarization...")
    
    # Use the diarizer directly for more control
    diarizer = SpeakerDiarizer(auth_token=hf_token)
    diarization = diarizer.diarize_file(str(audio_path), use_mock=False)
    
    print(f"Identified {diarization.num_speakers} speakers in the audio")
    
    # Transcribe the audio
    result1 = transcriber.transcribe_file(str(audio_path), verbose=False)
    
    # Align diarization with transcription
    segments_list = [
        {
            "start": seg.start_ms / 1000.0,
            "end": seg.end_ms / 1000.0,
            "text": seg.text
        }
        for seg in result1.segments
    ]
    
    enhanced_segments = diarizer.align_with_transcription(diarization, segments_list)
    
    # Update transcription with speaker info
    result1.has_speaker_labels = True
    result1.num_speakers = diarization.num_speakers
    
    for i, seg_data in enumerate(enhanced_segments):
        if i < len(result1.segments):
            result1.segments[i].speaker = seg_data.get("speaker", "UNKNOWN")
            result1.segments[i].confidence = seg_data.get("confidence", 0.0)
    
    # Set metadata
    result1.filename = "episode_001_pilot.mp3"
    result1.file_hash = "ep1_hash_" + str(hash(result1.full_text))[:8]
    
    # Extract voice embeddings
    print("\nExtracting voice embeddings for each speaker...")
    extractor = VoiceEmbeddingExtractor(method="mock")
    embeddings1 = extractor.extract_embeddings_for_transcription(str(audio_path), result1)
    
    print(f"Extracted embeddings for {len(embeddings1)} speakers")
    
    # Create initial speaker profiles
    print("\nCreating speaker profiles...")
    
    # Manually create profiles for first episode
    # In production, this would be automatic
    speaker_profiles = {}
    
    # Assume SPEAKER_0 is the host
    host_id = db.create_speaker_profile(
        "Podcast Host",
        feed_url=feed_url,
        canonical_label="HOST"
    )
    
    if "SPEAKER_0" in embeddings1:
        embedding, duration, indices = embeddings1["SPEAKER_0"]
        db.add_speaker_embedding(
            host_id,
            serialize_embedding(embedding),
            256,
            quality_score=0.95
        )
        speaker_profiles["SPEAKER_0"] = host_id
        print(f"  Created HOST profile (ID: {host_id})")
    
    # Other speakers are guests
    guest_num = 1
    for speaker_label, (embedding, duration, indices) in embeddings1.items():
        if speaker_label != "SPEAKER_0":
            guest_id = db.create_speaker_profile(
                f"Guest {guest_num}",
                feed_url=feed_url,
                canonical_label="GUEST"
            )
            db.add_speaker_embedding(
                guest_id,
                serialize_embedding(embedding),
                256,
                quality_score=0.90
            )
            speaker_profiles[speaker_label] = guest_id
            print(f"  Created GUEST profile (ID: {guest_id})")
            guest_num += 1
    
    # Save transcription
    trans_id1 = db.save_transcription(result1, feed_url=feed_url)
    
    # Link speaker occurrences
    for speaker_label, profile_id in speaker_profiles.items():
        db.link_speaker_occurrence(
            transcription_id=trans_id1,
            temporary_label=speaker_label,
            profile_id=profile_id,
            confidence=0.95,
            is_verified=True
        )
    
    print(f"\nSaved episode 1 (ID: {trans_id1})")
    
    print("\n=== EPISODE 2: Speaker Recognition ===\n")
    
    # Simulate a second episode
    # In reality, this would be a different audio file
    print("Transcribing episode 2...")
    
    # Get new diarization (simulated)
    diarization2 = diarizer.diarize_file(str(audio_path), use_mock=False)
    result2 = transcriber.transcribe_file(str(audio_path), verbose=False)
    
    # Process diarization
    segments_list2 = [
        {
            "start": seg.start_ms / 1000.0,
            "end": seg.end_ms / 1000.0,
            "text": seg.text
        }
        for seg in result2.segments
    ]
    
    enhanced_segments2 = diarizer.align_with_transcription(diarization2, segments_list2)
    
    result2.has_speaker_labels = True
    result2.num_speakers = diarization2.num_speakers
    
    for i, seg_data in enumerate(enhanced_segments2):
        if i < len(result2.segments):
            result2.segments[i].speaker = seg_data.get("speaker", "UNKNOWN")
            result2.segments[i].confidence = seg_data.get("confidence", 0.0)
    
    result2.filename = "episode_002_interview.mp3"
    result2.file_hash = "ep2_hash_" + str(hash(result2.full_text))[:8]
    
    # Extract embeddings for episode 2
    embeddings2 = extractor.extract_embeddings_for_transcription(str(audio_path), result2)
    
    # Match speakers using SpeakerMatcher
    print("\nMatching speakers to existing profiles...")
    matcher = SpeakerMatcher(db, threshold=0.85, embedding_method="mock")
    
    speaker_mapping = {}
    for speaker_label, (embedding, duration, indices) in embeddings2.items():
        matches = matcher.find_best_match(embedding, feed_url=feed_url)
        
        if matches and matches[0][1] >= matcher.threshold:
            # Found a match!
            profile_id = matches[0][0]
            confidence = matches[0][1]
            profile = matches[0][2]
            print(f"  {speaker_label} → {profile['display_name']} (confidence: {confidence:.2f})")
            speaker_mapping[speaker_label] = (profile_id, confidence)
        else:
            # New speaker
            new_profile_id = db.create_speaker_profile(
                f"Guest (Episode 2)",
                feed_url=feed_url,
                canonical_label="GUEST"
            )
            db.add_speaker_embedding(
                new_profile_id,
                serialize_embedding(embedding),
                256,
                quality_score=0.85
            )
            print(f"  {speaker_label} → NEW SPEAKER (ID: {new_profile_id})")
            speaker_mapping[speaker_label] = (new_profile_id, 1.0)
    
    # Save episode 2
    trans_id2 = db.save_transcription(result2, feed_url=feed_url)
    
    # Link occurrences
    for speaker_label, (profile_id, confidence) in speaker_mapping.items():
        db.link_speaker_occurrence(
            transcription_id=trans_id2,
            temporary_label=speaker_label,
            profile_id=profile_id,
            confidence=confidence,
            is_verified=False
        )
    
    print(f"\nSaved episode 2 (ID: {trans_id2})")
    
    print("\n=== SPEAKER STATISTICS ===\n")
    
    # Get all profiles for the podcast
    profiles = db.get_speaker_profiles_for_feed(feed_url)
    
    for profile in profiles:
        print(f"Speaker: {profile['display_name']}")
        print(f"  Role: {profile['canonical_label']}")
        print(f"  Total appearances: {profile['total_appearances']}")
        print(f"  Total speaking time: {profile['total_duration_seconds']:.1f} seconds")
        
        # Get sample statements
        statements = db.get_speaker_statements(profile['id'])
        if statements:
            print(f"  Sample statements:")
            for stmt in statements[:3]:
                preview = stmt['text'][:60] + "..." if len(stmt['text']) > 60 else stmt['text']
                print(f"    - \"{preview}\"")
        print()
    
    print("=== QUERY EXAMPLES ===\n")
    
    # Find host statements across episodes
    host_profile = next((p for p in profiles if p['canonical_label'] == 'HOST'), None)
    if host_profile:
        print(f"All statements by {host_profile['display_name']}:")
        host_statements = db.get_speaker_statements(host_profile['id'])
        print(f"  Total segments: {len(host_statements)}")
        
        # Group by episode
        by_episode = {}
        for stmt in host_statements:
            filename = stmt['filename']
            if filename not in by_episode:
                by_episode[filename] = 0
            by_episode[filename] += 1
        
        for episode, count in by_episode.items():
            print(f"  {episode}: {count} segments")
    
    print("\n✅ Demo complete!")
    print(f"\nDatabase saved at: {db_path}")
    print("You can explore it further with the database tools.")


if __name__ == "__main__":
    main()