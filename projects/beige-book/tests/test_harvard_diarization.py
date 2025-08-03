#!/usr/bin/env python3
"""
Test script for speaker diarization using harvard.wav file.
Creates a comprehensive database with both transcription and diarization data.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from beige_book.transcriber import AudioTranscriber
from beige_book.speaker_diarizer import SpeakerDiarizer, create_speaker_aware_transcription
from beige_book.database import TranscriptionDatabase


def test_harvard_with_diarization():
    """Test transcription and diarization on harvard.wav file."""
    
    # Path to harvard.wav
    harvard_path = "../../../resources/audio/harvard.wav"
    
    if not os.path.exists(harvard_path):
        print(f"Error: harvard.wav not found at {harvard_path}")
        print("Please check the path or provide the correct location.")
        return
    
    print(f"Using audio file: {os.path.abspath(harvard_path)}")
    
    # Initialize components
    print("\n1. Initializing components...")
    transcriber = AudioTranscriber(model_name="tiny")
    
    # Check if we have HF token for real diarization
    hf_token = os.getenv("HF_TOKEN")
    use_real_diarization = bool(hf_token)
    
    if use_real_diarization:
        print("   ✓ Using REAL speaker diarization (HF token found)")
    else:
        print("   ✓ Using MOCK speaker diarization (no HF token)")
        print("   Note: Set HF_TOKEN environment variable for real diarization")
    
    # Perform transcription with diarization
    print("\n2. Transcribing with speaker diarization...")
    result = transcriber.transcribe_file(
        harvard_path,
        enable_diarization=True,
        hf_token=hf_token
    )
    
    # Get result as dictionary
    result_dict = result.to_dict()
    
    # Display summary
    print("\n3. Transcription Summary:")
    print(f"   - File: {result_dict['filename']}")
    print(f"   - SHA256: {result_dict['file_hash'][:16]}...")
    print(f"   - Language: {result_dict['language']}")
    print(f"   - Duration: ~{len(result_dict['segments'])} segments")
    print(f"   - Speakers: {result_dict.get('num_speakers', 'Unknown')}")
    print(f"   - Has speaker labels: {result_dict.get('has_speaker_labels', False)}")
    
    # Show sample segments with speakers
    print("\n4. Sample segments with speaker labels:")
    for i, seg in enumerate(result_dict['segments'][:5]):
        speaker = seg.get('speaker', 'UNKNOWN')
        print(f"   [{speaker}] {seg['start']} - {seg['end']}")
        print(f"     \"{seg['text']}\"")
    
    # Save outputs
    print("\n5. Saving outputs...")
    
    # JSON with full details
    json_path = "harvard_diarization_result.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"   ✓ JSON saved: {json_path}")
    
    # CSV with speakers
    csv_path = "harvard_diarization_result.csv"
    with open(csv_path, "w") as f:
        f.write(result.to_csv())
    print(f"   ✓ CSV saved: {csv_path}")
    
    # Create enhanced database
    print("\n6. Creating enhanced database with speaker information...")
    db_path = "harvard_diarization.db"
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database with enhanced schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create enhanced tables
    cursor.execute("""
        CREATE TABLE transcription_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            language TEXT,
            created_at INTEGER,
            full_text TEXT,
            num_speakers INTEGER,
            has_speaker_labels BOOLEAN,
            diarization_mode TEXT
        )
    """)
    
    # Create speakers table
    cursor.execute("""
        CREATE TABLE speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata_id INTEGER NOT NULL,
            speaker_label TEXT NOT NULL,
            total_segments INTEGER DEFAULT 0,
            total_duration_ms INTEGER DEFAULT 0,
            first_appearance_ms INTEGER,
            last_appearance_ms INTEGER,
            FOREIGN KEY (metadata_id) REFERENCES transcription_metadata(id),
            UNIQUE(metadata_id, speaker_label)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE transcription_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata_id INTEGER NOT NULL,
            speaker_id INTEGER,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL,
            speaker_confidence REAL,
            FOREIGN KEY (metadata_id) REFERENCES transcription_metadata(id),
            FOREIGN KEY (speaker_id) REFERENCES speakers(id)
        )
    """)
    
    # Create indexes for performance
    cursor.execute("""
        CREATE INDEX idx_segments_metadata ON transcription_segments(metadata_id)
    """)
    cursor.execute("""
        CREATE INDEX idx_segments_speaker ON transcription_segments(speaker_id)
    """)
    cursor.execute("""
        CREATE INDEX idx_speakers_metadata ON speakers(metadata_id)
    """)
    
    # Insert metadata
    cursor.execute("""
        INSERT INTO transcription_metadata 
        (filename, file_hash, language, created_at, full_text, 
         num_speakers, has_speaker_labels, diarization_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result_dict['filename'],
        result_dict['file_hash'],
        result_dict['language'],
        result_dict.get('created_at', int(datetime.now().timestamp())),
        result_dict['full_text'],
        result_dict.get('num_speakers'),
        result_dict.get('has_speaker_labels', False),
        'real' if use_real_diarization else 'mock'
    ))
    
    metadata_id = cursor.lastrowid
    
    # Create speaker entries and build lookup
    speaker_ids = {}
    unique_speakers = set(seg.get('speaker') for seg in result_dict['segments'] if seg.get('speaker'))
    
    for speaker_label in unique_speakers:
        cursor.execute("""
            INSERT INTO speakers (metadata_id, speaker_label)
            VALUES (?, ?)
        """, (metadata_id, speaker_label))
        speaker_ids[speaker_label] = cursor.lastrowid
    
    # Insert segments with speaker foreign keys
    for seg in result_dict['segments']:
        # Parse time to milliseconds
        start_ms = parse_time_to_ms(seg['start'])
        end_ms = parse_time_to_ms(seg['end'])
        speaker_label = seg.get('speaker')
        speaker_id = speaker_ids.get(speaker_label) if speaker_label else None
        
        cursor.execute("""
            INSERT INTO transcription_segments 
            (metadata_id, speaker_id, start_ms, end_ms, text, speaker_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            metadata_id,
            speaker_id,
            start_ms,
            end_ms,
            seg['text'],
            seg.get('confidence')
        ))
    
    # Update speaker statistics
    cursor.execute("""
        UPDATE speakers
        SET total_segments = (
            SELECT COUNT(*) FROM transcription_segments 
            WHERE speaker_id = speakers.id
        ),
        total_duration_ms = (
            SELECT SUM(end_ms - start_ms) FROM transcription_segments 
            WHERE speaker_id = speakers.id
        ),
        first_appearance_ms = (
            SELECT MIN(start_ms) FROM transcription_segments 
            WHERE speaker_id = speakers.id
        ),
        last_appearance_ms = (
            SELECT MAX(end_ms) FROM transcription_segments 
            WHERE speaker_id = speakers.id
        )
        WHERE metadata_id = ?
    """, (metadata_id,))
    
    conn.commit()
    
    print(f"   ✓ Database created: {db_path}")
    
    # Show database statistics
    print("\n7. Database Statistics:")
    
    # Total segments
    cursor.execute("SELECT COUNT(*) FROM transcription_segments WHERE metadata_id = ?", (metadata_id,))
    total_segments = cursor.fetchone()[0]
    print(f"   - Total segments: {total_segments}")
    
    # Speaker statistics
    cursor.execute("""
        SELECT speaker_label, total_segments, total_duration_ms 
        FROM speakers 
        WHERE metadata_id = ?
        ORDER BY total_duration_ms DESC
    """, (metadata_id,))
    
    speakers = cursor.fetchall()
    if speakers:
        print("   - Speaker breakdown:")
        for speaker, segments, duration_ms in speakers:
            duration_sec = duration_ms / 1000.0
            print(f"     • {speaker}: {segments} segments, {duration_sec:.1f}s total")
    
    # Sample queries
    print("\n8. Sample Database Queries:")
    
    # Query 1: Longest utterances
    print("   - Top 3 longest utterances:")
    cursor.execute("""
        SELECT sp.speaker_label, seg.text, (seg.end_ms - seg.start_ms) as duration_ms
        FROM transcription_segments seg
        LEFT JOIN speakers sp ON seg.speaker_id = sp.id
        WHERE seg.metadata_id = ?
        ORDER BY duration_ms DESC
        LIMIT 3
    """, (metadata_id,))
    
    for speaker, text, duration_ms in cursor.fetchall():
        print(f"     • [{speaker}] {duration_ms}ms: \"{text[:50]}...\"")
    
    # Query 2: Speaker transitions
    print("\n   - Speaker transitions (first 5):")
    cursor.execute("""
        SELECT 
            sp1.speaker_label as speaker1,
            sp2.speaker_label as speaker2,
            s1.text as text1,
            s2.text as text2
        FROM transcription_segments s1
        JOIN transcription_segments s2 ON s2.id = s1.id + 1
        LEFT JOIN speakers sp1 ON s1.speaker_id = sp1.id
        LEFT JOIN speakers sp2 ON s2.speaker_id = sp2.id
        WHERE s1.metadata_id = ? 
          AND s2.metadata_id = ?
          AND s1.speaker_id != s2.speaker_id
        LIMIT 5
    """, (metadata_id, metadata_id))
    
    transitions = cursor.fetchall()
    for speaker1, speaker2, text1, text2 in transitions:
        print(f"     • {speaker1} → {speaker2}")
        print(f"       \"{text1[:40]}...\"")
        print(f"       \"{text2[:40]}...\"")
    
    conn.close()
    
    print("\n✅ Test complete! Check the generated files:")
    print(f"   - {json_path} (JSON with all data)")
    print(f"   - {csv_path} (CSV with speaker columns)")
    print(f"   - {db_path} (SQLite database with enhanced schema)")


def parse_time_to_ms(time_str):
    """Parse HH:MM:SS.mmm to milliseconds."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return int((hours * 3600 + minutes * 60 + seconds) * 1000)


if __name__ == "__main__":
    print("Harvard Audio Diarization Test")
    print("=" * 50)
    
    # Show HF token info
    print("\nHugging Face Token Requirements:")
    print("- Scope: 'read' permission is sufficient")
    print("- Usage: Access to pyannote/speaker-diarization models")
    print("- Setup: Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1")
    
    if not os.getenv("HF_TOKEN"):
        print("\n⚠️  No HF_TOKEN found - will use mock diarization")
        print("   To use real diarization: export HF_TOKEN='hf_...'")
    
    print("\nStarting test...\n")
    
    try:
        test_harvard_with_diarization()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()