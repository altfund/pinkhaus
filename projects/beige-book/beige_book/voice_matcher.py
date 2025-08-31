#!/usr/bin/env python3
"""
Voice matching tool to identify speakers in new audio against existing database.

This tool extracts voice embeddings from a provided audio file and matches them
against all known speakers in the database, returning information about where
those voices have been heard before.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from pinkhaus_models.database import TranscriptionDatabase
from .voice_embeddings import VoiceEmbeddingExtractor
from .speaker_diarizer import SpeakerDiarizer
from .speaker_matcher import SpeakerMatcher


def find_voice_matches(
    db: TranscriptionDatabase,
    audio_path: str,
    hf_token: str,
    embedding_method: str = "speechbrain",
    threshold: float = 0.85,
    min_duration: float = 3.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Find matches for voices in the provided audio file.
    
    Args:
        db: Database containing existing speaker profiles
        audio_path: Path to audio file to analyze
        hf_token: Hugging Face token for diarization
        embedding_method: Method for extracting embeddings
        threshold: Similarity threshold for matching (0.0-1.0)
        min_duration: Minimum speech duration for reliable matching
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing:
            - speakers: Dict mapping temporary labels to profile matches
            - appearances: List of appearances with metadata
            - summary: Summary statistics
    """
    if verbose:
        print(f"Analyzing: {Path(audio_path).name}")
        
    # Step 1: Perform speaker diarization
    if verbose:
        print("  1. Performing speaker diarization...")
    
    diarizer = SpeakerDiarizer(auth_token=hf_token)
    diarization = diarizer.diarize_file(audio_path, use_mock=False)
    
    if verbose:
        print(f"     ✓ Found {diarization.num_speakers} speaker(s)")
        
    # Step 2: Create a mock transcription result with diarization info
    if verbose:
        print("  2. Creating transcription structure...")
        
    # We need a TranscriptionResult to use the embedding extractor
    # Create a minimal one with just speaker labels
    from .transcriber import TranscriptionResult, Segment
    import os
    
    result = TranscriptionResult()
    result.filename = os.path.basename(audio_path)
    result.file_hash = "temp"
    result.language = "unknown"
    result.full_text = ""
    result.has_speaker_labels = True
    result.num_speakers = diarization.num_speakers
    
    # Convert diarization segments to transcription segments
    for i, seg in enumerate(diarization.segments):
        result.segments.append(
            Segment(
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                text=f"[Speech segment {i+1}]",  # Placeholder text
                speaker=seg.speaker,
                confidence=seg.confidence
            )
        )
    
    # Step 3: Extract voice embeddings
    if verbose:
        print("  3. Extracting voice embeddings...")
        
    extractor = VoiceEmbeddingExtractor(method=embedding_method)
    embeddings = extractor.extract_embeddings_for_transcription(
        audio_path,
        result,
        min_duration=min_duration
    )
    
    if verbose:
        print(f"     ✓ Extracted embeddings for {len(embeddings)} speaker(s)")
        
    # Step 4: Match against database
    if verbose:
        print("  4. Matching against database...")
        
    matcher = SpeakerMatcher(db, threshold=threshold, embedding_method=embedding_method)
    
    results = {
        "speakers": {},
        "appearances": [],
        "summary": {
            "total_speakers": diarization.num_speakers,
            "matched_speakers": 0,
            "total_matches": 0,
            "unique_podcasts": set(),
            "unique_feeds": set()
        }
    }
    
    # Match each speaker
    for speaker_label, (embedding, duration, segment_indices) in embeddings.items():
        matches = matcher.find_best_match(embedding, feed_url=None)  # Search across all feeds
        
        if matches and matches[0][1] >= threshold:
            # Found matches
            profile_id = matches[0][0]
            confidence = matches[0][1]
            profile_data = matches[0][2]
            
            if verbose:
                print(f"     {speaker_label} → {profile_data['display_name']} (confidence: {confidence:.3f})")
            
            results["speakers"][speaker_label] = {
                "profile_id": profile_id,
                "display_name": profile_data["display_name"],
                "confidence": confidence,
                "duration": duration,
                "matches": []
            }
            
            results["summary"]["matched_speakers"] += 1
            
            # Get all appearances of this speaker
            appearances = get_speaker_appearances(db, profile_id)
            
            for appearance in appearances:
                results["speakers"][speaker_label]["matches"].append({
                    "filename": appearance["filename"],
                    "feed_url": appearance["feed_url"],
                    "feed_title": appearance["feed_item_title"],
                    "published": appearance["feed_item_published"],
                    "transcription_id": appearance["transcription_id"]
                })
                
                results["summary"]["unique_podcasts"].add(appearance["filename"])
                results["summary"]["unique_feeds"].add(appearance["feed_url"])
                results["summary"]["total_matches"] += 1
                
                results["appearances"].append({
                    "speaker": speaker_label,
                    "profile": profile_data["display_name"],
                    "confidence": confidence,
                    **appearance
                })
        else:
            # No match found
            if verbose:
                print(f"     {speaker_label} → No match found")
                
            results["speakers"][speaker_label] = {
                "profile_id": None,
                "display_name": "Unknown",
                "confidence": 0.0,
                "duration": duration,
                "matches": []
            }
    
    # Convert sets to counts for JSON serialization
    results["summary"]["unique_podcasts"] = len(results["summary"]["unique_podcasts"])
    results["summary"]["unique_feeds"] = len(results["summary"]["unique_feeds"])
    
    return results


def get_speaker_appearances(db: TranscriptionDatabase, profile_id: int) -> List[Dict[str, Any]]:
    """
    Get all appearances of a speaker profile in the database.
    
    Args:
        db: Database instance
        profile_id: Speaker profile ID
        
    Returns:
        List of appearance dictionaries with metadata
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        
        # Get all transcriptions where this speaker appears
        cursor.execute("""
            SELECT DISTINCT 
                tm.id as transcription_id,
                tm.filename,
                tm.feed_url,
                tm.feed_item_title,
                tm.feed_item_published,
                tm.created_at,
                so.temporary_label,
                so.confidence
            FROM speaker_occurrences so
            JOIN transcription_metadata tm ON so.transcription_id = tm.id
            WHERE so.profile_id = ?
            ORDER BY tm.feed_item_published DESC, tm.created_at DESC
        """, (profile_id,))
        
        appearances = []
        for row in cursor.fetchall():
            appearances.append({
                "transcription_id": row[0],
                "filename": row[1],
                "feed_url": row[2],
                "feed_item_title": row[3],
                "feed_item_published": row[4],
                "created_at": row[5],
                "temporary_label": row[6],
                "match_confidence": row[7]
            })
            
        return appearances


def format_results(results: Dict[str, Any], format_type: str = "summary") -> str:
    """
    Format matching results for display.
    
    Args:
        results: Results from find_voice_matches
        format_type: Output format (summary, detailed, json)
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        import json
        return json.dumps(results, indent=2, default=str)
        
    output = []
    output.append("\n=== Voice Matching Results ===\n")
    
    # Summary
    summary = results["summary"]
    output.append(f"Total speakers detected: {summary['total_speakers']}")
    output.append(f"Speakers matched: {summary['matched_speakers']}")
    output.append(f"Total appearances: {summary['total_matches']}")
    output.append(f"Unique podcasts: {summary['unique_podcasts']}")
    output.append(f"Unique feeds: {summary['unique_feeds']}")
    output.append("")
    
    # Speaker details
    for speaker_label, speaker_data in results["speakers"].items():
        output.append(f"\n{speaker_label}:")
        output.append(f"  Profile: {speaker_data['display_name']}")
        output.append(f"  Confidence: {speaker_data['confidence']:.3f}")
        output.append(f"  Duration: {speaker_data['duration']:.1f}s")
        
        if speaker_data["matches"]:
            output.append(f"  Found in {len(speaker_data['matches'])} recordings:")
            
            if format_type == "detailed":
                # Show all matches
                for match in speaker_data["matches"]:
                    output.append(f"    - {match['filename']}")
                    if match["feed_title"]:
                        output.append(f"      Title: {match['feed_title']}")
                    if match["feed_url"]:
                        output.append(f"      Feed: {match['feed_url']}")
                    if match["published"]:
                        output.append(f"      Published: {match['published']}")
            else:
                # Show summary
                # Group by feed
                by_feed = defaultdict(list)
                for match in speaker_data["matches"]:
                    feed = match["feed_url"] or "Local files"
                    by_feed[feed].append(match)
                
                for feed, feed_matches in by_feed.items():
                    output.append(f"    - {feed}: {len(feed_matches)} episodes")
                    if format_type == "summary" and len(feed_matches) <= 3:
                        for match in feed_matches[:3]:
                            output.append(f"      • {match['filename']}")
                    elif len(feed_matches) > 3:
                        output.append(f"      • ... and {len(feed_matches) - 3} more")
        else:
            output.append("  No matches found in database")
    
    return "\n".join(output)


def main():
    """Main entry point for voice matching CLI."""
    parser = argparse.ArgumentParser(
        description="Match voices in audio against existing speaker database"
    )
    parser.add_argument(
        "audio_path",
        help="Path to audio file to analyze"
    )
    parser.add_argument(
        "db_path",
        help="Path to database containing speaker profiles"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for matching (0.0-1.0, default: 0.85)"
    )
    parser.add_argument(
        "--embedding-method",
        choices=["speechbrain", "pyannote", "mock"],
        default="speechbrain",
        help="Voice embedding extraction method"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum speech duration for reliable matching (seconds)"
    )
    parser.add_argument(
        "--format",
        choices=["summary", "detailed", "json"],
        default="summary",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.audio_path):
        parser.error(f"Audio file not found: {args.audio_path}")
        
    if not os.path.exists(args.db_path):
        parser.error(f"Database not found: {args.db_path}")
        
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        parser.error("HF_TOKEN environment variable is required for speaker diarization")
        
    try:
        # Initialize database
        db = TranscriptionDatabase(args.db_path)
        
        # Find matches
        results = find_voice_matches(
            db=db,
            audio_path=args.audio_path,
            hf_token=hf_token,
            embedding_method=args.embedding_method,
            threshold=args.threshold,
            min_duration=args.min_duration,
            verbose=not args.quiet
        )
        
        # Format output
        formatted = format_results(results, args.format)
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                f.write(formatted)
            if not args.quiet:
                print(f"\nResults written to: {args.output}")
        else:
            print(formatted)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()