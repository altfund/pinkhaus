#!/bin/bash
# Test voice matching with your own audio

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo "Please set: export HF_TOKEN='your-token-here'"
    exit 1
fi

# Check if audio file provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-your-audio-file> [database.db]"
    echo ""
    echo "Examples:"
    echo "  $0 my-voice.mp3                    # Test against new database"
    echo "  $0 my-voice.mp3 existing.db        # Test against existing database"
    exit 1
fi

AUDIO_FILE="$1"
DB_FILE="${2:-voice-test.db}"

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# If database doesn't exist, create it with some test data
if [ ! -f "$DB_FILE" ]; then
    echo "=== Creating new database with test data ==="
    echo "Processing harvard.wav as baseline..."
    
    HARVARD="/Users/price/development/ai-projects/pinkhaus2/resources/audio/harvard.wav"
    
    if [ -f "$HARVARD" ]; then
        beige-book "$HARVARD" \
            --db-path "$DB_FILE" \
            --format sqlite \
            --model tiny \
            --diarize \
            --speaker-profiles \
            --embedding-method speechbrain
        
        echo "✓ Database created with baseline speaker"
    else
        echo "Warning: harvard.wav not found, creating empty database"
        # Just create empty tables
        echo "import sys; sys.path.append('.'); from pinkhaus_models.database import TranscriptionDatabase; db = TranscriptionDatabase('$DB_FILE'); db.create_tables(); db.create_speaker_identity_tables()" | python
    fi
    echo ""
fi

# Now test the provided audio
echo "=== Testing your audio against database ==="
echo "Audio file: $AUDIO_FILE"
echo "Database: $DB_FILE"
echo ""

beige-book-match-voice "$AUDIO_FILE" "$DB_FILE" \
    --embedding-method speechbrain \
    --threshold 0.85 \
    --format detailed

echo ""
echo "=== Done ==="
echo ""
echo "Tips:"
echo "- If no matches found, try lowering threshold: --threshold 0.75"
echo "- To add this audio to the database for future matching:"
echo "  beige-book '$AUDIO_FILE' --db-path '$DB_FILE' --format sqlite --diarize --speaker-profiles"