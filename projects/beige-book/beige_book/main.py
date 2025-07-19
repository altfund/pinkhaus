#!/usr/bin/env python3

import whisper
import sys
import argparse


def transcribe_file(filename, model_name="tiny"):
    """Transcribe an audio file using Whisper"""
    model = whisper.load_model(model_name)
    result = model.transcribe(filename)
    return result["text"].strip()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("filename", help="Audio file to transcribe")
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use"
    )
    
    args = parser.parse_args()
    
    try:
        text = transcribe_file(args.filename, args.model)
        print(text)
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()