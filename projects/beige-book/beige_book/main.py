#!/usr/bin/env python3

import traceback
import whisper
import sys
import argparse
import os

def valid_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' is not a file")
    return path


def transcribe_file(filename, model_name="tiny"):
    """Transcribe an audio file using Whisper"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found")
    
    model = whisper.load_model(model_name)
    result = model.transcribe(audio=filename, verbose=True)
    return result["text"].strip()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("filename",
                        type=valid_path,
                        help="Audio file to transcribe")
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
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()