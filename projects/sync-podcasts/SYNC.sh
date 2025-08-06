#!/bin/bash

sync-podcasts --since 2025-06-01 --daemon --verbose --round-robin --model large --db ../../resources/fc/original-fc.db --feeds ../../resources/fc/feeds.toml --vector-store ../../resources/fc/grant_chroma_db --embedding-model nomic-embed-text:v1.5

#  -h, --help            show this help message and exit
#  --since, --date-threshold DATE_THRESHOLD
#                        Only process episodes published after this date (ISO8601 format)
#  --days DAYS           Process episodes from the last N days (alternative to --since)
#  --feeds FEEDS         Path to TOML file containing RSS feed URLs (default: ./resources/fc/feeds.toml)
#  --db DB               Path to beige-book database (default: ./resources/fc/fc.db)
#  --vector-store VECTOR_STORE
#                        Path to Grant vector store directory (default: ./projects/grant/grant_chroma_db)
#  --model {tiny,base,small,medium,large}
#                        Whisper model to use for transcription (default: tiny)
#  --round-robin         Process feeds in round-robin mode (newest episode from each feed before moving to next)
#  --daemon              Run in daemon mode (continuous processing with exponential backoff)
#  --ollama-base-url OLLAMA_BASE_URL
#                        Ollama API base URL (default: http://localhost:11434)
#  --verbose, -v         Enable verbose logging
