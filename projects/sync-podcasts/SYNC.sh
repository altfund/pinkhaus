#!/bin/bash

sync-podcasts --since 2025-02-28 --daemon --verbose --round-robin --model large --db ../../resources/fc/original-fc.db --feeds ../../resources/fc/feeds.toml --vector-store ../../resources/fc/grant_chroma_db --embedding-model nomic-embed-text:v1.5
